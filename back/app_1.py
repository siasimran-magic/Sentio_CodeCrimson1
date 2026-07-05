# --- Import necessary libraries ---
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pdf2image import convert_from_bytes
import pytesseract
import re
import io
import os
import traceback
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import ast
from haversine import haversine, Unit

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
CORS(app)

# --- Load ML Model (if available) ---
try:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
    
    with open('model_vitals.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    with open('scaler_vitals.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('vital_signs_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    MODEL_LOADED = True
    print("✅ ML Model loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠️ ML Model not found. Using rule-based risk assessment. error: {e}")

# --- Load Blood Bank Data ---
try:
    blood_banks_df = pd.read_csv('blood_banks_formatted.csv')
    blood_banks_df = blood_banks_df.rename(columns={'O': 'O+', 'A': 'A+', 'B': 'B+', 'AB': 'AB+'})
    
    blood_types = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
    for btype in blood_types:
        if btype not in blood_banks_df.columns:
            blood_banks_df[btype] = 0
        blood_banks_df[btype] = pd.to_numeric(blood_banks_df[btype], errors='coerce').fillna(0).astype(int)
    
    blood_banks_df['expiry_dates'] = blood_banks_df['expiry_dates'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    blood_banks_df['lat'] = pd.to_numeric(blood_banks_df['lat'], errors='coerce')
    blood_banks_df['lon'] = pd.to_numeric(blood_banks_df['lon'], errors='coerce')
    
    BLOOD_BANKS_LOADED = True
    print("✅ Blood bank data loaded successfully!")
except Exception as e:
    BLOOD_BANKS_LOADED = False
    blood_banks_df = None
    print(f"⚠️ Blood bank data not found: {e}")

# --- Load Patient Data ---
try:
    patients_df = pd.read_csv('patients.csv')
    patients_df['name'] = patients_df['name'].fillna(lambda x: f'Patient {x.index + 1}')
    PATIENTS_LOADED = True
    print("✅ Patient data loaded successfully!")
except Exception as e:
    PATIENTS_LOADED = False
    patients_df = None
    print(f"⚠️ Patient data not found: {e}")
def preprocess_for_ocr(image):
    """Improve OCR accuracy on photographed/scanned medical reports by
    normalizing uneven lighting and binarizing text before passing to Tesseract."""
    image = image.convert('L')                     # grayscale
    image = ImageOps.autocontrast(image, cutoff=2)  # fix shadows/glare
    threshold = 150
    image = image.point(lambda p: 255 if p > threshold else 0)  # clean black/white text
    return image
# --- OCR Function ---
def extract_text_from_file(uploaded_file):
    """Extract text from PDF or image using OCR."""
    try:
        filename = uploaded_file.filename.lower()
        file_bytes = uploaded_file.read()

        if not file_bytes:
            raise ValueError("Empty file received")

        text = ""

        if filename.endswith(".pdf"):
            pages = convert_from_bytes(file_bytes, size=(1500, None), dpi=200)
            for page in pages:
                page = preprocess_for_ocr(page)
                text += pytesseract.image_to_string(page, config='--psm 6')
        else:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            width, height = image.size
            max_width = 1500
            if width > max_width:
                aspect_ratio = height / width
                new_height = int(max_width * aspect_ratio)
                image = image.resize((max_width, new_height), Image.LANCZOS)
            image = preprocess_for_ocr(image)
            text = pytesseract.image_to_string(image, config='--psm 6')

    except Exception as e:
        print(f"Error during OCR processing: {e}")
        traceback.print_exc()
        raise Exception(f"OCR processing failed: {str(e)}")

    return text

# --- Extraction Logic ---
def extract_parameters(text):
    """Extract key PPH parameters using regex."""
    results = {}
    text_normalized = ' '.join(text.split())

    patterns = {
        "Patient Name": r'(?i)(?:Patient\s*Name|Name|Mother\s*Name)\s*[:\s]*([\w\s]+?)(?=\n|Patient|Age|DOB|$)',
        "Patient ID": r'(?i)(?:Patient\s*ID|ID\s*No|Registration\s*No|MRN)\s*[:\s]*([A-Z0-9-]+)',
        "Blood Type": r'(?i)(?:Blood\s*Type|Blood\s*Group)\s*[:\s]*([ABO]+[+-])',
        "Hospital": r'(?i)(?:Hospital|Facility|Health\s*Center|Wellness\s*Hospital)\s*[:\s]*([\w\s,]+?)(?=\n|Patient|$)',
        "Expected Delivery Date": r'(?i)(?:EDD|Expected\s*Delivery\s*Date|Due\s*Date)\s*[:\s]*([\d/-]+)',
        "Age (years)": r'(?i)Age\s*[:\s]*(\d+)\s*years?',
        "Parity": r'(?i)Parity\s*[:\s]*[Gg](\d+)[Pp](\d+)',
        "History of PPH": r'(?i)History\s*of\s*(?:Postpartum\s*)?Hemorrhage\s*[:\s]*(Yes|No)',
        "History of Previous Cesarean": r'(?i)(?:Previous\s*(?:Cesarean|Caesarean|C[- ]?section)|History\s*of\s*(?:CS|C[- ]?section))\s*[:\s]*(Yes|No)',
        "Pre-pregnancy BMI": r'(?i)Pre[- ]?Pregnancy\s*BMI\s*[:\s]*([\d.]+)\s*(?:kg/m[²2])?',
        "Hemoglobin (g/dL)": r'(?i)Hemoglobin\s*(?:level|Level)?\s*[:\s]*([\d.]+)\s*(?:g/dL)?',
        "Platelet Count (x10^3/μL)": r'(?i)Platelet\s*Count\s*[:\s]*([\d,]+)\s*(?:/mm[³3])?',
        "Systolic Blood Pressure (mmHg)": r'(?i)Blood\s*Pressure\s*[:\s]*(\d+)/\d+\s*mmHg',
        "Diastolic Blood Pressure (mmHg)": r'(?i)Blood\s*Pressure\s*[:\s]*\d+/(\d+)\s*mmHg',
        "Heart Rate (bpm)": r'(?i)Heart\s*Rate\s*[:\s]*(\d+)\s*bpm',
        "Placenta Previa": r'(?i)Placenta\s*Previa\s*[:\s]*(Yes|No)',
        "Multiple Pregnancy": r'(?i)Multiple\s*Pregnancy\s*[:\s]*(?:sal\s*[:\s]*)?(Yes|No)',
        "Estimated Fetal Weight (kg)": r'(?i)Estimated\s*Fetal\s*Weight\s*[:\s]*([\d.]+)\s*kg',
        "Gestational Age (weeks)": r'(?i)Gestational\s*Age\s*[:\s]*(\d+)\s*weeks?',
        "Mode of Delivery": r'(?i)Mode\s*of\s*Delivery\s*[:\s]*(C[- ]?Section|Cesarean|Caesarean|Vaginal|Normal)',
        "Induced Labor": r'(?i)Induced\s*Labor\s*[:\s]*(Yes|No)',
        "Anesthesia Type": r'(?i)Anesthesia\s*type\s*[:\s]*(General|Spinal|Epidural|Local)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            if key == "Parity":
                results[key] = match.group(2).strip()
            elif key == "Platelet Count (x10^3/μL)":
                results[key] = match.group(1).replace(',', '').strip()
            elif key == "Mode of Delivery":
                value = match.group(1).strip()
                results[key] = "C-Section" if re.search(r'(?i)c[- ]?section|cesarean|caesarean', value) else value
            else:
                results[key] = match.group(1).strip()

    # Special handling
    if "Multiple Pregnancy" not in results:
        sal_match = re.search(r'(?i)sal\s*[:\s]*(Yes|No)', text)
        if sal_match:
            results["Multiple Pregnancy"] = sal_match.group(1)

    if "Gestational Age (weeks)" not in results:
        ga_match = re.search(r'(?i)(?:GA|Gestational\s*Age)\s*[:\s]*(\d+)\s*w(?:eeks?|ks?)', text)
        if ga_match:
            results["Gestational Age (weeks)"] = ga_match.group(1)

    if "Systolic Blood Pressure (mmHg)" not in results or "Diastolic Blood Pressure (mmHg)" not in results:
        bp_match = re.search(r'(?i)(?:BP|Blood\s*Pressure)\s*[:\s]*(\d+)/(\d+)', text)
        if bp_match:
            results["Systolic Blood Pressure (mmHg)"] = bp_match.group(1)
            results["Diastolic Blood Pressure (mmHg)"] = bp_match.group(2)

    return results

# --- Risk Calculation ---
def calculate_pph_risk(findings):
    """Calculate PPH risk using ML model or rule-based approach."""

    # Extract numeric values
    def safe_float(value, default=0):
        try:
            return float(str(value).replace(',', ''))
        except:
            return default

    age = safe_float(findings.get('Age (years)', 0))
    systolic_bp = safe_float(findings.get('Systolic Blood Pressure (mmHg)', 0))
    diastolic_bp = safe_float(findings.get('Diastolic Blood Pressure (mmHg)', 0))
    heart_rate = safe_float(findings.get('Heart Rate (bpm)', 0))
    hemoglobin = safe_float(findings.get('Hemoglobin (g/dL)', 0))
    bmi = safe_float(findings.get('Pre-pregnancy BMI', 0))

    # Create feature vector for ML model
    if MODEL_LOADED:
        try:
            # Create features matching training data
            age_risk = 1 if (age < 20 or age > 35) else 0
            hypertension = 1 if (systolic_bp >= 140 or diastolic_bp >= 90) else 0
            bp_product = systolic_bp * diastolic_bp
            hr_abnormal = 1 if (heart_rate < 60 or heart_rate > 100) else 0

            # Build feature dict
            features_dict = {
                'age': age,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'heart_rate': heart_rate,
                'age_risk': age_risk,
                'hypertension': hypertension,
                'bp_product': bp_product,
                'hr_abnormal': hr_abnormal
            }

            if bmi > 0:
                features_dict['bmi'] = bmi
                features_dict['obesity'] = 1 if bmi >= 30 else 0

            # Create feature array in correct order
            feature_array = []
            for feat_name in feature_names:
                feature_array.append(features_dict.get(feat_name, 0))

            # Scale and predict using DataFrame to preserve feature names
            X = pd.DataFrame([feature_array], columns=feature_names)
            X_scaled = scaler.transform(X)
            risk_prob = float(ml_model.predict_proba(X_scaled)[0][1] * 100)

            # Get feature importance
            feature_importance = dict(zip(feature_names, ml_model.feature_importances_))
            top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

            return {
                'riskScore': round(risk_prob, 1),
                'riskLevel': 'HIGH' if risk_prob > 60 else 'MODERATE' if risk_prob > 30 else 'LOW',
                'modelType': 'ML',
                'topRiskFactors': [{'factor': f[0].replace('_', ' ').title(), 'importance': float(round(f[1], 3))} for f in top_factors]
            }
        except Exception as e:
            print(f"ML prediction error: {e}")
            traceback.print_exc()

    # Rule-based fallback
    risk_score = 0
    risk_factors = []

    # Age risk (20 points)
    if age < 18:
        risk_score += 20
        risk_factors.append({'factor': 'Very Young Age (<18)', 'points': 20})
    elif age < 20 or age > 35:
        risk_score += 15
        risk_factors.append({'factor': 'Age Risk (<20 or >35)', 'points': 15})

    # Hypertension (25 points)
    if systolic_bp >= 160 or diastolic_bp >= 110:
        risk_score += 25
        risk_factors.append({'factor': 'Severe Hypertension', 'points': 25})
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        risk_score += 20
        risk_factors.append({'factor': 'Hypertension', 'points': 20})

    # Anemia (20 points)
    if hemoglobin > 0 and hemoglobin < 11:
        risk_score += 20
        risk_factors.append({'factor': 'Anemia (Hb < 11 g/dL)', 'points': 20})
    elif hemoglobin > 0 and hemoglobin < 9:
        risk_score += 25
        risk_factors.append({'factor': 'Severe Anemia (Hb < 9 g/dL)', 'points': 25})

    # History of PPH (25 points)
    if findings.get('History of PPH', '').lower() == 'yes':
        risk_score += 25
        risk_factors.append({'factor': 'Previous PPH History', 'points': 25})

    # Placenta Previa (30 points)
    if findings.get('Placenta Previa', '').lower() == 'yes':
        risk_score += 30
        risk_factors.append({'factor': 'Placenta Previa', 'points': 30})

    # Multiple pregnancy (15 points)
    if findings.get('Multiple Pregnancy', '').lower() == 'yes':
        risk_score += 15
        risk_factors.append({'factor': 'Multiple Pregnancy', 'points': 15})

    # C-Section history (10 points)
    if findings.get('History of Previous Cesarean', '').lower() == 'yes':
        risk_score += 10
        risk_factors.append({'factor': 'Previous C-Section', 'points': 10})

    # BMI (15 points)
    if bmi >= 35:
        risk_score += 15
        risk_factors.append({'factor': 'Severe Obesity (BMI ≥35)', 'points': 15})
    elif bmi >= 30:
        risk_score += 10
        risk_factors.append({'factor': 'Obesity (BMI ≥30)', 'points': 10})

    risk_score = min(risk_score, 100)

    return {
        'riskScore': risk_score,
        'riskLevel': 'HIGH' if risk_score > 60 else 'MODERATE' if risk_score > 30 else 'LOW',
        'modelType': 'Rule-Based',
        'topRiskFactors': sorted(risk_factors, key=lambda x: x['points'], reverse=True)[:5]
    }

# --- API Route ---
@app.route("/api/ocr", methods=["POST", "OPTIONS"])
def ocr_api():
    """Handle OCR API request with risk assessment."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Invalid file type"}), 400

        text = extract_text_from_file(file)
        if not text or len(text.strip()) < 10:
            return jsonify({
                "error": "Could not extract meaningful text",
                "extractedText": text
            }), 400

        findings = extract_parameters(text)

        # Calculate risk
        risk_assessment = calculate_pph_risk(findings)

        return jsonify({
            "success": True,
            "documentType": "Maternal Health Report",
            "keyFindings": findings,
            "riskAssessment": risk_assessment,
            "extractedText": text[:2000],
            "metadata": {
                "version": "4.0-ML-Integrated",
                "parameters": list(findings.keys()),
                "filename": file.filename,
                "textLength": len(text),
                "mlModelLoaded": MODEL_LOADED
            }
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

# --- PPH Risk Analysis Endpoint ---
@app.route("/api/risk-analysis", methods=["POST", "OPTIONS"])
def risk_analysis():
    """Direct PPH risk analysis endpoint (no OCR required)."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        global patients_df, PATIENTS_LOADED
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate and extract patient data
        patient_data = {
            "Age (years)": data.get("age"),
            "Systolic Blood Pressure (mmHg)": data.get("systolic_bp"),
            "Diastolic Blood Pressure (mmHg)": data.get("diastolic_bp"),
            "Heart Rate (bpm)": data.get("heart_rate"),
            "Hemoglobin (g/dL)": data.get("hemoglobin"),
            "Pre-pregnancy BMI": data.get("bmi"),
            "History of PPH": data.get("history_pph"),
            "Placenta Previa": data.get("placenta_previa"),
            "Multiple Pregnancy": data.get("multiple_pregnancy"),
            "History of Previous Cesarean": data.get("previous_cesarean"),
            "Parity": data.get("parity"),
            "Platelet Count (x10^3/μL)": data.get("platelet_count"),
            "Gestational Age (weeks)": data.get("gestational_age"),
            "Mode of Delivery": data.get("mode_of_delivery"),
            "Estimated Fetal Weight (kg)": data.get("fetal_weight")
        }

        # Remove None values
        patient_data = {k: v for k, v in patient_data.items() if v is not None}

        # Check if we have minimum required data
        required_fields = ["Age (years)", "Systolic Blood Pressure (mmHg)", 
                          "Diastolic Blood Pressure (mmHg)", "Heart Rate (bpm)"]
        missing_fields = [field for field in required_fields if field not in patient_data or patient_data[field] == ""]
        
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing": missing_fields,
                "required": required_fields
            }), 400

        # Calculate risk
        risk_assessment = calculate_pph_risk(patient_data)

        # --- Save patient to database ---
        patient_saved = False
        patient_details = {}
        
        try:
            # Get patient name from data (could be from OCR or manual entry)
            patient_name = data.get("name") or data.get("patient_name") or data.get("Patient Name")
            
            # Initialize or load patients dataframe
            if patients_df is None or patients_df.empty:
                # Create new dataframe with headers and explicit dtypes
                patients_df = pd.DataFrame({
                    'patient_id': pd.Series(dtype='str'),
                    'name': pd.Series(dtype='str'),
                    'age': pd.Series(dtype='float'),
                    'systolic_bp': pd.Series(dtype='float'),
                    'diastolic_bp': pd.Series(dtype='float'),
                    'heart_rate': pd.Series(dtype='float'),
                    'hemoglobin': pd.Series(dtype='float'),
                    'bmi': pd.Series(dtype='float'),
                    'expected_delivery_date': pd.Series(dtype='str'),
                    'status': pd.Series(dtype='str'),
                    'trend': pd.Series(dtype='str'),
                    'risk_score': pd.Series(dtype='float')
                })
                PATIENTS_LOADED = True
            
            # Check if patient already exists (by name if provided)
            patient_exists = False
            if patient_name and not patients_df.empty:
                patient_exists = (patients_df['name'].str.lower() == patient_name.lower()).any()
            
            if not patient_exists:
                # Count patients without proper names (Patient_X format)
                unnamed_count = 0
                if not patients_df.empty:
                    unnamed_count = patients_df['name'].str.match(r'^Patient_\d+$', case=False, na=False).sum()
                
                # Generate patient name if not provided
                if not patient_name:
                    patient_name = f"Patient_{unnamed_count + 1}"
                
                # Generate patient ID
                total_patients = len(patients_df)
                patient_id = f"MH2024-{total_patients + 1:03d}"
                
                # Prepare new patient row with explicit type conversions
                new_patient = {
                    'patient_id': str(patient_id),
                    'name': str(patient_name),
                    'age': float(patient_data.get('Age (years)')) if patient_data.get('Age (years)') is not None else None,
                    'systolic_bp': float(patient_data.get('Systolic Blood Pressure (mmHg)')) if patient_data.get('Systolic Blood Pressure (mmHg)') is not None else None,
                    'diastolic_bp': float(patient_data.get('Diastolic Blood Pressure (mmHg)')) if patient_data.get('Diastolic Blood Pressure (mmHg)') is not None else None,
                    'heart_rate': float(patient_data.get('Heart Rate (bpm)')) if patient_data.get('Heart Rate (bpm)') is not None else None,
                    'hemoglobin': float(patient_data.get('Hemoglobin (g/dL)')) if patient_data.get('Hemoglobin (g/dL)') is not None else None,
                    'bmi': float(patient_data.get('Pre-pregnancy BMI')) if patient_data.get('Pre-pregnancy BMI') is not None else None,
                    'expected_delivery_date': str(data.get('edd') or data.get('expected_delivery_date') or data.get('Expected Delivery Date') or 'N/A'),
                    'status': 'Monitoring',
                    'trend': 'stable',
                    'risk_score': float(risk_assessment['riskScore'])
                }
                
                # Add to dataframe using loc for better performance and no warnings
                patients_df.loc[len(patients_df)] = new_patient
                
                # Save to CSV
                patients_df.to_csv('patients.csv', index=False)
                
                patient_saved = True
                patient_details = {
                    'patient_id': str(patient_id),
                    'name': str(patient_name),
                    'total_patients': int(len(patients_df)),
                    'unnamed_patients': int(unnamed_count + (1 if not data.get("name") else 0))
                }
                
                print(f"✅ New patient saved: {patient_name} (ID: {patient_id})")
            else:
                print(f"ℹ️ Patient '{patient_name}' already exists in database")
                
        except Exception as save_error:
            print(f"⚠️ Failed to save patient to database: {save_error}")
            traceback.print_exc()

        response = {
            "success": True,
            "patientData": patient_data,
            "riskAssessment": risk_assessment,
            "metadata": {
                "version": "4.0-ML-Integrated",
                "mlModelLoaded": MODEL_LOADED,
                "fieldsProvided": len(patient_data),
                "requiredFields": required_fields
            }
        }
        
        # Add patient save details if successful
        if patient_saved:
            response["patientSaved"] = True
            response["patientDetails"] = patient_details
        
        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

# --- Patient Data Endpoint ---
@app.route("/api/patients", methods=["GET", "OPTIONS"])
def get_patients():
    """Return list of patients and their risk levels."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    try:
        if not PATIENTS_LOADED or patients_df is None:
            return jsonify({
                "error": "Patient database not available",
                "success": False
            }), 503

        patients = []
        for _, row in patients_df.iterrows():
            # Calculate risk score if not provided
            risk_score = row.get('risk_score', None)
            if risk_score is None or pd.isna(risk_score):
                risk_data = {
                    "Age (years)": row.get('age', 0),
                    "Systolic Blood Pressure (mmHg)": row.get('systolic_bp', 0),
                    "Diastolic Blood Pressure (mmHg)": row.get('diastolic_bp', 0),
                    "Heart Rate (bpm)": row.get('heart_rate', 0),
                    "Hemoglobin (g/dL)": row.get('hemoglobin', 0),
                    "Pre-pregnancy BMI": row.get('bmi', 0)
                }
                risk_assessment = calculate_pph_risk(risk_data)
                risk_score = risk_assessment['riskScore']
            
            # Safely convert values to JSON-serializable types
            patient_id = row.get('patient_id', f'MH2024-{len(patients)+1:03d}')
            patient_name = row.get('name', f'Patient {len(patients)+1}')
            patient_age = row.get('age', None)
            patient_edd = row.get('expected_delivery_date', 'N/A')
            patient_status = row.get('status', 'Monitoring')
            patient_trend = row.get('trend', 'stable')
            
            # Convert pandas types to native Python types
            if pd.isna(patient_age):
                patient_age = 'N/A'
            elif isinstance(patient_age, (int, float)):
                patient_age = int(patient_age) if patient_age == int(patient_age) else float(patient_age)
            
            patient_data = {
                "id": str(patient_id),
                "name": str(patient_name),
                "age": patient_age,
                "risk": float(risk_score),
                "edd": str(patient_edd) if not pd.isna(patient_edd) else 'N/A',
                "status": str(patient_status) if not pd.isna(patient_status) else 'Monitoring',
                "trend": str(patient_trend) if not pd.isna(patient_trend) else 'stable'
            }
            patients.append(patient_data)

        response_data = {
            "success": True,
            "patients": patients
        }
        return jsonify(response_data), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

# --- Health check endpoint ---
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "4.0-ML-Integrated",
        "mlModelLoaded": MODEL_LOADED
    }), 200

# --- Report generation endpoint ---
@app.route("/api/report", methods=["POST", "OPTIONS"])
def generate_report():
    """Generate a PDF report from patientData and riskAssessment."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        payload = request.get_json(silent=True) or {}
        patient_data = payload.get("patientData") or {}
        risk_assessment = payload.get("riskAssessment")

        # If risk is not provided, attempt to compute from patient data
        if not risk_assessment and patient_data:
            risk_assessment = calculate_pph_risk(patient_data)

        if not patient_data or not risk_assessment:
            return jsonify({
                "error": "Missing required data",
                "details": "Provide patientData and/or riskAssessment"
            }), 400

        # Build a simple PDF using Pillow (no extra system deps)
        img_w, img_h = 1240, 1754  # A4 @ ~150dpi
        bg = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(bg)

        # Safe font fallbacks
        try:
            font_title = ImageFont.truetype("Arial.ttf", 48)
            font_h2 = ImageFont.truetype("Arial.ttf", 36)
            font_body = ImageFont.truetype("Arial.ttf", 24)
            font_small = ImageFont.truetype("Arial.ttf", 20)
        except Exception:
            font_title = ImageFont.load_default()
            font_h2 = ImageFont.load_default()
            font_body = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Header bar
        header_h = 120
        draw.rectangle([(0, 0), (img_w, header_h)], fill=(179, 0, 0))
        draw.text((40, 30), "HemoSync AI - PPH Risk Report", fill=(255, 255, 255), font=font_title)

        # Risk summary box
        box_margin = 60
        y = header_h + 40
        draw.text((box_margin, y), "Risk Summary", fill=(51, 51, 51), font=font_h2)
        y += 60
        risk_score = risk_assessment.get("riskScore", 0)
        risk_level = risk_assessment.get("riskLevel", "UNKNOWN")
        model_type = risk_assessment.get("modelType", "N/A")

        draw.text((box_margin, y), f"Overall Risk: {risk_score:.1f}%", fill=(220, 38, 38), font=font_h2)
        y += 50
        draw.text((box_margin, y), f"Risk Level: {risk_level}", fill=(80, 80, 80), font=font_body)
        y += 40
        draw.text((box_margin, y), f"Model: {model_type}", fill=(80, 80, 80), font=font_body)

        # Top risk factors
        y += 60
        draw.text((box_margin, y), "Top Risk Factors", fill=(51, 51, 51), font=font_h2)
        y += 50
        top_factors = risk_assessment.get("topRiskFactors", [])
        for idx, f in enumerate(top_factors[:5], start=1):
            name = str(f.get("factor", "-")).strip()
            val = f.get("importance")
            points = f.get("points")
            display = f"{val:.3f}" if isinstance(val, (int, float)) else (str(points) + " pts" if points is not None else "")
            draw.text((box_margin, y), f"{idx}. {name}", fill=(60, 60, 60), font=font_body)
            draw.text((img_w - box_margin - 200, y), display, fill=(179, 0, 0), font=font_body)
            y += 36

        # Patient details section
        y += 40
        draw.text((box_margin, y), "Patient Details", fill=(51, 51, 51), font=font_h2)
        y += 50
        columns = [
            ("Age (years)", "Systolic Blood Pressure (mmHg)", "Diastolic Blood Pressure (mmHg)"),
            ("Heart Rate (bpm)", "Hemoglobin (g/dL)", "Pre-pregnancy BMI"),
            ("History of PPH", "History of Previous Cesarean", "Placenta Previa"),
            ("Multiple Pregnancy", "Parity", "Gestational Age (weeks)"),
            ("Mode of Delivery", "Estimated Fetal Weight (kg)", "Platelet Count (x10^3/μL)")
        ]
        col_x = [box_margin, img_w//2]
        cur_x = col_x[0]
        cur_y = y
        line_h = 30
        items = []
        for triplet in columns:
            for key in triplet:
                if key in patient_data:
                    items.append((key, patient_data.get(key)))
        # Render two columns
        half = (len(items) + 1)//2
        for i, (k, v) in enumerate(items):
            cx = col_x[0] if i < half else col_x[1]
            cy = cur_y + (i if i < half else i - half) * line_h
            draw.text((cx, cy), f"{k}: ", fill=(90, 90, 90), font=font_small)
            draw.text((cx + 370, cy), str(v), fill=(40, 40, 40), font=font_small)

        # Footer
        draw.text((box_margin, img_h - 80), "Generated by HemoSync AI - For clinical decision support only", fill=(120,120,120), font=font_small)

        # Save as PDF into memory
        pdf_io = io.BytesIO()
        bg.save(pdf_io, format="PDF", resolution=200.0)
        pdf_io.seek(0)

        # Stream PDF response
        from flask import send_file
        return send_file(
            pdf_io,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="pph_risk_report.pdf"
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to generate report", "details": str(e)}), 500

# --- Nearest Blood Banks Endpoint ---
@app.route("/api/getNearestBanks", methods=["POST", "OPTIONS"])
def get_nearest_banks():
    """Find nearest blood banks with required blood type."""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        if not BLOOD_BANKS_LOADED or blood_banks_df is None:
            return jsonify({
                "error": "Blood bank database not available",
                "success": False
            }), 503

        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        lat = data.get("lat")
        lon = data.get("lon")
        blood_type = data.get("bloodType") 
        if blood_type == 'N/A':
            blood_type = None # Can be None
        radius = data.get("radius", 100)  # Default 100 km

        if lat is None or lon is None:
            return jsonify({"error": "Latitude and longitude required"}), 400

        patient_loc = (float(lat), float(lon))
        results = []

        # Search blood banks within radius
        for _, row in blood_banks_df.iterrows():
            try:
                bank_loc = (row['lat'], row['lon'])
                dist = haversine(patient_loc, bank_loc, unit=Unit.KILOMETERS)
                
                # If no blood type specified, show all banks within radius
                # If blood type specified, only show banks with that blood type available
                if dist <= radius:
                    if blood_type is None or blood_type == "":
                        # Show all blood banks with total units available
                        blood_types = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
                        total_units = sum([int(row[bt]) for bt in blood_types if bt in row])
                        
                        if total_units > 0:
                            # Create inventory dict
                            inventory = {bt: int(row[bt]) for bt in blood_types if bt in row and int(row[bt]) > 0}
                            
                            results.append({
                                'name': row['name'],
                                'distance_km': round(dist, 2),
                                'units_available': total_units,
                                'inventory': inventory,
                                'expiry_dates': row['expiry_dates'],
                                'lat': float(row['lat']),
                                'lon': float(row['lon']),
                                'blood_type': 'All Types'
                            })
                    elif row[blood_type] > 0:
                        # Specific blood type requested
                        results.append({
                            'name': row['name'],
                            'distance_km': round(dist, 2),
                            'units_available': int(row[blood_type]),
                            'expiry_dates': row['expiry_dates'],
                            'lat': float(row['lat']),
                            'lon': float(row['lon']),
                            'blood_type': blood_type
                        })
            except Exception as e:
                continue  # Skip invalid rows

        # Sort by distance
        results = sorted(results, key=lambda x: x['distance_km'])[:10]  # Top 10 nearest

        return jsonify({
            "success": True,
            "patient_location": {"lat": lat, "lon": lon},
            "blood_type": blood_type if blood_type else "All Types",
            "radius_km": radius,
            "banks_found": len(results),
            "banks": results
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

# --- Start Server with ngrok ---
if __name__ == "__main__":
    NGROK_AUTH = "34E8Zy0QlUtdlZOrqCpJpSlvYv6_3a6jvKovTP96wSTuGumo2"

    try:
        port = 8000
        ngrok.set_auth_token(NGROK_AUTH)
        tunnel = ngrok.connect(port)
        public_url = tunnel.public_url  # Extract the actual URL string

        print("=" * 60)
        print("🚀 HemoSync AI-Powered OCR API Server Started!")
        print(f"📡 Public URL: {public_url}")
        print(f"🔗 OCR Endpoint: {public_url}/api/ocr")
        print(f"🩺 Risk Analysis Endpoint: {public_url}/api/risk-analysis")
        print(f"💚 Health Check: {public_url}/api/health")
        print(f"🤖 ML Model: {'✅ Loaded' if MODEL_LOADED else '⚠️ Using Rule-Based'}")
        print("=" * 60)
        print(f"👉 COPY THESE LINES TO YOUR HTML FRONTEND:")
        print(f"   const OCR_API_URL = '{public_url}/api/ocr';")
        print(f"   const RISK_API_URL = '{public_url}/api/risk-analysis';")
        print("=" * 60)

        app.run(port=port, debug=True, use_reloader=False)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        traceback.print_exc()
        exit(1)