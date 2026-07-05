# 🩸CodeCrimson by Sentio

> **AI-Powered Blood Management System for Emergency Healthcare**

**CodeCrimson** is an intelligent maternal health and blood management platform that predicts **Postpartum Hemorrhage (PPH) risk** and enables **rapid emergency response** through a **voice-activated OT Mode**, ensuring faster, safer, and more coordinated interventions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚠️ Disclaimer

CodeCrimson AI is a **decision-support tool**, not a substitute for clinical judgment.
All predictions and recommendations must be reviewed by **qualified healthcare professionals**.
This system is designed to **assist**, not replace, medical expertise.

---

##  Key Features

###  AI-Powered PPH Risk Prediction

* **XGBoost-based model** trained on over 3,200 maternal records
* Predicts **Postpartum Hemorrhage (PPH)** risk from vital and obstetric parameters
* Provides **real-time risk prediction** and explainable feature contributions
* Generates downloadable **medical-grade reports**

###  OT Mode – Voice-Activated Emergency Response

* Detects triple keyword **“blood, blood, blood”** to trigger emergency alert
* Instantly contacts **nearest blood bank** via Twilio integration
* Includes **visual confirmation**, **manual override**, and **activity logging**
* Designed as a **fallback system** during unpredictable hemorrhage emergencies

### Maternal Health Reporting

* Auto-generates detailed health summaries and risk reports
* PDF export for hospital documentation
* Tracks vitals: hemoglobin, platelets, blood pressure, BMI, and more

###  Blood Bank Management

* Real-time **inventory tracking** and **nearest bank locator**
* Automated **emergency request routing**
* Twilio-based **voice call system** for immediate response coordination

---

##  How It Works

### PPH Risk Prediction Pipeline

```
Patient Data Input → Feature Engineering → XGBoost Model → Risk Score → Medical Report
```

**Input Parameters (15–18 Clinical Features):**
Maternal demographics, vital signs, obstetric history, delivery details, and risk factors including:

* Age, parity, BMI, hemoglobin, platelet count
* Previous PPH / C-section history
* Placenta previa/accreta, multiple pregnancy
* Gestational age, fetal weight, BP, heart rate
* Mode of delivery, induction, anesthesia type, retained placenta

**Output:**

* Risk category: Low / Medium / High
* Explainable feature importance
* Auto-generated medical report (PDF)

---

##  System Architecture

```
┌───────────────────────────┐
│     Frontend (HTML/JS)    │
│ - Forms & OT Mode UI      │
│ - Report Generation (PDF) │
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│   Flask Backend (Python)  │
│ - XGBoost Inference       │
│ - Voice Trigger Handler   │
│ - Twilio Integration      │
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│       Data Layer           │
│ - Patient Records DB       │
│ - Blood Bank Directory     │
│ - Emergency Logs           │
└───────────────────────────┘
```

---

##  Tech Stack

**Frontend:** HTML, CSS, JavaScript (Web Speech API)
**Backend:** Flask (Python)
**Machine Learning:** XGBoost
**APIs:** Twilio Voice API for emergency alerts
**Libraries:** Pandas, NumPy, Scikit-learn, ReportLab (PDF generation)

---

##  ML Model Overview

* **Algorithm:** XGBoost (Extreme Gradient Boosting)
* **Training Data:** 3,200+ maternal health records from diverse datasets
* **Validation:** 5-fold cross-validation and benchmarking with literature
* **Accuracy:** 89% on validation set
* **Deployment Target:** District and civil hospitals (lightweight & explainable)

**Top Predictors:**

1. Previous PPH history
2. Placenta previa/accreta/increta
3. Multiple pregnancy
4. Hemoglobin level
5. Mode of delivery

---

## 🎤 Voice & Emergency System

### Workflow

```
Voice Input → Keyword Detection → Emergency Trigger → Twilio Call → Blood Bank Alert
```

**Highlights:**

* Powered by **Web Speech API** (browser-based detection)
* Latency <500ms for activation
* **Triple keyword** prevents accidental triggers
* No audio recorded — **privacy-first design**
* All emergency activations are **logged for auditing**

---

## Dashboard & Analytics

* Real-time PPH risk visualization
* Response time tracking for emergencies
* Blood usage & stock analytics
* Predictive inventory insights for hospitals

---

##  License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

