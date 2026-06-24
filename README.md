# 🛡️ AI-Based Phishing Website Detector

An intelligent cybersecurity platform that detects phishing websites using Machine Learning, URL feature engineering, WHOIS intelligence, SSL certificate analysis, and real-time threat scoring.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey)

---

## 📌 Project Overview

Phishing attacks are one of the most common cybersecurity threats used to steal sensitive information such as usernames, passwords, banking credentials, and personal data.

This project uses Machine Learning and Cyber Threat Intelligence techniques to analyze URLs and determine whether a website is:

* ✅ Legitimate
* ⚠️ Suspicious
* 🚨 Phishing

The system provides explainable predictions, threat scores, SSL verification, domain intelligence, and detailed security analytics through an interactive Streamlit dashboard.

---

## 🚀 Features

### 🔍 URL Analysis

* Real-time URL scanning
* Advanced feature extraction
* Suspicious keyword detection
* URL structure analysis

### 🤖 Machine Learning Detection

* Random Forest Classifier
* XGBoost Classifier
* Automatic best model selection
* ROC-AUC based evaluation

### 🧠 Explainable AI

* Detection reasoning
* Feature importance analysis
* Risk factor identification

### 🌐 Domain Intelligence

* WHOIS Lookup
* Domain Age Detection
* Registrar Information
* DNS Record Validation

### 🔒 SSL Security Analysis

* SSL Certificate Validation
* Certificate Expiry Detection
* SSL Issuer Verification

### 📊 Threat Dashboard

* Confidence Score
* Threat Score
* Risk Classification
* Interactive Visualizations

### 🗄️ Prediction Logging

* SQLite Database Storage
* Historical Analysis
* Threat Statistics
* Prediction History

---

## 🏗️ Project Architecture

```text
AI-Phishing-Detector/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── feature_extraction.py
│   ├── whois_lookup.py
│   ├── ssl_checker.py
│   ├── retrain_model.py
│   └── database.py
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── best_phishing_model.pkl
│   └── model_meta.json
│
├── reports/
│   └── metrics.json
│
├── database/
│   └── predictions.db
│
├── data/
│   └── processed/
│       └── phishing_data.csv
│
└── README.md
```

---

## ⚙️ Machine Learning Pipeline

### Data Processing

* Data Cleaning
* Missing Value Handling
* Label Encoding
* Feature Selection

### Models

#### Random Forest

* 200 Decision Trees
* Parallel Processing
* High Accuracy

#### XGBoost

* Gradient Boosting
* Optimized Performance
* Superior Generalization

### Model Evaluation

Metrics Used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

The best model is automatically selected and deployed for predictions.

---

## 📈 Extracted URL Features

The detector analyzes multiple phishing indicators including:

* URL Length
* Domain Length
* Number of Dots
* Number of Hyphens
* Number of Digits
* HTTPS Usage
* IP Address Detection
* Special Characters
* Suspicious Keywords
* Subdomain Count
* Entropy Score
* Domain Age
* DNS Records
* SSL Information

---

## 🛠️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AI-Phishing-Detector.git

cd AI-Phishing-Detector
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Dataset

Recommended Dataset:

### UCI Phishing Websites Dataset

Contains:

* 11,055 website samples
* 30 phishing indicators
* Binary classification labels

Place dataset in:

```text
data/processed/phishing_data.csv
```

---

## 🏋️ Train Model

```bash
python src/train.py
```

Or specify custom dataset:

```bash
python src/train.py --data data/processed/phishing_data.csv
```

Generated Files:

```text
models/
├── random_forest.pkl
├── xgboost.pkl
├── best_phishing_model.pkl
└── model_meta.json
```

---

## 🚀 Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open browser:

```text
http://localhost:8501
```

---

## 📊 Dashboard Modules

### Home

* Threat Statistics
* Model Accuracy
* Key Features

### URL Analyzer

* URL Scanning
* Threat Detection
* Confidence Scoring
* Explainable AI

### Model Dashboard

* Performance Metrics
* ROC-AUC Comparison
* Feature Importance

### Prediction Log

* Historical Predictions
* Threat Trends
* Risk Distribution

---

## 🔐 Cybersecurity Applications

* Phishing Website Detection
* Email Security
* SOC Monitoring
* Threat Intelligence
* Fraud Prevention
* Security Awareness Training

---

## 🎯 Future Enhancements

* Deep Learning Models
* SHAP Explainability
* VirusTotal Integration
* Real-Time Threat Feeds
* React Frontend Dashboard
* REST API Support
* Cloud Deployment
* AI Security Assistant

---

## 👨‍💻 Author

Kishore Kumar

B.Tech Artificial Intelligence and Data Science

Chennai Institute of Technology

---

## 📄 License

This project is developed for educational, research, and cybersecurity awareness purposes.
