# 🛡️ Intelligent Website Trust Evaluation System
> **An AI-powered cybersecurity platform for real-time phishing detection using Machine Learning, URL feature engineering, WHOIS domain intelligence, SSL certificate analysis, and explainable threat scoring.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-239120?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#%EF%B8%8F-system-architecture)
- [Machine Learning & Benchmarks](#-machine-learning--benchmarks)
- [Multi-Layered Detection Engine](#-multi-layered-detection-engine)
- [Extracted URL Features](#-extracted-url-features)
- [Streamlit Dashboard Modules](#-streamlit-dashboard-modules)
- [Project Structure](#-project-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Usage Guide](#-usage-guide)
- [Cybersecurity Use Cases](#-cybersecurity-use-cases)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 📌 Project Overview

Phishing attacks remain one of the most prevalent vector mechanisms for credential harvesting and malware distribution. Traditional blacklist-based detection tools often fail against zero-day phishing domains.

The **Intelligent Website Trust Evaluation System** addresses this challenge by providing a **hybrid intelligence pipeline** that combines:
1. **Machine Learning Classifiers** (Random Forest & XGBoost) trained on 30 URL behavioral features.
2. **Real-time Domain Intelligence** (WHOIS age, registrar validation, DNS records).
3. **SSL Certificate Security Analysis** (Issuer verification, certificate expiration, protocol validity).
4. **Hybrid Threat Scoring Algorithm** yielding explainable risk classifications (`Legitimate`, `Suspicious`, `Phishing`).

---

## 🚀 Key Features

* **🔍 Real-Time URL Inspection:** Rapid analysis of arbitrary web links with live feature extraction.
* **🤖 Multi-Model ML Ensemble:** Automated comparison and deployment of top-performing models (**96.92% Accuracy**, **0.9965 ROC-AUC**).
* **🧠 Explainable AI (XAI):** Clear breakdown of risk drivers, feature importances, and flagged anomalies.
* **🌐 Live WHOIS Lookup:** Real-time domain registration age detection and registrar details.
* **🔒 SSL Certificate Inspector:** Active validation of SSL chain, issuer validity, and expiration warning.
* **🗄️ SQLite Prediction Audit Log:** Persistent logging of past evaluations with search, filtering, and CSV export.
* **⚙️ Model Retraining Engine:** In-app interface to retrain models on new datasets directly from the dashboard.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[User / Web Input URL] --> B[Feature Extraction Engine]
    A --> C[Live WHOIS Lookup]
    A --> D[Live SSL Certificate Checker]
    
    B --> E[Extracted 30 Features]
    E --> F[ML Classification Model\nRandom Forest / XGBoost]
    
    F --> G[ML Prediction & Probabilities]
    C --> H[Domain Age & Metadata]
    D --> I[SSL Status & Expiry]
    
    G --> J[Hybrid Threat Scoring Engine]
    H --> J
    I --> J
    
    J --> K[Final Threat Score & Risk Level\nLegitimate | Suspicious | Phishing]
    K --> L[Streamlit Dashboard & SQLite Audit Log]
```

---

## 📊 Machine Learning & Benchmarks

The system trains and benchmarks multiple classification algorithms on the **UCI Phishing Websites Dataset** (11,055 samples). Model selection is automatically governed by test performance metrics saved in `reports/metrics.json`.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🌲 **Random Forest** | **96.92%** | **96.86%** | **97.65%** | **97.25%** | **0.9965** | 🏆 **Deployed Best Model** |
| ⚡ **XGBoost** | **96.79%** | **96.63%** | **97.65%** | **97.13%** | **0.9964** | 🥈 Runner-Up |

### Confusion Matrix (Best Model: Random Forest)
- **True Positives (Phishing correctly identified):** 1,203
- **True Negatives (Legitimate correctly identified):** 940
- **False Positives:** 39
- **False Negatives:** 29

---

## 🛡️ Multi-Layered Detection Engine

The system computes a unified **Threat Score (0–100)** by combining multiple security signals:

$$ \text{Threat Score} = (\text{ML Probability} \times 0.60) + (\text{WHOIS Risk} \times 0.20) + (\text{SSL Risk} \times 0.15) + (\text{Keyword Heuristics} \times 0.05) $$

| Threat Score Range | Risk Classification | Action / Recommendation |
| :--- | :--- | :--- |
| **0 – 35%** | ✅ **Legitimate** | Safe to proceed. No major anomalies detected. |
| **36 – 65%** | ⚠️ **Suspicious** | Exercise caution. Minor risk factors or young domain found. |
| **66 – 100%** | 🚨 **Phishing** | High Risk! Do not enter credentials or personal information. |

---

## 📈 Extracted URL Features

The feature extractor processes **30 distinct website indicators** categorised into four main groups:

1. **Address Bar Features:** `UsingIP`, `LongURL`, `ShortURL`, `Symbol@`, `Redirecting//`, `PrefixSuffix-`, `SubDomains`, `HTTPS`.
2. **Abnormal & Behavioral Features:** `RequestURL`, `AnchorURL`, `LinksInScriptTags`, `ServerFormHandler`, `InfoEmail`, `AbnormalURL`.
3. **HTML & JavaScript Features:** `WebsiteForwarding`, `StatusBarCust`, `DisableRightClick`, `UsingPopupWindow`, `IframeRedirection`.
4. **Domain & Network Intelligence:** `AgeofDomain`, `DNSRecording`, `WebsiteTraffic`, `PageRank`, `GoogleIndex`, `LinksPointingToPage`, `StatsReport`.

---

## 🖥️ Streamlit Dashboard Modules

The application is structured into 5 interactive modules:

1. **🏠 Overview & Health:** Executive summary, active model information, live database scan counts, and high-level KPI metrics.
2. **🔍 Real-Time URL Scanner:** Interactive URL entry, visual risk gauge meter, WHOIS details, SSL status, and explainable feature breakdowns.
3. **🤖 Model Insights & Evaluation:** Model comparison plots, confusion matrices, ROC curves, and top feature importance visualization.
4. **🗄️ Prediction History & Audit Logs:** Historical database view (`predictions.db`), filterable logs, and CSV report export functionality.
5. **⚙️ Model Retraining:** Upload new training CSV files to execute automated model re-fitting directly from the web interface.

---

## 📂 Project Structure

```text
Intelligent Website Trust Evaluation/
│
├── app/
│   └── streamlit_app.py        # Streamlit interactive multi-tab dashboard UI
│
├── src/
│   ├── feature_extraction.py   # Extracts 30 URL & domain security features
│   ├── whois_lookup.py         # Queries WHOIS metadata (age, registrar, dates)
│   ├── ssl_checker.py          # Validates SSL certificate chain & expiration
│   ├── predict.py              # Real-time URL prediction & threat scoring engine
│   ├── train.py                # Model training, cross-validation & selection
│   ├── evaluate.py             # Evaluation metrics calculation & visualization
│   ├── database.py             # SQLite database operations & log management
│   └── retrain_model.py        # Dynamic automated model retraining workflow
│
├── models/
│   ├── best_phishing_model.pkl # Deployed top-performing binary classifier
│   ├── random_forest.pkl       # Serialized Random Forest model artifact
│   ├── xgboost.pkl             # Serialized XGBoost model artifact
│   └── model_meta.json         # Feature list, classes, & active model metadata
│
├── reports/
│   └── metrics.json            # Model benchmark evaluation metrics & CM data
│
├── database/
│   └── predictions.db          # SQLite database storing historical scan records
│
├── data/
│   └── processed/
│       └── phishing_data.csv   # Processed training dataset (UCI Phishing)
│
├── requirements.txt            # Python dependency requirements
└── README.md                   # Complete system documentation
```

---

## 🛠️ Installation & Quick Start

### Prerequisites
- **Python:** `3.10` or higher
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AI-Phishing-Detector.git
cd AI-Phishing-Detector
```

### 2. Set Up Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Run Model Training (Optional)
To train models from scratch or re-evaluate performance:
```bash
python src/train.py
```
*This updates `models/best_phishing_model.pkl` and `reports/metrics.json`.*

### 2. Launch the Interactive Web Dashboard
```bash
streamlit run app/streamlit_app.py
```

Access the application in your browser at **`http://localhost:8501`**.

---

## 🔐 Cybersecurity Use Cases

* **🛡️ Security Operations Center (SOC):** Rapid triage of suspicious link submissions.
* **📧 Email Security Gateway Integration:** Automated scanning of hyperlinks in incoming emails.
* **🌐 Web Browser Extension Backend:** API-ready engine for real-time browsing protection.
* **🎓 Cybersecurity Education:** Demonstrating Explainable AI (XAI) in URL threat analysis.

---

## 👨‍💻 Author & Contact

**Kishore Kumar**  
*B.Tech in Artificial Intelligence & Data Science*  
Chennai Institute of Technology  

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

