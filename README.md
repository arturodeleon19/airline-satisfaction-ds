# ✈️ SkyPulse: Airline Passenger Satisfaction & Retention Analysis

An end-to-end Machine Learning pipeline and interactive web application predicting customer satisfaction from flight survey responses and operational metrics using **LightGBM**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airline-satisfaction-ds-ff4t2bgwuys7nkbitasgot.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Framework](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)

---

## 📌 Executive Summary
Understanding the root drivers of passenger dissatisfaction is vital for airline profitability and customer lifetime value. Utilizing a large-scale survey dataset comprising over 700,000 passenger records, this project delivers:
1. **Exploratory Data Analysis (EDA):** Uncovering demographic trends, target class dynamics, and service touchpoint correlations.
2. **Machine Learning Pipeline:** A robust, high-performance **LightGBM Booster** model predicting satisfaction probabilities in real-time.
3. **Interactive Deployment:** A deployed Streamlit application allowing stakeholders to simulate passenger profiles and instantly evaluate satisfaction risks.

---

## 📊 Exploratory Data Analysis & Key Visual Findings

### 1. Target Class Distribution
<img width="1584" height="484" alt="image" src="https://github.com/user-attachments/assets/7a02bbd2-f148-4531-bdab-4456227878d7" />

* **Observation:** The target variable exhibits a moderate class imbalance, with **Dissatisfied (False)** passengers accounting for roughly 390,000 records (~55.7%) and **Satisfied (True)** passengers representing about 310,000 records (~44.3%). 
* **Business Implication:** Handling class balance correctly during model evaluation (leveraging metrics like ROC-AUC and precision-recall curves rather than raw accuracy alone) is essential to avoid bias toward the majority class.

### 2. Service Touchpoint Multicollinearity (Correlation Matrix)

<img width="1080" height="884" alt="image" src="https://github.com/user-attachments/assets/8c2882b8-c26f-4c53-be63-f1a05b182b90" />

* **Observation:** The heatmap reveals strong feature collinearity among digital and physical amenities. For instance, **Inflight Wi-Fi service** and **Ease of Online booking** share a strong positive correlation ($r = 0.77$). Similarly, physical comfort features like *Seat comfort*, *Inflight entertainment*, *Food and drink*, and *Cleanliness* cluster tightly together with correlation coefficients ranging from $0.66$ to $0.76$.
* **Business Implication:** These clusters indicate that passengers evaluate travel experience holistically (e.g., digital convenience vs. physical comfort), guiding feature selection and tree-based model splitting logic.

### 3. Passenger Demographics: Age vs. Satisfaction

<img width="984" height="484" alt="image" src="https://github.com/user-attachments/assets/506b1eae-5fce-4a4e-86b5-b2d5b317a17e" />

* **Observation:** The age distribution density plot highlights distinct demographic sentiment patterns. Younger adult passengers (ages 20–30) experience a noticeably higher density of **dissatisfaction** (red distribution). Conversely, middle-aged passengers (ages 38–60) tilt significantly more toward the **satisfied** segment (green distribution).
* **Business Implication:** Airlines should investigate specific onboarding or service friction points tailored to younger demographics (such as digital expectations or budget-class constraints).

### 4. Service Rating Profile (Radar Analysis)

<img width="794" height="691" alt="image" src="https://github.com/user-attachments/assets/bdb46de7-512e-43f6-8ea9-07dc45a19898" />

* **Observation:** The multi-axis radar chart compares average scores across 14 service dimensions for satisfied vs. dissatisfied cohorts. **Satisfied passengers (green)** consistently rate critical touchpoints like *Online boarding*, *Inflight entertainment*, *Seat comfort*, and *Inflight Wi-Fi service* significantly higher, creating a wide geometric perimeter. Dissatisfied passengers (red) show sharp contractions in these exact digital and in-flight entertainment categories.
* **Business Implication:** Upgrading Wi-Fi reliability and online boarding workflows yields higher marginal returns on customer satisfaction than physical gate location adjustments.

---

## 🏗️ Repository Architecture

```text
├── app.py                      # Interactive Streamlit web application
├── lgbm_best_model.pkl         # Pre-trained LightGBM Booster model artifact
├── requirements.txt            # Project dependencies
├── notebooks/                  # Jupyter notebooks for EDA and model training
│   ├── Airline Satisfaction.ipynb
└── README.md                   # Project documentation
