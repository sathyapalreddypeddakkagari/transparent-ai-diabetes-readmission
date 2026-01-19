# Predicting Hospital Readmission for Diabetic Patients

## 🎯 Problem Statement

A hospital readmission occurs when a discharged patient returns within a specific period (typically 30 days). High readmission rates signal poor hospital quality and significantly increase healthcare costs. The Hospital Readmissions Reduction Program (HRRP) penalizes hospitals with excessive readmissions for certain conditions, costing approximately **$41 billion for diabetic patients in 2011**. Though diabetes isn't yet directly penalized under HRRP, it's a growing concern in healthcare quality management.

### Business Impact
- **Financial Burden**: High readmission rates increase hospital costs and reduce reimbursement rates
- **Quality Metrics**: Readmission rates are key indicators of hospital quality and patient outcomes
- **Clinical Outcomes**: Early identification of high-risk patients enables proactive interventions



## 📊 Objective

Using a medical claims dataset from 130 US hospitals (1999-2008), this project aims to answer:
1. **What are the strongest predictors of hospital readmission in diabetic patients?**
2. **How accurately can we predict readmissions with limited features using machine learning?**


## 📋 Dataset Description

### Source
- **Dataset**: Diabetes 130-US Hospitals for Years 1999-2008
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Records**: 101,766 patient encounters
- **Features**: 47+ features per encounter
- **Time Period**: 1999-2008
- **Hospitals**: 130 US hospitals and integrated delivery networks

### Dataset Characteristics
- **Subject Area**: Health and Medicine
- **Associated Tasks**: Classification, Clustering
- **Feature Types**: Categorical, Integer
- **Missing Values**: Yes (handled in preprocessing)

### Selection Criteria
Encounters included if they met all of:
1. Inpatient encounter (hospital admission)
2. Diabetic encounter (any diabetes diagnosis)
3. Length of stay: 1-14 days
4. Laboratory tests performed
5. Medications administered



## 🔑 Key Variables

| **Variable** | **Description** | **Values/Format** |
|--------------|------------------|-------------------|
| `encounter_id` | Unique encounter identifier | Numeric |
| `patient_nbr` | Unique patient identifier | Numeric |
| `race` | Patient's race | Caucasian, Asian, African American, Hispanic, Other, Unknown |
| `gender` | Patient's gender | Male, Female, Unknown/Invalid |
| `age` | Patient's age group | 10-year intervals: [0-10), [10-20), ..., [90-100) |
| `admission_type_id` | Type of admission | Integer (9 distinct values: Emergency, Urgent, Elective, etc.) |
| `discharge_disposition_id` | Discharge status | Integer (29 values: Home, Expired, etc.) |
| `admission_source_id` | Source of admission | Integer (21 values: Physician Referral, Emergency Room, etc.) |
| `time_in_hospital` | Days from admission to discharge | Integer (1-14) |
| `medical_specialty` | Admitting physician's specialty | 84 values (Cardiology, Internal Medicine, etc.) |
| `num_lab_procedures` | Lab tests performed | Numeric |
| `num_procedures` | Non-lab procedures | Numeric |
| `num_medications` | Distinct medications given | Numeric |
| `number_outpatient` | Outpatient visits in prior year | Numeric |
| `number_emergency` | Emergency visits in prior year | Numeric |
| `number_inpatient` | Inpatient visits in prior year | Numeric |
| `diag_1`, `diag_2`, `diag_3` | Primary, Secondary, Tertiary diagnosis | ICD-9 codes (848, 923, 954 unique values) |
| `number_diagnoses` | Total diagnoses recorded | Numeric |
| `max_glu_serum` | Blood glucose level | >200, >300, Norm, Not Tested |
| `A1Cresult` | A1c test outcome | >8%, >7%, Norm, Not Tested |
| `change` | Change in diabetic medication | Change (Ch), No Change (No) |
| `diabetesMed` | Diabetic medication prescribed | Yes, No |
| **Medications (8 features)** | Status of diabetic drugs | Binary (metformin, repaglinide, glimepiride, glipizide, glyburide, pioglitazone, rosiglitazone, insulin) |
| `readmitted` | Time to readmission | <30 days, >30 days, NO |

---

## 🏗️ Project Structure

```
Final-582_Project_code/
├── app.py                                    # Streamlit inference application
├── prediction-on-hospital-readmission.ipynb  # Model training & evaluation notebook
├── xgb_model.pkl                            # Trained XGBoost model artifact
├── scaler.pkl                               # StandardScaler fitted on training data
├── final_features.json                      # Feature schema (30 features)
├── diabetic_data.csv                        # Raw dataset (101,766 records)
├── final_data.csv                           # Processed/cleaned dataset
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## 🔄 Methodology & Step-by-Step Approach

### 1. **Data Analysis & Exploration**
- Loaded and examined 101,766 patient encounters
- Identified missing values (represented as '?')
- Analyzed feature distributions and correlations
- Explored readmission rate patterns

### 2. **Data Preprocessing**
- **Missing Value Handling**:
  - Replaced '?' with appropriate defaults (e.g., 'Unknown' for race, 'Missing' for diagnoses)
  - Dropped deceased patients (discharge_disposition_id = 11)
  - Removed low-variance medication columns
  
- **Data Aggregation**:
  - Grouped by `patient_nbr` to create patient-level features
  - Aggregated encounters per patient (mean, sum, max operations)
  - Created `encounter_count` feature

- **Feature Engineering**:
  - **Log Transforms**: Applied log1p to skewed features (time_in_hospital, num_lab_procedures, num_medications, number_inpatient, service_utilization)
  - **Service Utilization**: Combined outpatient + emergency + inpatient visits
  - **Numchange**: Count of active medications
  - **Ratio Features**: 
    - `meds_per_diag` = num_medications / number_diagnoses
    - `hospital_per_age` = time_in_hospital / age
  - **ICD-9 Grouping**: Categorized diagnoses into:
    - **Diabetes** (250.x)
    - **Circulatory** (390, 410, 428)
    - **Respiratory** (460, 786)
    - **Missing**
    - **Other**
  - **Medical Specialty Binning**: Grouped into InternalMedicine, Cardiology, Family/GeneralPractice, Other
  - **Interaction Terms**: 
    - `num_medications|time_in_hospital`
    - `num_medications|number_diagnoses`
  - **One-Hot Encoding**: Applied to categorical variables (medical_specialty_group, max_glu_serum, A1Cresult, diag_1/2/3_group)

### 3. **Feature Selection**
- Used **Recursive Feature Elimination (RFE)** with Random Forest
- Selected top 30 features from original feature set
- Final feature set: `final_features.json` (30 engineered features)

### 4. **Model Training**
- **Train-Test Split**: 80-20 split with random_state=0
- **Feature Scaling**: StandardScaler applied to training/test sets
- **Class Imbalance Handling**: **ADASYN** (Adaptive Synthetic Sampling) for oversampling
- **Models Evaluated**:
  1. **Decision Tree** (with GridSearchCV)
  2. **Random Forest**
  3. **XGBoost** (selected as final model)
  4. **LightGBM**
  5. **Logistic Regression**

- **Hyperparameter Tuning**: GridSearchCV with StratifiedKFold (5-fold CV)

### 5. **Model Evaluation**
- **Metrics Used**:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
  - ROC-AUC Score
  - Precision-Recall AUC
  - Threshold Optimization (for precision ≥ 0.50 and recall ≥ 0.70)

- **Final Model**: **XGBoost** with optimized threshold (0.60)

### 6. **Model Explainability**
- **SHAP (SHapley Additive exPlanations)** for feature importance
- TreeExplainer for XGBoost model
- Summary plots and feature contribution analysis

---

## 📁 Key Files & Artifacts

### `app.py` - Streamlit Inference Application
**Purpose**: Real-time patient readmission risk prediction interface

**Features**:
- **Real-Time Patient Form**: Input EHR data for any patient
- **Quick Demo Patients**: Pre-configured example cases
- **ICD-9 Diagnosis Groupings**: Automatic categorization of diagnoses
- **Primary Features Evaluation**: Display of top 11 predictive features
- **SHAP Explanation**: Visual explanation of model predictions
- **Clinical Decision Support**: Recommendations based on risk level
- **Interactive Dataset Dashboard**: Analytics and visualizations of cleaned data
- **ICD-9-CM Code Reference Table**: Complete mapping of diagnosis codes

**Key Functions**:
- `preprocess_single_patient()`: Applies same feature engineering pipeline as training
- `get_shap_explanation()`: Generates SHAP values for explainability
- `get_icd9_diagnosis_name()`: Maps ICD-9 codes to diagnosis names using ICD-9-CM classification
- `group_icd9()`: Groups ICD-9 codes into clinical categories

### `prediction-on-hospital-readmission.ipynb` - Training Notebook
**Purpose**: Complete model development pipeline

**Contents**:
- Data loading and exploration
- Data preprocessing and feature engineering
- Feature selection (RFE)
- Model training and hyperparameter tuning
- Model evaluation and comparison
- SHAP analysis for model interpretability
- Threshold optimization

### Model Artifacts

#### `xgb_model.pkl`
- **Type**: XGBoost Classifier (trained model)
- **Format**: joblib pickle file
- **Usage**: Loaded in `app.py` for real-time predictions

#### `scaler.pkl`
- **Type**: StandardScaler (fitted on training data)
- **Format**: joblib pickle file
- **Usage**: Scales input features before prediction (same scaling as training)

#### `final_features.json`
- **Type**: JSON array of feature names
- **Content**: 30 final selected features in exact order
- **Usage**: Ensures feature alignment between training and inference
- **Features Include**:
  - Log-transformed features (time_in_hospital_log, num_medications_log, etc.)
  - Ratio features (meds_per_diag, hospital_per_age)
  - Interaction terms
  - One-hot encoded categorical features
  - Medication indicators
  - Demographics (gender, change)

---

## 🚀 Usage

### Running the Streamlit App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501` with two main tabs:
1. **🔍 Real-Time Inference**: Patient risk prediction with SHAP explanations
2. **📊 Dataset Dashboard**: Interactive data exploration and analytics

### Using the Notebook

1. Open `prediction-on-hospital-readmission.ipynb` in Jupyter
2. Ensure `diabetic_data.csv` is in the same directory
3. Run cells sequentially to reproduce model training

---

## 🔧 Technology Stack

### Core Libraries
- **pandas** (≥2.0.0): Data manipulation and analysis
- **numpy** (≥1.24.0): Numerical computations
- **scikit-learn** (≥1.3.0): Machine learning algorithms and utilities

### Machine Learning
- **XGBoost** (≥2.0.0): Gradient boosting framework (final model)
- **LightGBM** (≥4.0.0): Gradient boosting alternative
- **imbalanced-learn** (≥0.11.0): ADASYN for class imbalance

### Explainability
- **SHAP** (≥0.42.0): Model interpretability and explainability

### Web Application
- **Streamlit** (≥1.28.0): Interactive web app framework
- **matplotlib** (≥3.7.0): Visualization and plotting

### Utilities
- **joblib** (≥1.3.0): Model serialization (pickle files)

---

## 📊 Model Performance

### Final Model: XGBoost
- **Threshold**: 0.60 (optimized for precision ≥ 0.50 and recall ≥ 0.70)
- **Primary Features**: 11 key features identified for evaluation
  - `age`
  - `time_in_hospital_log`
  - `num_medications_log`
  - `number_inpatient_log`
  - `service_utilization_log`
  - `numchange`
  - `encounter_count`
  - `meds_per_diag`
  - `hospital_per_age`
  - `num_medications|time_in_hospital`
  - `num_medications|number_diagnoses`

### Evaluation Metrics
(Exact metrics available in notebook output)
- **ROC-AUC**: Model performance on receiver operating characteristic curve
- **PR-AUC**: Precision-recall area under curve
- **Precision**: Positive predictive value
- **Recall**: Sensitivity (true positive rate)
- **F1-Score**: Harmonic mean of precision and recall

---

## 🔍 Key Insights & Findings

### Strongest Predictors of Readmission
Based on SHAP analysis and feature importance:

1. **Encounter History**: `encounter_count` - Multiple previous encounters
2. **Service Utilization**: `service_utilization_log` - High utilization across all services
3. **Hospital Stay**: `time_in_hospital_log` - Longer stays
4. **Medications**: `num_medications_log` - Higher medication count
5. **Medication Changes**: `numchange` - Active medication changes
6. **Diagnoses**: Number of diagnoses and ICD-9 groupings
7. **Age**: Patient age group
8. **Interaction Effects**: Combinations of medications × hospital stay, medications × diagnoses

### ICD-9 Diagnosis Patterns
- **Primary Categories**: Diabetes (250.x), Circulatory (390-459), Respiratory (460-519)
- **ICD-9-CM Classification**: Full mapping available in dashboard reference table

---

## 📝 Notes on Data Governance

### Privacy & HIPAA Compliance
- **No PHI in Logs**: Dashboard displays aggregated data only
- **Patient Identifiers**: `encounter_id` and `patient_nbr` are anonymized identifiers
- **Schema Versioning**: `final_features.json` ensures feature consistency

### Reproducibility
- **Fixed Random Seeds**: `random_state=0` throughout
- **Model Artifacts**: Saved models ensure consistent predictions
- **Feature Pipeline**: Same preprocessing applied in training and inference

### Model Versioning
- Model artifacts tagged with training date/version
- Feature schema stored separately for validation

---

## 🔗 References

- **Dataset**: [UCI ML Repository - Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **ICD-9-CM Classification**: [ICD9Data.com - 2015 ICD-9-CM Codes](https://www.icd9data.com/2015/Volume1/default.htm)
- **Paper**: "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records" - Beata Strack et al., 2014

---

## 📌 Keywords

**Electronic Health Records (EHR)**, **Diabetes**, **30-Day Readmission**, **Machine Learning**, **LightGBM**, **XGBoost**, **Random Forest**, **Logistic Regression**, **ICD-9 Grouping**, **Class Imbalance**, **ADASYN**, **SHAP**, **Clinical Decision Support**

---



**Citation**:
```
[1] Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository https://doi.org/10.24432/C5230J.

[2] B. Strack et al., “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, pp. 1–11, 2014, doi: https://doi.org/10.1155/2014/781670.
‌
[3] Emi-Johnson, O. G., & Nkrumah, K. J. (2025). Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data. Cureus, 17(4), e82437. https://doi.org/10.7759/cureus.82437.
```

---

*Last Updated: 2026*
