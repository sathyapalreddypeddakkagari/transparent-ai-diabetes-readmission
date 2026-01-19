import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Try to import SHAP, but handle gracefully if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # Warning will be shown after st.set_page_config()

# ---------- Load artifacts ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "final_features.json"

# Load model artifacts with error handling and caching
# Function will be decorated after Streamlit initialization
def _load_model_artifacts():
    """Load model artifacts (decorator applied after st.set_page_config)."""
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        if not FEATURES_PATH.exists():
            raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")
        
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH, "r") as f:
            final_features = json.load(f)
        return model, scaler, final_features
    except Exception as e:
        error_msg = f"Error loading model artifacts: {str(e)}"
        raise RuntimeError(error_msg) from e

# Initialize model variables (will be loaded when needed)
model = None
scaler = None
FINAL_FEATURES = None

# Primary features for evaluation (from correlation analysis and feature importance)
PRIMARY_FEATURES = [
    'age', 'time_in_hospital_log', 'num_medications_log', 'number_inpatient_log',
    'service_utilization_log', 'numchange', 'encounter_count', 'meds_per_diag',
    'hospital_per_age', 'num_medications|time_in_hospital', 'num_medications|number_diagnoses'
]

# ---------- Helper functions ----------
def group_icd9(code):
    """Enhanced ICD-9 grouping based on notebook implementation."""
    if pd.isna(code) or code == 'Missing' or code == '?' or code == '':
        return 'Missing'
    code_str = str(code)
    if code_str.startswith('250'):
        return 'Diabetes'
    elif code_str.startswith('390') or code_str.startswith('410') or code_str.startswith('428'):
        return 'Circulatory'
    elif code_str.startswith('460') or code_str.startswith('786'):
        return 'Respiratory'
    else:
        return 'Other'

def get_icd9_diagnosis_name(code):
    """Get ICD-9 diagnosis name from code using ICD-9-CM classification.
    Reference: https://www.icd9data.com/2015/Volume1/default.htm"""
    if pd.isna(code) or code == 'Missing' or code == '?' or code == '':
        return 'Missing/Unknown'
    
    code_str = str(code).strip()
    code_clean = code_str.replace('V', 'V').replace('E', 'E')
    
    # Comprehensive ICD-9-CM diagnosis mappings (based on icd9data.com structure)
    icd9_map = {
        # 001-139: Infectious And Parasitic Diseases
        '11': 'Tuberculosis', '112': 'Candidiasis',
        '117': 'Mycoses', '135': 'Sarcoidosis',
        # 140-239: Neoplasms
        '141': 'Malignant neoplasm of tongue', '150': 'Malignant neoplasm of esophagus',
        '151': 'Malignant neoplasm of stomach', '153': 'Malignant neoplasm of colon',
        '154': 'Malignant neoplasm of rectum', '155': 'Malignant neoplasm of liver',
        '156': 'Malignant neoplasm of gallbladder', '157': 'Malignant neoplasm of pancreas',
        '158': 'Malignant neoplasm of retroperitoneum', '160': 'Malignant neoplasm of nasal cavities',
        '161': 'Malignant neoplasm of larynx', '162': 'Malignant neoplasm of trachea, bronchus, and lung',
        '171': 'Malignant neoplasm of connective and other soft tissue', '172': 'Malignant melanoma of skin',
        '174': 'Malignant neoplasm of female breast', '180': 'Malignant neoplasm of cervix uteri',
        # 240-279: Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders
        '250': 'Diabetes mellitus', '250.0': 'Diabetes mellitus without complications',
        '250.00': 'Diabetes mellitus without complications, type II or unspecified type, not stated as uncontrolled',
        '250.01': 'Diabetes mellitus without complications, type I (juvenile type), not stated as uncontrolled',
        '250.02': 'Diabetes mellitus without complications, type II or unspecified type, uncontrolled',
        '250.03': 'Diabetes mellitus without complications, type I (juvenile type), uncontrolled',
        '250.1': 'Diabetes with ketoacidosis', '250.2': 'Diabetes with hyperosmolarity',
        '250.3': 'Diabetes with other coma', '250.4': 'Diabetes with renal manifestations',
        '250.5': 'Diabetes with ophthalmic manifestations', '250.6': 'Diabetes with neurological manifestations',
        '250.7': 'Diabetes with peripheral circulatory disorders', '250.8': 'Diabetes with other specified manifestations',
        '250.9': 'Diabetes with unspecified complication',
        '276': 'Disorders of fluid, electrolyte, and acid-base balance',
        # 280-289: Diseases Of The Blood And Blood-Forming Organs
        # 290-319: Mental Disorders
        # 320-389: Diseases Of The Nervous System And Sense Organs
        '38': 'Septicemia',
        # 390-459: Diseases Of The Circulatory System
        '401': 'Essential hypertension', '401.9': 'Essential hypertension, unspecified',
        '410': 'Acute myocardial infarction', '412': 'Old myocardial infarction',
        '414': 'Other forms of chronic ischemic heart disease', '415': 'Acute pulmonary heart disease',
        '427': 'Cardiac dysrhythmias', '428': 'Heart failure',
        '434': 'Occlusion of cerebral arteries', '440': 'Atherosclerosis',
        '453': 'Other venous embolism and thrombosis',
        # 460-519: Diseases Of The Respiratory System
        '460': 'Acute nasopharyngitis', '486': 'Pneumonia, organism unspecified',
        '491': 'Chronic bronchitis', '518': 'Other diseases of lung',
        # 520-579: Diseases Of The Digestive System
        # 580-629: Diseases Of The Genitourinary System
        '584': 'Acute kidney failure', '585': 'Chronic kidney disease',
        '599': 'Other disorders of urethra and urinary tract',
        # 630-679: Complications Of Pregnancy, Childbirth, And The Puerperium
        # 680-709: Diseases Of The Skin And Subcutaneous Tissue
        '682': 'Other cellulitis and abscess',
        # 710-739: Diseases Of The Musculoskeletal System And Connective Tissue
        '715': 'Osteoarthritis and allied disorders',
        # 740-759: Congenital Anomalies
        # 760-779: Certain Conditions Originating In The Perinatal Period
        # 780-799: Symptoms, Signs, And Ill-Defined Conditions
        '780': 'General symptoms', '786': 'Symptoms involving respiratory system and other chest symptoms',
        # 800-999: Injury And Poisoning
        '996': 'Complications peculiar to certain specified procedures',
        # V01-V91: Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services
        'V57': 'Care involving use of rehabilitation procedures',
    }
    
    # Exact match of Icd9 code
    if code_str in icd9_map:
        return icd9_map[code_str]
    
    # Matching by prefix (first 3 digits or decimal-based)
    code_prefix = code_str.split('.')[0] if '.' in code_str else code_str[:3]
    if len(code_prefix) >= 3 and code_prefix in icd9_map:
        return icd9_map[code_prefix]
    
    # Matching by first 3 digits (without decimal)
    if len(code_str) >= 3:
        prefix = code_str[:3]
        if prefix in icd9_map:
            return icd9_map[prefix]
    
    # Look2-digit prefix for codes like '11', '38'
    if len(code_str) >= 2:
        prefix2 = code_str[:2]
        if prefix2 in icd9_map:
            return icd9_map[prefix2]
    
    # Get ICD-9 category description based on code ranges (ICD-9-CM structure)
    def get_icd9_category(code_num):
        """Get category description based on ICD-9-CM code ranges."""
        try:
            # Remove 'V' or 'E' prefix if present
            if code_num.startswith('V'):
                num = int(code_num[1:]) if code_num[1:].isdigit() else 0
                if 1 <= num <= 91:
                    return 'Supplementary Classification: Factors Influencing Health Status'
                return 'Supplementary Classification'
            elif code_num.startswith('E'):
                return 'Supplementary Classification: External Causes of Injury'
            
            num = int(code_num.split('.')[0]) if '.' in code_num else int(code_num[:3])
            
            if 1 <= num <= 139:
                return 'Infectious And Parasitic Diseases'
            elif 140 <= num <= 239:
                return 'Neoplasms'
            elif 240 <= num <= 279:
                return 'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders'
            elif 280 <= num <= 289:
                return 'Diseases Of The Blood And Blood-Forming Organs'
            elif 290 <= num <= 319:
                return 'Mental Disorders'
            elif 320 <= num <= 389:
                return 'Diseases Of The Nervous System And Sense Organs'
            elif 390 <= num <= 459:
                return 'Diseases Of The Circulatory System'
            elif 460 <= num <= 519:
                return 'Diseases Of The Respiratory System'
            elif 520 <= num <= 579:
                return 'Diseases Of The Digestive System'
            elif 580 <= num <= 629:
                return 'Diseases Of The Genitourinary System'
            elif 630 <= num <= 679:
                return 'Complications Of Pregnancy, Childbirth, And The Puerperium'
            elif 680 <= num <= 709:
                return 'Diseases Of The Skin And Subcutaneous Tissue'
            elif 710 <= num <= 739:
                return 'Diseases Of The Musculoskeletal System And Connective Tissue'
            elif 740 <= num <= 759:
                return 'Congenital Anomalies'
            elif 760 <= num <= 779:
                return 'Certain Conditions Originating In The Perinatal Period'
            elif 780 <= num <= 799:
                return 'Symptoms, Signs, And Ill-Defined Conditions'
            elif 800 <= num <= 999:
                return 'Injury And Poisoning'
        except (ValueError, AttributeError):
            pass
        return None
    
    # Try to get category description
    category = get_icd9_category(code_str)
    if category:
        return f'{category} (ICD-9: {code_str})'
    
    # Return grouped category as final fallback
    group = group_icd9(code_str)
    if group != 'Other':
        return f'{group} (ICD-9: {code_str})'
    
    return f'Other (ICD-9: {code_str})'


def bin_specialty(spec):
    if spec in ['InternalMedicine', 'Cardiology', 'Family/GeneralPractice']:
        return spec
    else:
        return 'Other'

medication_cols = [
    'metformin', 'repaglinide', 'glimepiride', 'glipizide', 'glyburide',
    'pioglitazone', 'rosiglitazone', 'insulin'
]

# ---------- Demo patients (for quick testing) ----------
patients = [
    {
        "age": 4, "time_in_hospital": 3, "num_lab_procedures": 25, "num_procedures": 0,
        "num_medications": 5, "number_outpatient": 0, "number_emergency": 0,
        "number_inpatient": 0, "number_diagnoses": 2, "encounter_count": 1,
        "gender": 'Female', "change": 'No', "diabetesMed": 'No',
        "max_glu_serum": "Norm", "A1Cresult": "Norm",
        "medical_specialty": "Family/GeneralPractice",
        "diag_1": "250.00", "diag_2": "401.9", "diag_3": "486",
        "metformin": 0, "repaglinide": 0, "glimepiride": 0,
        "glipizide": 0, "glyburide": 0, "pioglitazone": 0,
        "rosiglitazone": 0, "insulin": 0
    },
    {
        "age": 6, "time_in_hospital": 4, "num_lab_procedures": 30, "num_procedures": 1,
        "num_medications": 6, "number_outpatient": 1, "number_emergency": 0,
        "number_inpatient": 1, "number_diagnoses": 3, "encounter_count": 2,
        "gender": 'Male', "change": 'No', "diabetesMed": 'Yes',
        "max_glu_serum": "Norm", "A1Cresult": "Norm",
        "medical_specialty": "InternalMedicine",
        "diag_1": "250.00", "diag_2": "401.9", "diag_3": "486",
        "metformin": 1, "repaglinide": 0, "glimepiride": 0,
        "glipizide": 0, "glyburide": 0, "pioglitazone": 0,
        "rosiglitazone": 0, "insulin": 0
    },
    {
        "age": 8, "time_in_hospital": 9, "num_lab_procedures": 55, "num_procedures": 2,
        "num_medications": 14, "number_outpatient": 2, "number_emergency": 2,
        "number_inpatient": 4, "number_diagnoses": 6, "encounter_count": 7,
        "gender": 'Male', "change": 'Ch', "diabetesMed": 'Yes',
        "max_glu_serum": ">300", "A1Cresult": ">7",
        "medical_specialty": "InternalMedicine",
        "diag_1": "250.00", "diag_2": "401.9", "diag_3": "486",
        "metformin": 0, "repaglinide": 0, "glimepiride": 0,
        "glipizide": 1, "glyburide": 0, "pioglitazone": 0,
        "rosiglitazone": 0, "insulin": 1
    }
]


def preprocess_single_patient(patient_dict):
    """Preprocess a single patient record for prediction."""
    # Safety check - ensure model artifacts are loaded
    if FINAL_FEATURES is None or scaler is None:
        raise RuntimeError("Model artifacts not loaded. Cannot preprocess patient data.")
    
    df_new = pd.DataFrame([patient_dict])

    # Log transforms
    for col in ['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_inpatient']:
        df_new[col + '_log'] = np.log1p(df_new[col])

    # Service utilization
    df_new['service_utilization'] = (
        df_new['number_outpatient'] +
        df_new['number_emergency'] +
        df_new['number_inpatient']
    )
    df_new['service_utilization_log'] = np.log1p(df_new['service_utilization'])

    # Numchange (medication changes)
    df_new['numchange'] = df_new[medication_cols].gt(0).sum(axis=1)

    # Ratio features
    df_new['meds_per_diag'] = df_new['num_medications'] / df_new['number_diagnoses'].replace(0, 1)
    df_new['hospital_per_age'] = df_new['time_in_hospital'] / df_new['age'].replace(0, 1)

    # ICD-9 grouping (for all three diagnoses)
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df_new[col + '_group'] = df_new[col].apply(group_icd9)

    # Specialty binning
    df_new['medical_specialty_group'] = df_new['medical_specialty'].apply(bin_specialty)

    # Encoding
    df_new['gender'] = df_new['gender'].replace({'Male': 1, 'Female': 0})
    df_new['diabetesMed'] = df_new['diabetesMed'].replace({'Yes': 1, 'No': 0})
    df_new['max_glu_serum'] = df_new['max_glu_serum'].replace(
        {'>200': 1, '>300': 1, 'Norm': 0, 'Not Tested': -1}
    )
    df_new['A1Cresult'] = df_new['A1Cresult'].replace(
        {'>7': 1, '>8': 1, 'Norm': 0, 'Not Tested': -1}
    )
    df_new['change'] = df_new['change'].replace({'No': 0, 'Ch': 1})

    # Interaction terms
    df_new['num_medications|time_in_hospital'] = (
        df_new['num_medications'] * df_new['time_in_hospital']
    )
    df_new['num_medications|number_diagnoses'] = (
        df_new['num_medications'] * df_new['number_diagnoses']
    )

    # One-hot encoding
    categorical_cols_for_ohe = [
        'medical_specialty_group', 'max_glu_serum',
        'A1Cresult', 'diag_1_group',
        'diag_2_group', 'diag_3_group'
    ]

    df_new = pd.get_dummies(df_new, columns=categorical_cols_for_ohe)

    # Align to FINAL_FEATURES
    df_processed = pd.DataFrame(columns=FINAL_FEATURES)
    df_processed = pd.concat([df_processed, df_new], ignore_index=True)
    df_processed = df_processed.fillna(0)
    df_final = df_processed[FINAL_FEATURES]

    # Scaling
    X_scaled = scaler.transform(df_final)
    return X_scaled, df_final


def get_shap_explanation(X_scaled, patient_features_df):
    """Generate SHAP explanation for the prediction."""
    if not SHAP_AVAILABLE or model is None:
        return None, None
    
    try:
        # Ensure X_scaled is 2D matrix (n_samples, n_features)
        if len(X_scaled.shape) == 1:
            X_scaled = X_scaled.reshape(1, -1)
        
        # Use TreeExplainer for XGBoost (much faster than KernelExplainer)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Handle binary classification: shap_values can be a list with class 1 values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get SHAP values for positive class (readmission)
        
        # If multiple samples, take the first one
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Get feature names
        feature_names = patient_features_df.columns.tolist()
        
        # Summary plot data
        shap_series = pd.Series(shap_values, index=feature_names)
        shap_series = shap_series.sort_values(key=abs, ascending=False)
        
        return shap_series, shap_values
    except Exception as e:
        st.warning(f"SHAP explanation failed: {str(e)}")
        return None, None


def load_diabetes_dataset(file) -> pd.DataFrame:
    return pd.read_csv(file, low_memory=False)


def safe_value_counts(df, col, top_n=10):
    if col in df.columns:
        return df[col].value_counts().head(top_n)
    return pd.Series(dtype="int64")

# Cache functions for data loading (functions defined, decorators applied after st.set_page_config)
def _load_raw_data():
    """Load raw diabetes dataset."""
    raw_data_path = BASE_DIR / "diabetic_data.csv"
    if raw_data_path.exists():
        return load_diabetes_dataset(raw_data_path)
    return None

def _load_cleaned_data():
    """Load cleaned diabetes dataset."""
    cleaned_data_path = BASE_DIR / "final1_data.csv"
    if not cleaned_data_path.exists():
        cleaned_data_path = BASE_DIR / "final_data.csv"
    if cleaned_data_path.exists():
        return load_diabetes_dataset(cleaned_data_path), cleaned_data_path.name
    return None, None


# ---------- Streamlit UI ----------
# CRITICAL: st.set_page_config() MUST be called first, before any other Streamlit commands
st.set_page_config(
    page_title="Transparent AI Models for Early Identification of High-Risk Diabetes Readmissions",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show SHAP import warning if needed (after st.set_page_config)
if not SHAP_AVAILABLE:
    st.warning("⚠️ SHAP not installed. Run: `pip install shap` for explainability features. App will work without SHAP explanations.")

# App initialization success message (can be hidden after verification)
st.success("✅ App initialized successfully")

# Inject custom CSS theme
st.markdown("""
<style>
@layer base {
  :root {
    --background: 240 5% 95%;
    --foreground: 240 5% 10%;
    --card: 240 5% 95%;
    --card-foreground: 240 5% 10%;
    --popover: 240 5% 95%;
    --popover-foreground: 240 5% 10%;
    --primary: 220 80% 50%;
    --primary-foreground: 220 90% 95%;
    --secondary: 260 25% 85%;
    --secondary-foreground: 240 20% 20%;
    --muted: 260 15% 90%;
    --muted-foreground: 260 10% 40%;
    --accent: 260 15% 90%;
    --accent-foreground: 240 5% 10%;
    --destructive: 0 70% 55%;
    --destructive-foreground: 0 0% 98%;
    --border: 260 25% 85%;
    --input: 260 25% 85%;
    --ring: 220 80% 50%;
    --radius: 0.5rem;
    --chart-1: 220 80% 50%;
    --chart-2: 260 40% 60%;
    --chart-3: 280 60% 70%;
    --chart-4: 200 60% 50%;
    --chart-5: 240 50% 50%;
    
    --sidebar-background: 240 5% 92%;
    --sidebar-foreground: 240 5% 38%;
    --sidebar-primary: 220 80% 50%;
    --sidebar-primary-foreground: 220 90% 95%;
    --sidebar-accent: 260 15% 87%;
    --sidebar-accent-foreground: 240 5% 10%;
    --sidebar-border: 260 25% 82%;
    --sidebar-ring: 220 80% 47%;
  }

  /* Navigation Bar Styles */
  .navbar {
    background: linear-gradient(135deg, hsl(220 80% 50%), hsl(260 60% 50%));
    color: hsl(var(--primary-foreground));
    padding: 1rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    border-bottom: 3px solid hsl(var(--primary));
  }

  .navbar h1 {
    color: white;
    font-size: 1.8rem;
    font-weight: 900;
    margin: 0;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    letter-spacing: 1px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }

  /* Copyright Footer */
  .copyright-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(135deg, hsl(240 10% 15%), hsl(240 10% 10%));
    color: hsl(var(--foreground));
    padding: 1rem 2rem;
    text-align: center;
    border-top: 3px solid hsl(var(--primary));
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.2);
    z-index: 999;
    font-size: 0.9rem;
  }

  .copyright-footer .copyright-text {
    color: hsl(var(--muted-foreground));
    font-weight: 500;
    letter-spacing: 1px;
    margin: 0;
  }

  .copyright-footer .author {
    color: hsl(var(--primary));
    font-weight: 600;
    text-decoration: none;
    transition: color 0.3s ease;
  }

  .copyright-footer .author:hover {
    color: hsl(var(--primary-foreground));
  }

  /* Add padding to main content to account for fixed footer */
  .main .block-container {
    padding-bottom: 5rem;
  }

  /* Tab Styling - Make tabs bold and larger */
  [data-testid="stTabs"] [role="tab"] {
    font-weight: 900 !important;
    font-size: 1.3rem !important;
    padding: 0.75rem 1.5rem !important;
  }

  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    font-weight: 900 !important;
    font-size: 1.35rem !important;
    color: hsl(var(--primary)) !important;
  }

  .dark {
    --background: 240 10% 5%;
    --foreground: 220 50% 90%;
    --card: 240 10% 5%;
    --card-foreground: 220 50% 90%;
    --popover: 240 10% 5%;
    --popover-foreground: 220 50% 90%;
    --primary: 220 80% 50%;
    --primary-foreground: 220 90% 95%;
    --secondary: 260 25% 25%;
    --secondary-foreground: 220 50% 85%;
    --muted: 260 15% 15%;
    --muted-foreground: 220 60% 70%;
    --accent: 260 15% 15%;
    --accent-foreground: 220 50% 90%;
    --destructive: 0 80% 50%;
    --destructive-foreground: 220 90% 95%;
    --border: 260 25% 25%;
    --input: 260 25% 25%;
    --ring: 220 80% 50%;
    --radius: 0.5rem;
    --chart-1: 220 70% 40%;
    --chart-2: 260 60% 50%;
    --chart-3: 240 50% 60%;
    --chart-4: 280 70% 70%;
    --chart-5: 200 50% 50%;
    
    --sidebar-background: 0 0% 0%;
    --sidebar-foreground: 220 50% 63%;
    --sidebar-primary: 220 80% 42%;
    --sidebar-primary-foreground: 220 90% 95%;
    --sidebar-accent: 260 15% 7%;
    --sidebar-accent-foreground: 220 50% 90%;
    --sidebar-border: 260 25% 17%;
    --sidebar-ring: 220 80% 42%;
  }
}

/* Apply theme colors to Streamlit components */
.stApp {
    background: hsl(var(--background));
    color: hsl(var(--foreground));
}

.main .block-container {
    background: hsl(var(--card));
    color: hsl(var(--card-foreground));
}

[data-testid="stSidebar"] {
    background: hsl(var(--sidebar-background));
    color: hsl(var(--sidebar-foreground));
}

button[kind="primary"] {
    background: hsl(var(--primary));
    color: hsl(var(--primary-foreground));
}

h1, h2, h3 {
    color: hsl(var(--foreground));
}

.stSelectbox label, .stRadio label, .stSlider label {
    color: hsl(var(--foreground));
}
</style>
""", unsafe_allow_html=True)

# Load model artifacts after Streamlit is initialized
# Apply cache decorators now that Streamlit is initialized (AFTER st.set_page_config)
try:
    load_model_artifacts = st.cache_resource(_load_model_artifacts)
except Exception as decorator_error:
    st.error(f"❌ Error setting up model caching: {str(decorator_error)}")
    load_model_artifacts = None

try:
    load_raw_data_cached = st.cache_data(_load_raw_data)
    load_cleaned_data_cached = st.cache_data(_load_cleaned_data)
except Exception as decorator_error:
    st.warning(f"⚠️ Warning: Error setting up data caching: {str(decorator_error)}")
    load_raw_data_cached = _load_raw_data  # Fallback without caching
    load_cleaned_data_cached = _load_cleaned_data  # Fallback without caching

# Try to load models, but don't crash if they're missing
try:
    if load_model_artifacts is not None:
        try:
            model, scaler, FINAL_FEATURES = load_model_artifacts()
        except Exception as e:
            st.error(f"❌ Error loading model artifacts: {str(e)}")
            st.error(f"Current directory: {BASE_DIR}")
            st.error(f"Files checked:")
            st.error(f"  - Model: {MODEL_PATH} (exists: {MODEL_PATH.exists()})")
            st.error(f"  - Scaler: {SCALER_PATH} (exists: {SCALER_PATH.exists()})")
            st.error(f"  - Features: {FEATURES_PATH} (exists: {FEATURES_PATH.exists()})")
            st.error(f"Please ensure all model files are in the repository.")
            # Don't stop - allow app to show error message
            model = None
            scaler = None
            FINAL_FEATURES = None
    else:
        model = None
        scaler = None
        FINAL_FEATURES = None
except Exception as init_error:
    st.error(f"❌ Critical error during initialization: {str(init_error)}")
    import traceback
    st.code(traceback.format_exc())
    model = None
    scaler = None
    FINAL_FEATURES = None

# Navigation Bar
st.markdown("""
<div class="navbar">
    <h1>Transparent AI Models for Early Identification of High-Risk Diabetes Readmissions</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar removed - using tabs for navigation instead

# Create tabs with clear labels matching sidebar navigation
tab_infer, tab_dashboard = st.tabs(["🔍 Patient EHR Data Input & Prediction", "📊 Dataset Analytics"])

with tab_infer:
    st.subheader("Patient EHR Data Input & Prediction")
    
    input_method = st.radio(
        "Input Method:",
        ["📝 Real-Time Patient Form", "⚡ Quick Demo Patients"],
        horizontal=True
    )
    
    patient = None
    
    if input_method == "📝 Real-Time Patient Form":
        st.markdown("### Enter Patient Electronic Health Record (EHR) Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demographics")
            age = st.number_input("Age Group (1-10: [0-10) to [90-100))", 1, 10, 5)
            gender = st.selectbox("Gender", ["Male", "Female"])
            change = st.selectbox("Medication Change", ["No", "Ch"])
            diabetesMed = st.selectbox("Diabetes Medication", ["No", "Yes"])
        
        with col2:
            st.markdown("#### Hospital Stay & Procedures")
            time_in_hospital = st.number_input("Time in Hospital (days)", 1, 14, 3)
            num_lab_procedures = st.number_input("Number of Lab Procedures", 0, 200, 25)
            num_procedures = st.number_input("Number of Procedures", 0, 10, 0)
            num_medications = st.number_input("Number of Medications", 0, 100, 5)
            number_diagnoses = st.number_input("Number of Diagnoses", 1, 20, 2)
            encounter_count = st.number_input("Encounter Count", 1, 20, 1)
        
        with col3:
            st.markdown("#### Utilization History")
            number_outpatient = st.number_input("Outpatient Visits (prior year)", 0, 50, 0)
            number_emergency = st.number_input("Emergency Visits (prior year)", 0, 50, 0)
            number_inpatient = st.number_input("Inpatient Visits (prior year)", 0, 50, 0)
        
        st.markdown("#### Test Results")
        col_test1, col_test2 = st.columns(2)
        with col_test1:
            max_glu_serum = st.selectbox("Max Glucose Serum", [">200", ">300", "Norm", "Not Tested"])
        with col_test2:
            A1Cresult = st.selectbox("A1C Result", [">7", ">8", "Norm", "Not Tested"])
        
        st.markdown("#### Medical Specialty & Diagnoses")
        col_spec, col_diag1, col_diag2, col_diag3 = st.columns(4)
        with col_spec:
            medical_specialty = st.selectbox(
                "Medical Specialty",
                ["InternalMedicine", "Cardiology", "Family/GeneralPractice", "Other"]
            )
        with col_diag1:
            diag_1 = st.text_input("Primary Diagnosis (ICD-9)", "250.00", help="E.g., 250.00 (Diabetes), 401.9 (Hypertension)")
        with col_diag2:
            diag_2 = st.text_input("Secondary Diagnosis (ICD-9)", "401.9")
        with col_diag3:
            diag_3 = st.text_input("Tertiary Diagnosis (ICD-9)", "486")
        
        st.markdown("#### Medications (1 = Yes, 0 = No)")
        med_cols = st.columns(4)
        med_dict = {}
        for i, med in enumerate(medication_cols):
            with med_cols[i % 4]:
                med_dict[med] = st.number_input(f"{med.capitalize()}", 0, 1, 0, key=f"med_{med}")
        
        if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
            patient = {
                "age": age, "time_in_hospital": time_in_hospital, 
                "num_lab_procedures": num_lab_procedures, "num_procedures": num_procedures,
                "num_medications": num_medications, "number_outpatient": number_outpatient,
                "number_emergency": number_emergency, "number_inpatient": number_inpatient,
                "number_diagnoses": number_diagnoses, "encounter_count": encounter_count,
                "gender": gender, "change": change, "diabetesMed": diabetesMed,
                "max_glu_serum": max_glu_serum, "A1Cresult": A1Cresult,
                "medical_specialty": medical_specialty,
                "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
                **med_dict
            }
    
    else:
        # Quick demo patients
        options = [f"Patient {i+1}" for i in range(len(patients))]
        idx = st.selectbox(
            "Select demo patient:", range(len(patients)), 
            format_func=lambda i: options[i]
        )
        patient = patients[idx]
        st.json(patient)
        
        if st.button("🔮 Predict Readmission Risk", type="primary"):
            pass  # Will process below
    
    if patient is not None:
        # Check if model is loaded
        if model is None or scaler is None or FINAL_FEATURES is None:
            st.error("❌ **Model not loaded!** Please check that all model files are in the repository.")
            st.error("Required files: `xgb_model.pkl`, `scaler.pkl`, `final_features.json`")
            st.stop()
        
        # Show ICD-9 Groupings
        st.markdown("---")
        st.markdown("### 📋 ICD-9 Diagnosis Groupings")
        diag_info = [
            ("Primary (diag_1)", patient["diag_1"]),
            ("Secondary (diag_2)", patient["diag_2"]),
            ("Tertiary (diag_3)", patient["diag_3"])
        ]
        for diag_name, diag_code in diag_info:
            group = group_icd9(diag_code)
            st.write(f"- **{diag_name}:** `{diag_code}` → **{group}**")
        
        # Process prediction
        with st.spinner("Processing patient data and generating prediction..."):
            X_scaled, patient_features_df = preprocess_single_patient(patient)
            proba = float(model.predict_proba(X_scaled)[0][1])
            pred = 1 if proba >= 0.60 else 0
            
            # Get SHAP explanation
            shap_series, shap_values = get_shap_explanation(X_scaled, patient_features_df)
        
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            if pred == 1:
                st.error(f"⚠️ **HIGH RISK** of 30-Day Readmission")
            else:
                st.success(f"✅ **LOW RISK** of 30-Day Readmission")
        
        with col_pred2:
            st.metric("Readmission Probability", f"{proba:.1%}", 
                     delta=f"{(proba - 0.60) * 100:.1f}%" if proba >= 0.60 else f"{(0.60 - proba) * 100:.1f}% below threshold",
                     delta_color="inverse")
        
        st.markdown(f"**Prediction (Binary):** {pred} (Threshold: 0.60)")
        st.markdown(f"**Probability:** {proba:.4f}")
        
        # Primary Features Evaluation
        st.markdown("---")
        st.markdown("### 📊 Primary Features Evaluation")
        primary_data = {}
        for feat in PRIMARY_FEATURES:
            if feat in patient_features_df.columns:
                idx_feat = list(patient_features_df.columns).index(feat)
                primary_data[feat] = {
                    "Value": patient_features_df.iloc[0, idx_feat],
                    "SHAP Value": shap_series[feat] if shap_series is not None and feat in shap_series.index else None
                }
        
        primary_df = pd.DataFrame(primary_data).T
        primary_df.index.name = "Feature"
        st.dataframe(primary_df, use_container_width=True)
        
        # SHAP Explanation
        if shap_series is not None:
            st.markdown("---")
            st.markdown("### 🔍 SHAP Explanation (Explainable AI)")
            st.markdown("SHAP (SHapley Additive exPlanations) shows how each feature contributes to the prediction.")
            
            # Top contributing features
            top_shap = shap_series.head(15).sort_values()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['red' if x > 0 else 'blue' for x in top_shap.values]
            ax.barh(range(len(top_shap)), top_shap.values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_shap)))
            ax.set_yticklabels(top_shap.index, fontsize=9)
            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
            ax.set_title('Top 15 Features Contributing to Readmission Prediction', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretation
            st.info(
                "**Interpretation:**\n"
                "- 🔴 **Red (Positive SHAP):** Features pushing prediction toward HIGH RISK\n"
                "- 🔵 **Blue (Negative SHAP):** Features pushing prediction toward LOW RISK\n"
                "- **Magnitude:** Larger absolute values indicate stronger influence"
            )
        
        # Clinical Recommendations
        st.markdown("---")
        st.markdown("### 💡 Clinical Decision Support Recommendations")
        if pred == 1:
            st.warning(
                "**High-Risk Patient Identified.** Consider:\n"
                "- Enhanced discharge planning\n"
                "- Post-discharge follow-up within 7 days\n"
                "- Medication reconciliation review\n"
                "- Patient education on self-management\n"
                "- Care coordination with primary care provider"
            )
        else:
            st.info(
                "**Low-Risk Patient.** Standard discharge protocols may be appropriate.\n"
                "Continue routine monitoring and follow standard care plans."
            )
    
    with st.expander("📦 Model Artifacts & Configuration"):
        st.write(f"- **Model:** `{MODEL_PATH.name}` (XGBoost Classifier)")
        st.write(f"- **Scaler:** `{SCALER_PATH.name}` (StandardScaler)")
        if FINAL_FEATURES is not None:
            st.write(f"- **Feature Schema:** `{FEATURES_PATH.name}` ({len(FINAL_FEATURES)} features)")
        else:
            st.write(f"- **Feature Schema:** `{FEATURES_PATH.name}` (Not loaded)")
        st.write(f"- **Primary Features for Evaluation:** {len(PRIMARY_FEATURES)} key features")


with tab_dashboard:
    st.subheader("Interactive Patient Data Dashboard")
    st.markdown(
        "Analytics and visualizations from the raw and cleaned EHR datasets. "
        "All data displayed in aggregated form (no PHI in logs)."
    )
    
    # Dataset Access Link
    st.markdown("---")
    st.markdown(
        "**📊 Dataset Access Link:** "
        "[Diabetes 130-US Hospitals for Years 1999-2008 - UCI Machine Learning Repository]"
        "(https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)"
    )
    st.caption("The dataset represents ten years (1999-2008) of clinical care at 130 US hospitals.")
    st.markdown("---")

    # Load raw data (diabetic_data.csv) for Dataset Snapshot
    # Use cached loading functions (defined at module level)
    df_raw = load_raw_data_cached()
    df, cleaned_data_name = load_cleaned_data_cached()
    
    if df is None:
        data_file = st.file_uploader("Upload cleaned data CSV (final1_data.csv or final_data.csv)", type=["csv"])
        if data_file is not None:
            df = load_diabetes_dataset(data_file)
            cleaned_data_name = data_file.name
    
    # Display Raw Dataset Snapshot (diabetic_data.csv)
    if df_raw is not None:
        st.write("### 📋 Dataset Snapshot (Raw Data: diabetic_data.csv)")
        st.info(f"**Raw Dataset:** {len(df_raw):,} encounter-level records with {len(df_raw.columns)} columns")
        st.dataframe(df_raw.head(50), use_container_width=True)
        st.caption(f"Showing first 50 rows of {len(df_raw):,} total records from `diabetic_data.csv`")
    
    # Display Cleaned Data Table
    if df is not None:
        st.markdown("---")
        st.write(f"### ✨ Cleaned & Processed Dataset ({cleaned_data_name if cleaned_data_name else 'uploaded file'})")
        st.success(f"**Cleaned Dataset:** {len(df):,} patient-level records with {len(df.columns)} engineered features")
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Showing first 50 rows of {len(df):,} total records from `{cleaned_data_name if cleaned_data_name else 'uploaded file'}`")

        total_rows = len(df)
        # Calculate unique patients - check for patient identifier column
        if "patient_nbr" in df.columns:
            total_patients = df["patient_nbr"].nunique()
        elif "patient_id" in df.columns:
            total_patients = df["patient_id"].nunique()
        elif "encounter_id" in df.columns:
            # If we only have encounter_id, each row is an encounter
            # But if this is patient-level aggregated data, each row = 1 patient
            total_patients = total_rows  # Assuming patient-level aggregation
        else:
            # If no patient identifier found, assume each row represents a unique patient
            # (This is typical for patient-level aggregated datasets)
            total_patients = total_rows
        # Readmission rate: Check if readmitted column is binary (0/1) or text ("<30", ">30", "NO")
        if "readmitted" in df.columns:
            if df["readmitted"].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Binary encoding: 1 = readmitted <30 days, 0 = not readmitted or >30 days
                readmitted_30 = (df["readmitted"] == 1).mean()
            else:
                # Text encoding: "<30" = readmitted <30 days
                readmitted_30 = (df["readmitted"] == "<30").mean() if "<30" in df["readmitted"].values else 0
        else:
            readmitted_30 = 0

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Encounters", f"{total_rows:,}")
        col_b.metric("Unique Patients", f"{total_patients:,}")
        col_c.metric("30-Day Readmission Rate", f"{readmitted_30:.1%}")

        st.write("### Filters")
        filtered_df = df.copy()
        if "race" in df.columns:
            races = sorted(df["race"].dropna().unique().tolist())
            race_sel = st.multiselect("Race", races, default=races[:3])
            if race_sel:
                filtered_df = filtered_df[filtered_df["race"].isin(race_sel)]
        if "gender" in df.columns:
            genders = sorted(df["gender"].dropna().unique().tolist())
            gender_sel = st.multiselect("Gender", genders, default=genders)
            if gender_sel:
                filtered_df = filtered_df[filtered_df["gender"].isin(gender_sel)]
        if "age" in df.columns:
            ages = sorted(df["age"].dropna().unique().tolist())
            age_sel = st.multiselect("Age Band", ages, default=ages[:4])
            if age_sel:
                filtered_df = filtered_df[filtered_df["age"].isin(age_sel)]

        st.write("### Readmission Distribution")
        if "readmitted" in filtered_df.columns:
            st.bar_chart(filtered_df["readmitted"].value_counts())
        else:
            st.warning("Column `readmitted` not found in the cleaned dataset.")

        st.write("### ICD-9 Diagnosis Groupings Distribution")
        if all(col in filtered_df.columns for col in ['diag_1', 'diag_2', 'diag_3']):
            diag_1_groups = filtered_df['diag_1'].apply(group_icd9).value_counts()
            diag_2_groups = filtered_df['diag_2'].apply(group_icd9).value_counts()
            diag_3_groups = filtered_df['diag_3'].apply(group_icd9).value_counts()
            
            col_diag1, col_diag2, col_diag3 = st.columns(3)
            with col_diag1:
                st.write("**Primary Diagnosis (diag_1)**")
                st.bar_chart(diag_1_groups)
            with col_diag2:
                st.write("**Secondary Diagnosis (diag_2)**")
                st.bar_chart(diag_2_groups)
            with col_diag3:
                st.write("**Tertiary Diagnosis (diag_3)**")
                st.bar_chart(diag_3_groups)

        st.write("### Utilization & Clinical Indicators")
        numeric_cols = [
            c
            for c in [
                "time_in_hospital",
                "num_lab_procedures",
                "num_medications",
                "number_emergency",
                "number_inpatient",
                "number_outpatient",
            ]
            if c in filtered_df.columns
        ]
        if numeric_cols:
            st.line_chart(filtered_df[numeric_cols].sample(min(500, len(filtered_df))))

        st.write("### Top Medical Specialties")
        specialty_counts = safe_value_counts(filtered_df, "medical_specialty", top_n=10)
        if not specialty_counts.empty:
            st.bar_chart(specialty_counts)

        st.write("### ICD-9 Primary Diagnoses (Raw Codes)")
        
        # Display ICD-9-CM Code Range Reference Table
        icd9_categories = pd.DataFrame({
            "Code Range": [
                "001-139", "140-239", "240-279", "280-289", "290-319",
                "320-389", "390-459", "460-519", "520-579", "580-629",
                "630-679", "680-709", "710-739", "740-759", "760-779",
                "780-799", "800-999", "V01-V91", "E000-E999"
            ],
            "Category Description": [
                "Infectious And Parasitic Diseases",
                "Neoplasms",
                "Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders",
                "Diseases Of The Blood And Blood-Forming Organs",
                "Mental Disorders",
                "Diseases Of The Nervous System And Sense Organs",
                "Diseases Of The Circulatory System",
                "Diseases Of The Respiratory System",
                "Diseases Of The Digestive System",
                "Diseases Of The Genitourinary System",
                "Complications Of Pregnancy, Childbirth, And The Puerperium",
                "Diseases Of The Skin And Subcutaneous Tissue",
                "Diseases Of The Musculoskeletal System And Connective Tissue",
                "Congenital Anomalies",
                "Certain Conditions Originating In The Perinatal Period",
                "Symptoms, Signs, And Ill-Defined Conditions",
                "Injury And Poisoning",
                "Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services",
                "Supplementary Classification Of External Causes Of Injury And Poisoning"
            ]
        })
        
        with st.expander("📚 ICD-9-CM Code Range Reference Table", expanded=False):
            st.dataframe(icd9_categories, use_container_width=True, hide_index=True)
        
        if "diag_1" in filtered_df.columns:
            diag_counts = safe_value_counts(filtered_df, "diag_1", top_n=30)
            if not diag_counts.empty:
                # Create a DataFrame with ICD-9 codes, counts, and diagnosis names
                diag_data = []
                for code, count in diag_counts.items():
                    diag_name = get_icd9_diagnosis_name(code)
                    diag_data.append({
                        "ICD-9 Code": code,
                        "Diagnosis Name": diag_name,
                        "Count": count,
                        "Percentage": f"{(count / len(filtered_df) * 100):.2f}%"
                    })
                
                diag_df = pd.DataFrame(diag_data)
                st.dataframe(diag_df, use_container_width=True, hide_index=True)
                
                # Also show bar chart
                st.bar_chart(diag_counts)
                
                # Show all unique ICD-9 codes in the dataset
                with st.expander("📋 View All ICD-9 Codes in Dataset"):
                    all_codes = filtered_df["diag_1"].dropna().unique()
                    all_codes = sorted([str(c) for c in all_codes if str(c) not in ['?', 'Missing', '']])
                    
                    all_diag_data = []
                    for code in all_codes:
                        diag_name = get_icd9_diagnosis_name(code)
                        count = (filtered_df["diag_1"] == code).sum()
                        all_diag_data.append({
                            "ICD-9 Code": code,
                            "Diagnosis Name": diag_name,
                            "ICD-9 Group": group_icd9(code),
                            "Count": count,
                            "Percentage": f"{(count / len(filtered_df) * 100):.2f}%"
                        })
                    
                    all_diag_df = pd.DataFrame(all_diag_data)
                    all_diag_df = all_diag_df.sort_values("Count", ascending=False)
                    st.dataframe(all_diag_df, use_container_width=True, hide_index=True)
                    
                    st.caption(f"Total unique ICD-9 codes in dataset: {len(all_codes)}")

# Copyright Footer
st.markdown("---")
st.markdown("""
<div class="copyright-footer">
    <p class="copyright-text">
        © 2026 <span class="author">sathyapalreddy@2026</span> | 
        Transparent AI Models for Early Identification of High-Risk Diabetes Readmissions
    </p>
</div>
""", unsafe_allow_html=True)
