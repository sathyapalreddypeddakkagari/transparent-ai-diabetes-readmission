
-- Input:  diabetes_data.csv
--   - 101,766 encounter-level records
--   - 50+ columns
--   - Multiple encounters per patient
--
-- Output: final_data.csv
--   - ~70,000 patient-level records (one row per patient)
--   - 30 engineered features + target variable (readmitted)
--   - Ready for machine learning model training
--
-- Key Transformations: (Database : PostgreSQL)
--   1. Missing value handling (replaced '?' with defaults)
--   2. Patient-level aggregation (GROUP BY patient_nbr)
--   3. Feature engineering (log transforms, ratios, interactions)
--   4. ICD-9 grouping (reduced 848+ codes to 5 categories)
--   5. One-hot encoding (categorical to binary features)

DROP TABLE IF EXISTS public.diabetes_data;

CREATE TABLE public.diabetes_data (
  encounter_id              bigint,
  patient_nbr               bigint,
  race                      text,
  gender                    text,
  age                       text,
  weight                    text,
  admission_type_id         int,
  discharge_disposition_id  int,
  admission_source_id       int,
  time_in_hospital          int,
  payer_code                text,
  medical_specialty         text,
  num_lab_procedures        int,
  num_procedures            int,
  num_medications           int,
  number_outpatient         int,
  number_emergency          int,
  number_inpatient          int,
  diag_1                    text,
  diag_2                    text,
  diag_3                    text,
  number_diagnoses          int,
  max_glu_serum             text,
  A1Cresult                 text,
  metformin                 text,
  repaglinide               text,
  nateglinide               text,
  chlorpropamide            text,
  glimepiride               text,
  acetohexamide             text,
  glipizide                 text,
  glyburide                 text,
  tolbutamide               text,
  pioglitazone              text,
  rosiglitazone             text,
  acarbose                  text,
  miglitol                  text,
  troglitazone              text,
  tolazamide                text,
  examide                   text,
  citoglipton               text,
  insulin                   text,
  glyburide_metformin       text,
  glipizide_metformin       text,
  glimepiride_pioglitazone  text,
  metformin_rosiglitazone   text,
  metformin_pioglitazone    text,
  change                    text,
  diabetesMed               text,
  readmitted                text
);

-- STEP 1: Load Raw Data into diabetes_data table(from CSV import)
-- COPY diabetes_data FROM 'C:\Users\satya\Downloads\AIT582-team_3-Final_Submission\Final-582_Project_code\diabetic_data.csv' WITH (FORMAT csv, HEADER true);
select * from diabetes_data;

-- STEP 2: Handle Missing Values (Replace '?' with defaults)
-- Replace '?' placeholders with appropriate default values for missing data
UPDATE diabetes_data
SET race = CASE 
    WHEN race = '?' OR race IS NULL THEN 'Unknown' 
    ELSE race 
END
WHERE race = '?' OR race IS NULL;

UPDATE diabetes_data
SET medical_specialty = CASE 
    WHEN medical_specialty = '?' OR medical_specialty IS NULL THEN 'Unknown' 
    ELSE medical_specialty 
END
WHERE medical_specialty = '?' OR medical_specialty IS NULL;

UPDATE diabetes_data
SET diag_1 = CASE 
    WHEN diag_1 = '?' OR diag_1 IS NULL THEN 'Missing' 
    ELSE diag_1 
END
WHERE diag_1 = '?' OR diag_1 IS NULL;

UPDATE diabetes_data
SET diag_2 = CASE 
    WHEN diag_2 = '?' OR diag_2 IS NULL THEN 'Missing' 
    ELSE diag_2 
END
WHERE diag_2 = '?' OR diag_2 IS NULL;

UPDATE diabetes_data
SET diag_3 = CASE 
    WHEN diag_3 = '?' OR diag_3 IS NULL THEN 'Missing' 
    ELSE diag_3 
END
WHERE diag_3 = '?' OR diag_3 IS NULL;

-- STEP 3: Remove Invalid Records
-- Exclude deceased patients (discharge_disposition_id = 11)
-- Exclude invalid gender entries

DELETE FROM diabetes_data
WHERE discharge_disposition_id = 11  -- Deceased patients
   OR gender = 'Unknown/Invalid';     -- Invalid gender entries

-- STEP 4: Encode Age from Brackets to Numeric (1-10)
-- Convert age bracket strings [0-10), [10-20), etc. to numeric values 1-10
-- First convert to text, then we'll alter column type later

UPDATE diabetes_data
SET age = CASE age
    WHEN '[0-10)'   THEN '1'
    WHEN '[10-20)'  THEN '2'
    WHEN '[20-30)'  THEN '3'
    WHEN '[30-40)'  THEN '4'
    WHEN '[40-50)'  THEN '5'
    WHEN '[50-60)'  THEN '6'
    WHEN '[60-70)'  THEN '7'
    WHEN '[70-80)'  THEN '8'
    WHEN '[80-90)'  THEN '9'
    WHEN '[90-100)' THEN '10'
    ELSE age
END
WHERE age IS NOT NULL;

-- Convert age column from text to integer after encoding
ALTER TABLE diabetes_data 
ALTER COLUMN age TYPE INTEGER USING age::INTEGER;

-- STEP 5: Encode Medication Columns (Binary: 0 = No, 1 = Steady/Up/Down)
-- Convert medication status from categorical to binary numeric

UPDATE diabetes_data
SET
  metformin = CASE WHEN metformin IN ('Steady','Up','Down') THEN '1' ELSE '0' END,
  repaglinide = CASE WHEN repaglinide IN ('Steady','Up','Down') THEN '1' ELSE '0' END,
  glimepiride = CASE WHEN glimepiride IN ('Steady','Up','Down') THEN '1' ELSE '0' END,
  glipizide = CASE WHEN glipizide IN ('Steady','Up','Down') THEN '1' ELSE '0' END,
  glyburide = CASE WHEN glyburide IN ('Steady','Up','Down') THEN '1' ELSE '0' END,
  pioglitazone = CASE WHEN pioglitazone IN ('Steady','Up','Down') THEN '1' ELSE '0' END,
  rosiglitazone = CASE WHEN rosiglitazone IN ('Steady','Up','Down') THEN '1' ELSE '0' END,
  insulin = CASE WHEN insulin IN ('Steady','Up','Down') THEN '1' ELSE '0' END
WHERE
  metformin IS NOT NULL;

ALTER TABLE public.diabetes_data
ALTER COLUMN metformin TYPE INTEGER USING metformin::INTEGER,
ALTER COLUMN repaglinide TYPE INTEGER USING repaglinide::INTEGER,
ALTER COLUMN glimepiride TYPE INTEGER USING glimepiride::INTEGER,
ALTER COLUMN glipizide TYPE INTEGER USING glipizide::INTEGER,
ALTER COLUMN glyburide TYPE INTEGER USING glyburide::INTEGER,
ALTER COLUMN pioglitazone TYPE INTEGER USING pioglitazone::INTEGER,
ALTER COLUMN rosiglitazone TYPE INTEGER USING rosiglitazone::INTEGER,
ALTER COLUMN insulin TYPE INTEGER USING insulin::INTEGER;


-- STEP 6: Encode Target and Change Variables
-- Convert readmitted to binary: <30 days = 1, else = 0
-- Convert change to binary: Ch = 1, No = 0

-- STEP 6: Encode Target and Change Variables (add new binary columns)

ALTER TABLE diabetes_data
ADD COLUMN IF NOT EXISTS readmitted_bin INTEGER,
ADD COLUMN IF NOT EXISTS change_bin INTEGER,
ADD COLUMN IF NOT EXISTS diabetesMed_bin INTEGER;

UPDATE diabetes_data
SET
  readmitted_bin = CASE
      WHEN readmitted = '<30' THEN 1
      WHEN readmitted IN ('>30','NO') THEN 0
      ELSE NULL
  END,
  change_bin = CASE
      WHEN change = 'Ch' THEN 1
      WHEN change = 'No' THEN 0
      ELSE NULL
  END,
  diabetesMed_bin = CASE
      WHEN diabetesMed = 'Yes' THEN 1
      WHEN diabetesMed = 'No' THEN 0
      ELSE NULL
  END;

-- STEP 7: Handle Test Results (Fill NULL with 'Not Tested')
-- Ensure test results have default values if missing

UPDATE diabetes_data
SET max_glu_serum = COALESCE(max_glu_serum, 'Not Tested')
WHERE max_glu_serum IS NULL OR max_glu_serum = '';

UPDATE diabetes_data
SET A1Cresult = COALESCE(A1Cresult, 'Not Tested')
WHERE A1Cresult IS NULL OR A1Cresult = '';

-- STEP 8: Patient-Level Aggregation (CRITICAL STEP)
-- Convert encounter-level data to patient-level by aggregating all encounters
-- per patient (patient_nbr). This is the core transformation.
--
-- Aggregation Strategy:
--   - MEAN:  Continuous variables that vary per encounter
--   - SUM:   Variables that accumulate across encounters
--   - MAX:   Binary/categorical flags (if ANY encounter has value)
--   - MIN:   Demographics and diagnoses (take first occurrence)

DROP TABLE IF EXISTS public.patient_aggregated;

CREATE TABLE public.patient_aggregated AS
SELECT
    patient_nbr,

    -- Numeric Aggregations
    AVG(time_in_hospital)     AS time_in_hospital,
    AVG(num_lab_procedures)   AS num_lab_procedures,
    AVG(num_procedures)       AS num_procedures,
    AVG(num_medications)      AS num_medications,
    AVG(number_diagnoses)     AS number_diagnoses,

    -- Accumulated Values
    SUM(number_outpatient)    AS number_outpatient,
    SUM(number_emergency)     AS number_emergency,
    SUM(number_inpatient)     AS number_inpatient,

    -- Sum of medication changes across encounters
    SUM(CASE WHEN change = 'Ch' OR change::text = '1' THEN 1 ELSE 0 END) AS change,

    -- Target: 1 if ANY encounter had <30
    MAX(CASE WHEN readmitted = '<30' OR readmitted::text = '1' THEN 1 ELSE 0 END) AS readmitted,

    -- Demographics (take one consistent value)
    MIN(gender)               AS gender,
    MIN(age)                  AS age,

    -- Test Results
    MIN(max_glu_serum)        AS max_glu_serum,
    MIN(A1Cresult)            AS A1Cresult,

    -- Diabetes Medication (ever)
    MAX(CASE WHEN diabetesMed = 'Yes' OR diabetesMed::text = '1' THEN 1 ELSE 0 END) AS diabetesMed,

    -- Medication Usage (ever used)
     MAX(metformin)      AS metformin,
    MAX(repaglinide)    AS repaglinide,
    MAX(glimepiride)    AS glimepiride,
    MAX(glipizide)      AS glipizide,
    MAX(glyburide)      AS glyburide,
    MAX(pioglitazone)   AS pioglitazone,
    MAX(rosiglitazone)  AS rosiglitazone,
    MAX(insulin)        AS insulin,

    -- Diagnoses / specialty
    MIN(diag_1)               AS diag_1,
    MIN(diag_2)               AS diag_2,
    MIN(diag_3)               AS diag_3,
    MIN(medical_specialty)    AS medical_specialty

FROM public.diabetes_data
GROUP BY patient_nbr;

-- STEP 9: Add Encounter Count (New Feature)
-- Count total number of encounters per patient (important predictor)

ALTER TABLE patient_aggregated
ADD COLUMN encounter_count INTEGER DEFAULT 0;

UPDATE patient_aggregated pa
SET encounter_count = (
    SELECT COUNT(*) 
    FROM diabetes_data d 
    WHERE d.patient_nbr = pa.patient_nbr
);

-- STEP 10: Create Helper Function for ICD-9 Grouping
-- Group ICD-9 diagnosis codes into clinical categories

CREATE OR REPLACE FUNCTION group_icd9(code VARCHAR) 
RETURNS VARCHAR AS $$
BEGIN
    IF code IS NULL OR code = 'Missing' OR code = '?' THEN
        RETURN 'Missing';
    ELSIF code LIKE '250%' THEN
        RETURN 'Diabetes';
    ELSIF code LIKE '390%' OR code LIKE '410%' OR code LIKE '428%' THEN
        RETURN 'Circulatory';
    ELSIF code LIKE '460%' OR code LIKE '786%' THEN
        RETURN 'Respiratory';
    ELSE
        RETURN 'Other';
    END IF;
END;
$$ LANGUAGE plpgsql;

-- STEP 11: Feature Engineering - Service Utilization
-- Calculate total service utilization as sum of all visit types

ALTER TABLE patient_aggregated
ADD COLUMN service_utilization INTEGER;

UPDATE patient_aggregated
SET service_utilization = number_outpatient + number_emergency + number_inpatient;

-- STEP 12: Feature Engineering - Numchange (Active Medication Count)
-- Count how many different medications patient is actively taking

ALTER TABLE patient_aggregated
ADD COLUMN numchange INTEGER;

UPDATE patient_aggregated
SET numchange = metformin + repaglinide + glimepiride + glipizide + 
                glyburide + pioglitazone + rosiglitazone + insulin;

-- STEP 13: Feature Engineering - Log Transformations
-- Apply log1p transformation to normalize skewed distributions
-- Only transform specific columns (not all columns need log transform)

ALTER TABLE patient_aggregated
ADD COLUMN time_in_hospital_log FLOAT,
ADD COLUMN num_lab_procedures_log FLOAT,
ADD COLUMN num_medications_log FLOAT,
ADD COLUMN number_inpatient_log FLOAT,
ADD COLUMN service_utilization_log FLOAT;

UPDATE patient_aggregated
SET time_in_hospital_log = LN(1 + time_in_hospital),
    num_lab_procedures_log = LN(1 + num_lab_procedures),
    num_medications_log = LN(1 + num_medications),
    number_inpatient_log = LN(1 + number_inpatient),
    service_utilization_log = LN(1 + service_utilization);

-- STEP 14: Feature Engineering - Ratio Features
-- Create clinically interpretable ratios

ALTER TABLE patient_aggregated
ADD COLUMN meds_per_diag FLOAT,
ADD COLUMN hospital_per_age FLOAT;

UPDATE patient_aggregated
SET meds_per_diag = num_medications / NULLIF(number_diagnoses, 0),
    hospital_per_age = time_in_hospital / NULLIF(age, 0);

-- STEP 15: Feature Engineering - Interaction Terms
-- Create interaction terms to capture complex relationships

ALTER TABLE patient_aggregated
ADD COLUMN num_medications_time_in_hospital FLOAT,
ADD COLUMN num_medications_number_diagnoses FLOAT;

UPDATE patient_aggregated
SET num_medications_time_in_hospital = num_medications * time_in_hospital,
    num_medications_number_diagnoses = num_medications * number_diagnoses;

-- STEP 16: ICD-9 Diagnosis Grouping
-- Apply ICD-9 grouping function to all three diagnosis columns

ALTER TABLE patient_aggregated
ADD COLUMN diag_1_group VARCHAR(50),
ADD COLUMN diag_2_group VARCHAR(50),
ADD COLUMN diag_3_group VARCHAR(50);

UPDATE patient_aggregated
SET diag_1_group = group_icd9(diag_1),
    diag_2_group = group_icd9(diag_2),
    diag_3_group = group_icd9(diag_3);

-- STEP 17: Medical Specialty Binning
-- Bin medical specialties into InternalMedicine, Cardiology, Family/GeneralPractice, Other

ALTER TABLE patient_aggregated
ADD COLUMN medical_specialty_group VARCHAR(50);

UPDATE patient_aggregated
SET medical_specialty_group = CASE 
    WHEN medical_specialty = 'InternalMedicine' THEN 'InternalMedicine'
    WHEN medical_specialty = 'Cardiology' THEN 'Cardiology'
    WHEN medical_specialty = 'Family/GeneralPractice' THEN 'Family/GeneralPractice'
    ELSE 'Other'
END;

-- STEP 18: Categorical Variable Encoding
-- Convert categorical variables to numeric for modeling

UPDATE patient_aggregated
SET gender = CASE WHEN gender = 'Male' THEN 1 ELSE 0 END;

-- Test results encoding (convert to numeric after aggregation)
ALTER TABLE patient_aggregated
ADD COLUMN max_glu_serum_num INTEGER,
ADD COLUMN A1Cresult_num INTEGER;

UPDATE patient_aggregated
SET max_glu_serum_num = CASE 
    WHEN max_glu_serum IN ('>200', '>300') THEN 1
    WHEN max_glu_serum = 'Norm' THEN 0
    WHEN max_glu_serum = 'Not Tested' THEN -1
    ELSE -1  -- Default to -1 for any other values
END;

UPDATE patient_aggregated
SET A1Cresult_num = CASE 
    WHEN A1Cresult IN ('>7', '>8') THEN 1
    WHEN A1Cresult = 'Norm' THEN 0
    WHEN A1Cresult = 'Not Tested' THEN -1
    ELSE -1  -- Default to -1 for any other values
END;

-- Drop old text columns and rename numeric ones
ALTER TABLE patient_aggregated
DROP COLUMN max_glu_serum,
DROP COLUMN A1Cresult;

ALTER TABLE patient_aggregated
RENAME COLUMN max_glu_serum_num TO max_glu_serum;

ALTER TABLE patient_aggregated
RENAME COLUMN A1Cresult_num TO A1Cresult;

-- STEP 19: One-Hot Encoding for Categorical Features
-- Create binary indicator columns for categorical variables
-- This prepares features for machine learning (30 final features)

-- Medical Specialty Group (drop_first: only create InternalMedicine and Other)
ALTER TABLE patient_aggregated
ADD COLUMN medical_specialty_group_InternalMedicine INTEGER DEFAULT 0,
ADD COLUMN medical_specialty_group_Other INTEGER DEFAULT 0;

UPDATE patient_aggregated
SET medical_specialty_group_InternalMedicine = CASE 
    WHEN medical_specialty_group = 'InternalMedicine' THEN 1 
    ELSE 0 
END,
medical_specialty_group_Other = CASE 
    WHEN medical_specialty_group = 'Other' THEN 1 
    ELSE 0 
END;

-- A1Cresult encoding (binary: >7 or >8 = 1)
ALTER TABLE patient_aggregated
ADD COLUMN A1Cresult_1 INTEGER DEFAULT 0;

UPDATE patient_aggregated
SET A1Cresult_1 = CASE 
    WHEN A1Cresult = 1 THEN 1 
    ELSE 0 
END;

-- Diagnosis Group One-Hot Encoding
ALTER TABLE patient_aggregated
ADD COLUMN diag_1_group_Other INTEGER DEFAULT 0,
ADD COLUMN diag_2_group_Diabetes INTEGER DEFAULT 0,
ADD COLUMN diag_2_group_Other INTEGER DEFAULT 0,
ADD COLUMN diag_3_group_Diabetes INTEGER DEFAULT 0,
ADD COLUMN diag_3_group_Other INTEGER DEFAULT 0;

UPDATE patient_aggregated
SET diag_1_group_Other = CASE WHEN diag_1_group = 'Other' THEN 1 ELSE 0 END,
    diag_2_group_Diabetes = CASE WHEN diag_2_group = 'Diabetes' THEN 1 ELSE 0 END,
    diag_2_group_Other = CASE WHEN diag_2_group = 'Other' THEN 1 ELSE 0 END,
    diag_3_group_Diabetes = CASE WHEN diag_3_group = 'Diabetes' THEN 1 ELSE 0 END,
    diag_3_group_Other = CASE WHEN diag_3_group = 'Other' THEN 1 ELSE 0 END;

-- STEP 20: Create Final Dataset with Selected Features

CREATE TABLE final_data AS
SELECT 
    patient_nbr,
    age, -- Demographics & Basic Info
    -- Log-transformed Features
    time_in_hospital_log,
    num_lab_procedures_log,
    num_medications_log,
    number_inpatient_log,
    service_utilization_log,
    -- Numeric Features
    num_procedures,
    number_outpatient,
    number_emergency,
    number_diagnoses,
    -- Engineered Features
    numchange,                          -- Active medication count
    encounter_count,                    -- Total encounters per patient
    meds_per_diag,                     -- Medication complexity ratio
    hospital_per_age,                   -- Hospitalization intensity ratio
    num_medications_time_in_hospital AS "num_medications|time_in_hospital",      -- Interaction term
    num_medications_number_diagnoses AS "num_medications|number_diagnoses",      -- Interaction term
    -- One-Hot Encoded Features
    medical_specialty_group_InternalMedicine,
    medical_specialty_group_Other,
    A1Cresult_1,
    diag_1_group_Other,
    diag_2_group_Diabetes,
    diag_2_group_Other,
    diag_3_group_Diabetes,
    diag_3_group_Other,
    -- Medication Features
    metformin,
    glipizide,
    glyburide,
    insulin,
    gender,
    change,                             -- Medication change indicator
    -- Target Variable
    readmitted                          -- Binary target: 1 = readmitted <30 days
    
FROM patient_aggregated;

-- STEP 21: Handle Missing Values in Final Dataset
-- Fill any remaining NULL values with 0 (for numeric) or appropriate defaults

UPDATE final_data
SET 
    age = COALESCE(age, 0),
    time_in_hospital_log = COALESCE(time_in_hospital_log, 0),
    num_lab_procedures_log = COALESCE(num_lab_procedures_log, 0),
    num_procedures = COALESCE(num_procedures, 0),
    num_medications_log = COALESCE(num_medications_log, 0),
    number_outpatient = COALESCE(number_outpatient, 0),
    number_emergency = COALESCE(number_emergency, 0),
    number_inpatient_log = COALESCE(number_inpatient_log, 0),
    number_diagnoses = COALESCE(number_diagnoses, 0),
    service_utilization_log = COALESCE(service_utilization_log, 0),
    numchange = COALESCE(numchange, 0),
    encounter_count = COALESCE(encounter_count, 1),
    meds_per_diag = COALESCE(meds_per_diag, 0),
    hospital_per_age = COALESCE(hospital_per_age, 0),
    "num_medications|time_in_hospital" = COALESCE("num_medications|time_in_hospital", 0),
    "num_medications|number_diagnoses" = COALESCE("num_medications|number_diagnoses", 0);

-- STEP 22: Export Final Dataset to CSV
-- Export final_data table to CSV file for use in model training

-- COPY final_data TO 'C:\Users\satya\Downloads\AIT582-team_3-Final_Submission\Final-582_Project_code\final_data.csv' WITH (FORMAT csv, HEADER true);

select * from final_data; -- to check the final dataset and download it from the database.

