import functions
import numpy as np
import joblib
from pathlib import Path
age = input("Enter your age: ")
gender = input("Enter your gender (Male = 1/Female = 0): ")
glucose_mg_dl = input("Enter your glucose level (mg/dL): ")
cholesterol_mg_dl = input("Enter your cholesterol level (mg/dL): ")
systolic_bp = input("Enter your systolic blood pressure (mmHg): ")
diastolic_bp = input("Enter your diastolic blood pressure (mmHg): ")
bmi = input("Enter your BMI: ")
heart_rate = input("Enter your heart rate (bpm): ")
smoking_status = input("Enter your smoking status (Smoker = 1/Non-Smoker = 0): ")
alcohol_consumption = input("Enter your alcohol consumption (Yes = 1/No = 0): ")
physical_activity = input("Enter your physical activity level (high = 2 /medium = 1/ low = 0): ")
family_history = input("Enter your family history of heart disease (Yes = 1/No = 0): ")
Map = functions.MAP(systolic_bp, diastolic_bp)
Rpp = functions.RPP(systolic_bp, heart_rate)
Pp = functions.PP(systolic_bp, diastolic_bp)
unhealthy_lifestyle_score = functions.UnhealthyLifeScore(smoking_status, alcohol_consumption, physical_activity)
atherogenic_index_coefficient = functions.AtherogenicIndexCoefficient(cholesterol_mg_dl, systolic_bp)
smoking_hypertension_interaction = functions.SmokingHypertensionInteraction(smoking_status, systolic_bp)
cardiac_adiposity_proxy = functions.CardiacAdiposityProxy(bmi, heart_rate)
cardiovascular_stress_index = functions.CardiovascularStressIndex(Map, heart_rate)
input = [float(age), float(gender), float(glucose_mg_dl), float(cholesterol_mg_dl), float(systolic_bp), float(diastolic_bp), float(heart_rate), float(alcohol_consumption), float(smoking_status), float(bmi), float(physical_activity), float(family_history), float(Map), float(Rpp), float(Pp), float(unhealthy_lifestyle_score), float(atherogenic_index_coefficient), float(smoking_hypertension_interaction), float(cardiac_adiposity_proxy), float(cardiovascular_stress_index)]
input = np.array(input).reshape(1, 20)
BASE_DIR = Path("Heart attack and diseases model").resolve().parent
scaler_path = BASE_DIR / "scaler.joblib"
scaler = joblib.load(scaler_path)
input = scaler.transform(input) 
BASE_DIR = Path("Heart attack and diseases model").resolve().parent
model_path = BASE_DIR / "disease_model.joblib"
model = joblib.load(model_path)
output = model.predict(input)
print("Model output:", output)
if output[0] == 1:
    print("The model predicts a high risk of heart disease.")
else:
    print("The model predicts a low risk of heart disease.")