import functions
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
BASE_DIR = Path("Heart attack and diseases model").resolve().parent
scaler_path = BASE_DIR / "scaler.joblib"
scaler = joblib.load(scaler_path)
BASE_DIR = Path("Heart attack and diseases model").resolve().parent
model_path = BASE_DIR / "disease_model.joblib"
model = joblib.load(model_path)
while True:
    try:
        age = functions.get_float_input("Enter your age: ")
        gender = functions.get_float_input("Enter your gender (Male = 1/Female = 0): ")
        glucose_mg_dl = functions.get_float_input("Enter your glucose level (mg/dL): ")
        cholesterol_mg_dl = functions.get_float_input("Enter your cholesterol level (mg/dL): ")
        systolic_bp = functions.get_float_input("Enter your systolic blood pressure (mmHg): ")
        diastolic_bp = functions.get_float_input("Enter your diastolic blood pressure (mmHg): ")
        bmi = functions.get_float_input("Enter your BMI: ")
        heart_rate = functions.get_float_input("Enter your heart rate (bpm): ")
        smoking_status = functions.get_float_input("Enter your smoking status (Smoker = 1/Non-Smoker = 0): ")
        alcohol_consumption = functions.get_float_input("Enter your alcohol consumption (Yes = 1/No = 0): ")
        physical_activity = functions.get_float_input("Enter your physical activity level (high = 2 /medium = 1/ low = 0): ")
        family_history = functions.get_float_input("Enter your family history of heart disease (Yes = 1/No = 0): ")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

    Map = functions.MAP(systolic_bp, diastolic_bp)
    Rpp = functions.RPP(systolic_bp, heart_rate)
    Pp = functions.PP(systolic_bp, diastolic_bp)
    unhealthy_lifestyle_score = functions.UnhealthyLifeScore(smoking_status, alcohol_consumption, physical_activity)
    atherogenic_index_coefficient = functions.AtherogenicIndexCoefficient(cholesterol_mg_dl, systolic_bp)
    smoking_hypertension_interaction = functions.SmokingHypertensionInteraction(smoking_status, systolic_bp)
    cardiac_adiposity_proxy = functions.CardiacAdiposityProxy(bmi, heart_rate)
    cardiovascular_stress_index = functions.CardiovascularStressIndex(Map, heart_rate)
    columns = [
    'age', 'gender', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp',
    'diastolic_bp', 'heart_rate', 'alcohol_consumption', 'smoking_status',
    'bmi', 'physical_activity', 'family_history', 'MAP',
    'RPP Rate Pressure Product', 'PP Pulse Pressure', 'unhealthy_lifestyle_score',
    'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction',
    'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index'
    ]
    values = [age, gender, glucose_mg_dl, cholesterol_mg_dl, systolic_bp,
          diastolic_bp, heart_rate, alcohol_consumption, smoking_status,
          bmi, physical_activity, family_history, Map, Rpp, Pp,
          unhealthy_lifestyle_score, atherogenic_index_coefficient,
          smoking_hypertension_interaction, cardiac_adiposity_proxy,
          cardiovascular_stress_index]
    input_df = pd.DataFrame([values], columns=columns)
    scale_cols = ['age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp',
              'diastolic_bp', 'heart_rate', 'bmi', 'MAP',
              'RPP Rate Pressure Product', 'PP Pulse Pressure',
              'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction',
              'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index']
    scaled_input = scaler.transform(input_df[scale_cols])         
    nonscaled_input = input_df.drop(columns=scale_cols).values   
    final_input = np.concatenate((scaled_input, nonscaled_input), axis=1) 
    final_input = final_input.drop(columns=['gender', 'alcohol_consumption', 'heart_rate'])
    
    
    output = model.predict(final_input)
    print("Model output:", output)
    if output[0][0] > 0.5:
       print("The model predicts a high risk of heart disease.")
    else:
       print("The model predicts a low risk of heart disease.")
    diseasation = functions.get_yes_no("Do you want to run the model again? (yes/no): ")
    if diseasation == "no":
        print("Exiting the program.")
        break
    elif diseasation == "yes":
        print("Restarting the model...")
        continue
