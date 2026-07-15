# Heart Attack and Disease Risk Predictor

A Streamlit web application that predicts a person's risk of heart disease using **two different machine learning models**: a **Neural Network** and a **Random Forest Classifier**. The app takes in health metrics (age, blood pressure, cholesterol, BMI, lifestyle factors, etc.), engineers additional clinical features, and returns a risk prediction from both models for comparison.

## Features

- Interactive web form to input patient health data
- Automatic calculation of derived clinical indicators:
  - MAP (Mean Arterial Pressure)
  - RPP (Rate Pressure Product)
  - PP (Pulse Pressure)
  - Unhealthy Lifestyle Score
  - Atherogenic Index Coefficient
  - Smoking-Hypertension Interaction
  - Cardiac Adiposity Proxy
  - Cardiovascular Stress Index
- Predicts heart disease risk using **two models**:
  - **Neural Network (NN)**
  - **Random Forest (RF)**
- Displays both predictions side by side for comparison

## Model Performance (Test Data)

| Model          | Accuracy | Recall |
|----------------|----------|--------|
| Neural Network | 86%      | 87.34% |
| Random Forest  | 87.3%    | 86.58% |

## Project Structure

```
.
├── app.py                     # Streamlit application
├── functions.py                # Feature engineering helper functions
├── NN_model.joblib              # Trained Neural Network model
├── NN_scaler.joblib             # Scaler used for NN input features
├── random_forest_model.joblib   # Trained Random Forest model
└── README.md
```

## Installation

1. Clone or download this repository.
2. Install the required dependencies:

```bash
pip install streamlit numpy pandas joblib scikit-learn tensorflow --break-system-packages
```

## Usage

Run the Streamlit app from the project folder:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`) in your browser.

1. Enter the requested health information in the form.
2. Click **Predict**.
3. View the risk prediction from both the Neural Network and Random Forest models.

## Disclaimer

⚠️ This tool is for **educational purposes only** and is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider regarding any medical condition.

## MIT License
## photos from the program
<img width="1920" height="1020" alt="Screenshot 2026-07-16 002003" src="https://github.com/user-attachments/assets/3a957401-a3d4-47fb-ba5c-ffa40953a23a" />
<img width="1920" height="1020" alt="Screenshot 2026-07-16 001806" src="https://github.com/user-attachments/assets/fa0ce9bb-8f94-4e1a-a7c4-749a9ce86a9c" />

