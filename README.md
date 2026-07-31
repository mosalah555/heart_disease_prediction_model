




# Heart Attack and Disease Risk Predictor

A Streamlit app that guesses your risk of heart disease using two different models — a Neural Network and a Random Forest — trained side by side so you can compare what each one thinks.

You plug in basic health numbers (age, blood pressure, cholesterol, BMI, smoking habits, etc.) and the app does the rest: it calculates a handful of extra clinical values behind the scenes, feeds everything into both models, and shows you both predictions at once.

## What it actually does

- Simple form where you type in your health data
- Automatically works out some extra numbers that doctors actually use, like:
  - Mean Arterial Pressure (MAP)
  - Rate Pressure Product (RPP)
  - Pulse Pressure (PP)
  - An "unhealthy lifestyle" score
  - Atherogenic index
  - Smoking + hypertension interaction
  - A cardiac adiposity proxy
  - A cardiovascular stress index
- Runs your data through both models and shows you what each one predicts, side by side

## How good are the models?

| Model          | Accuracy | Recall |
|----------------|----------|--------|
| Neural Network | 86%      | 87.34% |
| Random Forest  | 87.3%    | 86.58% |

Pretty close to each other, honestly — neither one clearly wins.

## Files in here

```
.
├── app.py                     # the actual Streamlit app
├── functions.py                # feature engineering helpers
├── NN_model.joblib              # trained neural network
├── NN_scaler.joblib             # scaler for the NN's inputs
├── random_forest_model.joblib   # trained random forest
└── README.md
```

## How to run it

First, install what you need:

```bash
pip install streamlit numpy pandas joblib scikit-learn tensorflow --break-system-packages
```

Then just run:

```bash
streamlit run app.py
```

It'll open a local link (usually `http://localhost:8501`) — click that and you're in.

1. Fill out the form with your health info
2. Hit **Predict**
3. Check what both models say

## Heads up

⚠️ This is a learning project, not a medical tool. Don't use it to actually diagnose anything — talk to a real doctor for that.
## photos from the program
<img width="1920" height="1020" alt="Screenshot 2026-07-16 002003" src="https://github.com/user-attachments/assets/3a957401-a3d4-47fb-ba5c-ffa40953a23a" />
<img width="1920" height="1020" alt="Screenshot 2026-07-16 001806" src="https://github.com/user-attachments/assets/fa0ce9bb-8f94-4e1a-a7c4-749a9ce86a9c" />

