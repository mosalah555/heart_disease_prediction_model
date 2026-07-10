# Heart Disease Prediction Model

A machine learning project that predicts the risk of heart disease based on patient health metrics, using a neural network built with TensorFlow/Keras.

## Overview

This project trains a binary classification model to predict heart disease risk from a set of clinical and lifestyle features. It includes:

- Feature engineering (derived cardiovascular risk indicators)
- Data preprocessing and scaling
- A neural network model (with comparisons against Logistic Regression and Random Forest baselines)
- A command-line interface (`app.py`) for making live predictions from user input

## Project Structure

```
├── model.py       # Data preprocessing, feature engineering, model training, and evaluation
├── app.py         # CLI application for making predictions with the trained model
├── functions.py   # Helper functions (feature calculations, input validation)
├── disease_model.joblib   # Saved trained model
├── scaler.joblib          # Saved StandardScaler for preprocessing new inputs
└── heart_disease_model.keras  # Saved Keras model (native format)
```

## Features Used

### Raw input features
- `age`
- `gender` (Male = 1, Female = 0)
- `glucose_mg_dl`
- `cholesterol_mg_dl`
- `systolic_bp`
- `diastolic_bp`
- `heart_rate`
- `bmi`
- `smoking_status` (Smoker = 1, Non-Smoker = 0)
- `alcohol_consumption` (Yes = 1, No = 0)
- `physical_activity` (Low = 0, Medium = 1, High = 2)
- `family_history` (Yes = 1, No = 0)

### Engineered features
| Feature | Description |
|---|---|
| MAP | Mean Arterial Pressure |
| RPP (Rate Pressure Product) | Cardiac workload indicator |
| PP (Pulse Pressure) | Difference between systolic and diastolic BP |
| Unhealthy Lifestyle Score | Composite of smoking, alcohol use, and physical activity |
| Atherogenic Index Coefficient | Cholesterol/BP-based risk indicator |
| Smoking-Hypertension Interaction | Combined smoking and blood pressure effect |
| Cardiac Adiposity Proxy | BMI/heart rate-based risk indicator |
| Cardiovascular Stress Index | Combined MAP/heart rate stress indicator |

Feature importance analysis (via Random Forest) was used to drop low-impact features (`gender`, `alcohol_consumption`, `heart_rate`) before final model training, in order to reduce dimensionality and overfitting risk given the dataset size.

## Model

The final model is a **Sequential neural network** built with Keras:

```
Dense(16, activation='relu', kernel_regularizer=l2(0.01))
Dropout(0.4)
Dense(4, activation='relu')
Dropout(0.3)
Dense(1, activation='sigmoid')
```

- **Optimizer:** Adam (learning rate = 0.0005)
- **Loss:** Binary Crossentropy
- **Regularization:** L2 weight regularization + Dropout
- **Early Stopping:** Monitors validation loss (patience = 10, restores best weights)

### Data Split
- 70% training
- 15% validation
- 15% test

### Baseline Comparisons
The model was benchmarked against:
- **Logistic Regression** (initial baseline)
- **Random Forest Classifier** (used both as a comparison model and for feature importance analysis)

### Results

| Model | Test Accuracy | Test Recall |
|---|---|---|
| Logistic Regression (baseline) | — | — |
| Neural Network | ~86–88% | ~89–91% |
| Random Forest | ~89% | — |

Recall was prioritized alongside accuracy, since minimizing false negatives is critical in a health-risk prediction context.

## Requirements

```
numpy
pandas
scikit-learn
tensorflow
joblib
```

Install with:
```bash
pip install numpy pandas scikit-learn tensorflow joblib
```

## Usage

### Train the model
```bash
python model.py
```
This will preprocess the data, train the neural network, evaluate it on the test set, and save the trained model and scaler to disk.

### Run predictions (CLI)
```bash
python app.py
```
You'll be prompted to enter patient health metrics one by one. The model will output a predicted risk level (high/low) based on the trained neural network.

## Notes

- The dataset used contains ~1,000 patient records with 20 original features.
- Given the relatively small dataset size, the model uses dropout, L2 regularization, and early stopping to reduce overfitting.
- Feature scaling (`StandardScaler`) is applied only to continuous numerical features; binary/categorical features are left unscaled.

## Future Improvements

- Increase dataset size for better generalization
- Add k-fold cross-validation for more robust performance estimates
- Perform systematic hyperparameter tuning (e.g., Keras Tuner)
- Deploy as a web API instead of CLI-only interface
