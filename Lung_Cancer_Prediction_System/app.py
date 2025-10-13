from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Initialize app
app = Flask(__name__)

# Load model
model = pickle.load(open("data.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs
        age = float(request.form['age'])
        pack_years = float(request.form['pack_years'])
        gender = request.form['gender']
        radon_exposure = request.form['radon_exposure']
        asbestos_exposure = request.form['asbestos_exposure']
        secondhand_smoke = request.form['secondhand_smoke']
        copd_diagnosis = request.form['copd_diagnosis']
        alcohol = request.form['alcohol']
        family_history = request.form['family_history']

        # Convert categorical to dummy (same as training)
        input_dict = {
            'age': age,
            'pack_years': pack_years,
            'gender_Male': 1 if gender == 'Male' else 0,
            'radon_exposure_Low': 1 if radon_exposure == 'Low' else 0,
            'radon_exposure_Medium': 1 if radon_exposure == 'Medium' else 0,
            'asbestos_exposure_Yes': 1 if asbestos_exposure == 'Yes' else 0,
            'secondhand_smoke_exposure_Yes': 1 if secondhand_smoke == 'Yes' else 0,
            'copd_diagnosis_Yes': 1 if copd_diagnosis == 'Yes' else 0,
            'alcohol_consumption_Moderate': 1 if alcohol == 'Moderate' else 0,
            'family_history_Yes': 1 if family_history == 'Yes' else 0
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Predict
        prediction = model.predict(input_df)[0]
        result = "Positive for Lung Cancer" if prediction == 1 else "Negative for Lung Cancer"

        return render_template('index.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)



