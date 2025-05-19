from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Directory where models are saved
MODEL_DIR = './models'

# Load the scaler
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))

# Load all available models
models = {}
for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith('.joblib') and model_file != 'scaler.joblib':
        model_name = model_file.replace('.joblib', '')
        models[model_name] = joblib.load(os.path.join(MODEL_DIR, model_file))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            # Get form data
            model_name = request.form['model']
            open_price = float(request.form['open'])
            high = float(request.form['high'])
            low = float(request.form['low'])
            volume = float(request.form['volume'])

            # Prepare input data
            input_data = pd.DataFrame({
                'Open': [open_price],
                'High': [high],
                'Low': [low],
                'Volume': [volume]
            })

            # Scale the input data
            input_scaled = scaler.transform(input_data)

            # Make prediction
            model = models[model_name]
            prediction = model.predict(input_scaled)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            error = str(e)

    return render_template('index.html', models=models.keys(), prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)