import os
from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

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
            model_name = request.form['model']
            open_price = float(request.form['open'])
            high = float(request.form['high'])
            low = float(request.form['low'])
            volume = float(request.form['volume'])

            input_data = pd.DataFrame({
                'Open': [open_price],
                'High': [high],
                'Low': [low],
                'Volume': [volume]
            })

            input_scaled = scaler.transform(input_data)
            model = models[model_name]
            prediction = model.predict(input_scaled)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            error = str(e)

    return render_template('index.html', models=models.keys(), prediction=prediction, error=error)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)