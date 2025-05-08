import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('models/fire_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Input features used for prediction
features = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

@app.route('/')
def index():
    return render_template('index.html', features=features)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        try:
            # Collect and convert inputs
            data = [float(request.form.get(f)) for f in features]

            # Scale input
            scaled_data = scaler.transform([data])

            # Predict
            prediction = model.predict(scaled_data)[0]
            result = "🔥 Fire" if prediction == 1 else "🌿 No Fire"

            return render_template('result.html', prediction=result)
        except Exception as e:
            return f"Error: {e}"
    return render_template('index.html', features=features)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
