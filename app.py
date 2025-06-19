from flask import Flask, render_template_string, request, redirect, url_for
import pandas as pd
import joblib
import os

app = Flask(__name__)

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

# === Load model and scaler ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === HTML template ===
UPLOAD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Diabetes Prediction Upload</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background-color: #f9fafb;
      margin: 0;
      padding: 2rem;
      color: #333;
    }
    h1 {
      text-align: center;
      color: #1f2937;
      font-size: 2.5rem;
    }
    .container {
      max-width: 600px;
      margin: 0 auto;
      padding: 1rem;
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    input[type="file"] {
      width: 100%;
      padding: 10px;
      margin: 10px 0;
      border: 2px solid #6366f1;
      border-radius: 5px;
    }
    button {
      background-color: #6366f1;
      color: white;
      border: none;
      border-radius: 5px;
      padding: 10px 20px;
      cursor: pointer;
      font-size: 1rem;
    }
    button:hover {
      background-color: #4f46e5;
    }
  </style>
</head>
<body>
  <h1>Upload Patient Data for Diabetes Prediction</h1>
  <div class="container">
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required>
      <button type="submit">Upload and Predict</button>
    </form>
  </div>
</body>
</html>
"""

PATIENT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Diabetes Prediction Report</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background-color: #f9fafb;
      margin: 0;
      padding: 2rem;
      color: #333;
    }
    h1 {
      text-align: center;
      color: #1f2937;
      font-size: 2.5rem;
    }
    .container-caption {
      max-width: 800px;
      margin: 2rem auto 1rem;
      font-size: 1.5rem;
      font-weight: bold;
      border-bottom: 3px solid #6366f1;
      padding-bottom: 0.25rem;
    }
    .patient-list {
      max-width: 800px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 2rem;
    }
    .patient-card {
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
      padding: 1.5rem 2rem;
      display: grid;
      grid-template-columns: 180px 1fr;
      gap: 1rem 1.5rem;
    }
    .label {
      font-weight: 600;
      color: #4b5563;
      text-align: right;
      padding-right: 0.75rem;
      border-right: 2px solid #e5e7eb;
    }
    .value {
      color: #111827;
      font-size: 1rem;
    }
    .prediction-yes {
      color: #dc2626;
      font-weight: bold;
    }
    .prediction-no {
      color: #16a34a;
      font-weight: bold;
    }
    @media (max-width: 600px) {
      .patient-card {
        grid-template-columns: 1fr;
      }
      .label {
        text-align: left;
        border-right: none;
        padding-right: 0;
      }
    }
  </style>
</head>
<body>
  <h1>Diabetes Prediction Report</h1>
  <div class="container-caption">Patient Data & Predictions</div>
  <section class="patient-list">
    {% for patient in patients %}
    <div class="patient-card">
      <div class="label">Pregnancies</div>
      <div class="value">{{ patient.Pregnancies }}</div>

      <div class="label">Glucose</div>
      <div class="value">{{ patient.Glucose }}</div>

      <div class="label">Blood Pressure</div>
      <div class="value">{{ patient.BloodPressure }}</div>

      <div class="label">Skin Thickness</div>
      <div class="value">{{ patient.SkinThickness }}</div>

      <div class="label">Insulin</div>
      <div class="value">{{ patient.Insulin }}</div>

      <div class="label">BMI</div>
      <div class="value">{{ patient.BMI }}</div>

      <div class="label">Diabetes Pedigree Function</div>
      <div class="value">{{ patient.DiabetesPedigreeFunction }}</div>

      <div class="label">Age</div>
      <div class="value">{{ patient.Age }}</div>

      <div class="label">Prediction</div>
      <div class="value {{ 'prediction-yes' if patient.Prediction == 'Diabetic' else 'prediction-no' }}">
        {{ patient.Prediction }}
      </div>

      <div class="label">Probability (%)</div>
      <div class="value">{{ patient['Probability (%)'] }}%</div>
    </div>
    {% endfor %}
  </section>
</body>
</html>
"""

# === Main route to show upload form ===
@app.route("/", methods=["GET"])
def upload_form():
    return render_template_string(UPLOAD_TEMPLATE)

# === Route to handle file upload and prediction ===
@app.route("/predict", methods=["POST"])
def predict_from_csv():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected.", 400

    if not file.filename.endswith('.csv'):
        return "File must be a CSV.", 400

    # Read the CSV file
    df = pd.read_csv(file)

    # Check required columns
    required = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    if not all(col in df.columns for col in required):
        return "Missing required columns in the uploaded CSV.", 400

    # Scale and predict
    X = df[required]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    df['Prediction'] = ['Diabetic' if p == 1 else 'Not Diabetic' for p in preds]
    df['Probability (%)'] = (probs * 100).round(2)

    return render_template_string(PATIENT_TEMPLATE, patients=df.to_dict(orient='records'))

# === Run the server ===
if __name__ == "__main__":
    app.run(debug=True)
