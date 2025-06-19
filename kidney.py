from flask import Flask, render_template_string, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from io import StringIO

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load the kidney disease dataset
def load_dataset():
    data = """id,age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane,classification
1,48,80,1.02,1,0,normal,normal,notpresent,notpresent,121,36,1.2,137,4.63,15.4,44,7800,5.2,yes,no,no,good,no,no,ckd
2,60,80,1.01,2,3,abnormal,abnormal,notpresent,notpresent,423,53,1.8,137,5.73,11.3,32,6700,4.3,yes,yes,no,poor,yes,yes,ckd
3,55,70,1.02,0,0,normal,normal,notpresent,notpresent,117,56,1.6,139,4.8,12.6,38,7200,4.7,no,no,no,good,no,no,notckd"""
    return pd.read_csv(StringIO(data))

# Feature names expected by the model
FEATURE_NAMES = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
    'cad', 'appet', 'pe', 'ane'
]

# Load or train the model
def train_model():
    df = load_dataset()
    
    # Preprocessing
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
    
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0, 'ckd\t': 1})
    
    categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    
    # Clean data - remove \t and whitespace
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()
            df[col] = df[col].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'})
    
    # Fill numerical missing values with median
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
            df[col] = df[col].map({
                'normal': 1, 'abnormal': 0, 
                'yes': 1, 'no': 0, 
                'good': 1, 'poor': 0, 
                'present': 1, 'notpresent': 0
            })
    
    # Split data
    X = df[FEATURE_NAMES]
    y = df['classification']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

# Try to load model, otherwise train a new one
try:
    model = pickle.load(open('kidney_model.pkl', 'rb'))
    print("Loaded pre-trained model")
except:
    print("Training new model...")
    model = train_model()
    pickle.dump(model, open('kidney_model.pkl', 'wb'))

def preprocess_input(df):
    """Preprocess the input data to match model requirements"""
    df = df.copy()
    
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
    
    categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({
                'normal': 1, 'abnormal': 0, 
                'yes': 1, 'no': 0, 
                'good': 1, 'poor': 0, 
                'present': 1, 'notpresent': 0,
                '\tno': 0, '\tyes': 1
            })
    
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = np.nan
    
    for col in numerical_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    df = df[FEATURE_NAMES]
    
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part. Please upload a CSV file.', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file. Please choose a CSV file to upload.', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                
                # Check for required columns
                missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
                if missing_cols:
                    flash(f'Missing required columns: {", ".join(missing_cols)}', 'error')
                    return redirect(request.url)
                
                processed_df = preprocess_input(df)
                
                # Make predictions
                predictions = model.predict(processed_df)
                probabilities = model.predict_proba(processed_df)[:, 1]  # Probability of CKD
                
                # Add predictions to original dataframe
                result_df = df.copy()
                result_df['Prediction'] = ['CKD' if p == 1 else 'Not CKD' for p in predictions]
                result_df['Probability'] = [f"{p:.1%}" for p in probabilities]
                
                # Convert to HTML
                result_html = result_df.to_html(
                    classes='table table-striped table-responsive', 
                    index=False,
                    justify='center'
                )
                
                return render_template_string(HTML_TEMPLATE, 
                                           result_html=result_html, 
                                           show_results=True)
            
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
    
    return render_template_string(HTML_TEMPLATE, show_results=False)

# HTML Template with embedded CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kidney Disease Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            padding: 20px;
        }
        .header {
            background-color: #0d6efd;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .upload-card {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .results-card {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .instructions {
            background-color: #e7f1ff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .required-field {
            color: #dc3545;
            font-weight: bold;
        }
        .feature-list {
            columns: 3;
        }
        @media (max-width: 768px) {
            .feature-list {
                columns: 1;
            }
        }
        .table-responsive {
            max-height: 400px; /* Set a max height for the table */
            overflow-y: auto; /* Enable vertical scrolling */
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1>Chronic Kidney Disease Predictor</h1>
            <p class="lead">Upload patient data to predict kidney disease risk</p>
        </div>
        
        <div class="card upload-card">
            <div class="card-body">
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }}">{{ message }}</div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                
                <div class="instructions">
                    <h4><i class="bi bi-info-circle"></i> Instructions</h4>
                    <p>Upload a CSV file containing patient data with the following <span class="required-field">required</span> columns:</p>
                    <div class="feature-list">
                        <ul>
                            <li>age (years)</li>
                            <li>bp (blood pressure mm/Hg)</li>
                            <li>sg (specific gravity)</li>
                            <li>al (albumin 0-5)</li>
                            <li>su (sugar 0-5)</li>
                            <li>rbc (normal/abnormal)</li>
                            <li>pc (normal/abnormal)</li>
                            <li>pcc (present/notpresent)</li>
                            <li>ba (present/notpresent)</li>
                            <li>bgr (blood glucose mg/dl)</li>
                            <li>bu (blood urea mg/dl)</li>
                            <li>sc (serum creatinine mg/dl)</li>
                            <li>sod (sodium mEq/L)</li>
                            <li>pot (potassium mEq/L)</li>
                            <li>hemo (hemoglobin gms)</li>
                            <li>pcv (packed cell volume)</li>
                            <li>wc (white blood cell count)</li>
                            <li>rc (red blood cell count)</li>
                            <li>htn (hypertension yes/no)</li>
                            <li>dm (diabetes yes/no)</li>
                            <li>cad (coronary artery disease yes/no)</li>
                            <li>appet (appetite good/poor)</li>
                            <li>pe (pedal edema yes/no)</li>
                            <li>ane (anemia yes/no)</li>
                        </ul>
                    </div>
                </div>
                
                <form method="POST" enctype="multipart/form-data" class="mt-4">
                    <div class="mb-3">
                        <label for="file" class="form-label">Upload CSV file:</label>
                        <input class="form-control" type="file" id="file" name="file" accept=".csv" required>
                    </div>
                    <button type="submit" class="btn btn-primary btn-lg">
                        <i class="bi bi-upload"></i> Predict Kidney Disease
                    </button>
                </form>
            </div>
        </div>
        
        {% if show_results %}
        <div class="card results-card">
            <div class="card-body">
                <h2 class="card-title">Prediction Results</h2>
                <div class="table-responsive">
                    {{ result_html|safe }}
                </div>
                <div class="mt-3">
                    <a href="/" class="btn btn-secondary">
                        <i class="bi bi-arrow-left"></i> Upload Another File
                    </a>
                </div>
            </div>
        </div>
        {% endif %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5050)

