from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from werkzeug.exceptions import RequestEntityTooLarge
import os
import logging

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
model = None
scaler = None
selector = None
selected_features = None
optimal_threshold = 0.5

def load_models():
    """Load the trained models and preprocessing objects"""
    global model, scaler, selector, selected_features, optimal_threshold
    
    try:
        # Load the saved models
        model_path = 'optimized_heart_disease_model.pkl'
        scaler_path = 'robust_scaler.pkl'
        selector_path = 'feature_selector.pkl'
        threshold_path = 'optimal_threshold.pkl'
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found: {model_path}")
            
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        else:
            logger.error(f"Scaler file not found: {scaler_path}")
            
        if os.path.exists(selector_path):
            selector = joblib.load(selector_path)
            selected_features = selector.get_feature_names_out() if hasattr(selector, 'get_feature_names_out') else None
            logger.info("Feature selector loaded successfully")
        else:
            logger.error(f"Selector file not found: {selector_path}")
        
        if os.path.exists(threshold_path):
            optimal_threshold = joblib.load(threshold_path)
            logger.info(f"Optimal threshold loaded: {optimal_threshold:.4f}")
        else:
            optimal_threshold = 0.5
            logger.warning(f"Threshold file not found, using default: {optimal_threshold}")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

def create_features(df):
    """Apply feature engineering to input data"""
    df_feat = df.copy()
    
    # Age groups
    df_feat['age_group'] = pd.cut(df_feat['age'], bins=[0, 40, 55, 70, 100], 
                                 labels=[0, 1, 2, 3])
    df_feat['age_group'] = df_feat['age_group'].astype(int)
    
    # BMI categories
    df_feat['bmi_category'] = pd.cut(df_feat['bmi'], 
                                    bins=[0, 18.5, 25, 30, 100], 
                                    labels=[0, 1, 2, 3])
    df_feat['bmi_category'] = df_feat['bmi_category'].astype(int)
    
    # Blood pressure categories
    df_feat['bp_category'] = pd.cut(df_feat['trestbps'], 
                                   bins=[0, 120, 140, 180, 300], 
                                   labels=[0, 1, 2, 3])
    df_feat['bp_category'] = df_feat['bp_category'].astype(int)
    
    # Cholesterol categories
    df_feat['chol_category'] = pd.cut(df_feat['chol'], 
                                     bins=[0, 200, 240, 1000], 
                                     labels=[0, 1, 2])
    df_feat['chol_category'] = df_feat['chol_category'].astype(int)
    
    # Heart rate categories
    df_feat['hr_category'] = pd.cut(df_feat['thalach'], 
                                   bins=[0, 100, 150, 220], 
                                   labels=[0, 1, 2])
    df_feat['hr_category'] = df_feat['hr_category'].astype(int)
    
    # Risk combinations
    df_feat['risk_score'] = (df_feat['smoking'] + df_feat['diabetes'] + 
                            (df_feat['age'] > 55).astype(int) + 
                            (df_feat['sex'] == 1).astype(int))
    
    # Interaction features
    df_feat['age_sex'] = df_feat['age'] * df_feat['sex']
    df_feat['chol_age'] = df_feat['chol'] * df_feat['age']
    df_feat['bp_age'] = df_feat['trestbps'] * df_feat['age']
    
    return df_feat

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, 
                         thalach, exang, oldpeak, slope, ca, thal, 
                         smoking, diabetes, bmi):
    """
    Predict heart disease probability using the trained model
    """
    try:
        # Create feature array with original features
        original_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                      thalach, exang, oldpeak, slope, ca, thal, 
                                      smoking, diabetes, bmi]])
        
        # Create DataFrame for feature engineering
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 
                        'smoking', 'diabetes', 'bmi']
        
        input_df = pd.DataFrame(original_features, columns=feature_names)
        
        # Apply feature engineering
        engineered_df = create_features(input_df)
        
        # Get all features (removing heart_disease if it exists)
        engineered_features = engineered_df.drop('heart_disease', axis=1, errors='ignore')
        
        # For simplified version, if models are not loaded properly, use basic prediction
        if selector is not None:
            # Select features
            selected_features_array = selector.transform(engineered_features)
        else:
            # Use original features if selector not available
            selected_features_array = original_features
        
        # Scale features if scaler is available
        if scaler is not None:
            selected_features_array = scaler.transform(selected_features_array)
        
        # Make prediction using optimal threshold
        if model is not None:
            probability = model.predict_proba(selected_features_array)[0, 1]
            prediction = 1 if probability >= optimal_threshold else 0
        else:
            # Fallback prediction logic
            prediction = 1 if (age > 50 and chol > 240) or (smoking == 1 and diabetes == 1) else 0
            probability = 0.7 if prediction == 1 else 0.3
        
        # Determine risk level
        if probability < 0.3:
            risk_level = 'Low'
            risk_color = 'success'
        elif probability < 0.7:
            risk_level = 'Moderate'
            risk_color = 'warning'
        else:
            risk_level = 'High'
            risk_color = 'danger'
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'confidence': f"{probability:.1%}"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            'prediction': 0,
            'probability': 0.0,
            'risk_level': 'Unknown',
            'risk_color': 'secondary',
            'confidence': '0%',
            'error': str(e)
        }

@app.route('/')
def index():
    """Home page with hero section and about"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page with form"""
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'age': int(request.form['age']),
                'sex': int(request.form['sex']),
                'cp': int(request.form['cp']),
                'trestbps': int(request.form['trestbps']),
                'chol': int(request.form['chol']),
                'fbs': int(request.form['fbs']),
                'restecg': int(request.form['restecg']),
                'thalach': int(request.form['thalach']),
                'exang': int(request.form['exang']),
                'oldpeak': float(request.form['oldpeak']),
                'slope': int(request.form['slope']),
                'ca': int(request.form['ca']),
                'thal': int(request.form['thal']),
                'smoking': int(request.form['smoking']),
                'diabetes': int(request.form['diabetes']),
                'bmi': float(request.form['bmi'])
            }
            
            # Make prediction
            result = predict_heart_disease(**data)
            
            # Add input data to result for display
            result['input_data'] = data
            
            return render_template('result.html', result=result)
            
        except ValueError as e:
            flash(f'Please enter valid numeric values: {str(e)}', 'error')
            return render_template('predict.html')
        except Exception as e:
            flash(f'An error occurred during prediction: {str(e)}', 'error')
            return render_template('predict.html')
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        result = predict_heart_disease(
            data['age'], data['sex'], data['cp'], data['trestbps'],
            data['chol'], data['fbs'], data['restecg'], data['thalach'],
            data['exang'], data['oldpeak'], data['slope'], data['ca'],
            data['thal'], data['smoking'], data['diabetes'], data['bmi']
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/algorithms')
def algorithms():
    """Page explaining all advanced ML algorithms used"""
    return render_template('algorithms.html')

@app.route('/architecture')
def architecture():
    """Page showing system architecture and pipeline"""
    return render_template('architecture.html')

@app.route('/use-cases')
def use_cases():
    """Page showing real-world use cases and applications"""
    return render_template('use_cases.html')

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash('File too large. Please upload a smaller file.', 'error')
    return redirect(url_for('predict'))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
