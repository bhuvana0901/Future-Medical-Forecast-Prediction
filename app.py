import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Load the trained model
model = joblib.load('insurance_model.pkl')

# Serve the index.html file
@app.route('/')
def index():
    return send_from_directory('', 'index.html')

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the request
        data = request.get_json()
        
        # Prepare the feature array
        features = pd.DataFrame([{
            'age': float(data['age']),
            'sex': int(data['sex']),
            'bmi': float(data['bmi']),
            'children': int(data['children']),
            'smoker': int(data['smoker']),
            'region_northwest': 1 if data['region'] == 'northwest' else 0,
            'region_southeast': 1 if data['region'] == 'southeast' else 0,
            'region_southwest': 1 if data['region'] == 'southwest' else 0
        }])

        # Predict using the model
        prediction = model.predict(features)
        
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
