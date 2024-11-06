# app.py

from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

# Load the model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.json

    # Convert the data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Ensure the input data has the same feature columns as the training data
    required_columns = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'partial paresis', 'Gender_Male']
    
    for col in required_columns:
        if col not in input_data.columns:
            return jsonify({'error': f'Missing feature: {col}'}), 400

    # Make prediction and get probabilities
    probabilities = model.predict_proba(input_data)

    # Map predictions and probabilities to results
    prediction = 'Positive' if probabilities[0][1] >= 0.5 else 'Negative'
    positive_prob = probabilities[0][1] * 100  # Probability of positive class
    negative_prob = probabilities[0][0] * 100  # Probability of negative class

    return jsonify({
        'prediction': prediction,
        'positive_prob': round(positive_prob, 2),
        'negative_prob': round(negative_prob, 2)
    })


if __name__ == '__main__':
    app.run(debug=True)
