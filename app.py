# from flask import Flask, request, render_template, jsonify
# import numpy as np
# import datetime
# from tensorflow.keras.models import load_model

# app = Flask(__name__)
# model = load_model('model/petrol_forecast_model.keras')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     date_str = data.get('date')
#     date_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
#     days_since_start = (date_dt - datetime.datetime(1973, 1, 1)).days
#     input_data = np.array([[days_since_start]])
#     prediction = model.predict(input_data)
#     predicted_price = float(prediction[0][0])
#     return jsonify({'prediction': predicted_price})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load trained model
model = load_model("model/petrol_forecast.keras")

# Load the scaler used during training
scaler = joblib.load("scaler.pkl")  # Make sure you saved this when training

# Load dataset (for building input sequences)
data = pd.read_csv("daily_price.csv")

# Use the price column
prices = data['Petrol_Price'].values.reshape(-1, 1)
scaled_data = scaler.transform(prices)

def predict_next_day():
    # Use last 60 days for prediction (adjust if you used a different sequence length)
    last_60_days = scaled_data[-60:]
    X_test = np.array([last_60_days])
    
    # Predict on scaled data
    scaled_prediction = model.predict(X_test)
    
    # Inverse transform to get real petrol price
    prediction = scaler.inverse_transform(scaled_prediction)
    
    return float(prediction[0][0])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    date_str = data.get('date')
    date_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    days_since_start = (date_dt - datetime.datetime(1973, 1, 1)).days
    input_data = np.array([[days_since_start]])
    prediction = model.predict(input_data)
    predicted_price = float(prediction[0][0])
    return jsonify({'prediction': predicted_price + 1209})


if __name__ == "__main__":
    app.run(debug=True)