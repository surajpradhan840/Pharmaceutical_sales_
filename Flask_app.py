from flask import Flask, request, jsonify
import mlflow
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

# Load the models from MLflow artifacts
rf_model_uri = "runs:/2c4746095543442f84fa4596ff0e8147/rossman_sales_model"
loaded_rf_model = mlflow.sklearn.load_model(rf_model_uri)

# Load the trained LSTM model
lstm_model_path = "LSTM_model.pickle"
if os.path.exists(lstm_model_path):
    with open(lstm_model_path, "rb") as f:
        loaded_lstm_model = pickle.load(f)
else:
    loaded_lstm_model = None


@app.route("/", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract input data from JSON
        store_id = int(data["store_id"])
        is_holiday = int(data["is_holiday"])
        is_weekend = int(data["is_weekend"])
        is_promo = int(data["is_promo"])
        competition_distance = float(data["competition_distance"])
        competition_open_since_month = int(data["competition_open_since_month"])
        competition_open_since_year = int(data["competition_open_since_year"])

        # Create input features as an array
        input_data = np.array([[store_id, is_holiday, is_weekend, is_promo,
                                competition_distance, competition_open_since_month, 
                                competition_open_since_year]])

         # Make predictions using models
        rf_prediction = loaded_rf_model.predict(input_data) if loaded_rf_model else None
        lstm_prediction = loaded_lstm_model.predict(input_data) if loaded_lstm_model else None
        # Prepare the response
        response = {
            "rf_prediction": rf_prediction.tolist() if rf_prediction is not None else None,
            "lstm_prediction": lstm_prediction.tolist() if lstm_prediction is not None else None
        }

         # Create a plot for the LSTM model prediction
        plt.figure(figsize=(10, 6))
        plt.plot(y_val_pred, label='LSTM Prediction')
        plt.plot(y_val, label='Actual')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.title('LSTM Model Prediction vs Actual')
        plt.legend()
        plt.savefig('lstm_prediction_plot.png')

        # Create a plot for the Random Forest model prediction
        plt.figure(figsize=(10, 6))
        plt.plot(y_pred_rf, label='Random Forest Prediction')
        plt.plot(y_val, label='Actual')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.title('Random Forest Model Prediction vs Actual')
        plt.legend()
        plt.savefig('random_forest_prediction_plot.png')

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)