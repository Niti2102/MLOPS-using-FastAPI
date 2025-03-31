import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI
app = FastAPI()
model_file_path='car_price.py'
# Load the models and scaler at the start
with open("linear_model.pkl", "rb") as f:
    linear_model = pickle.load(f)

with open("lasso_model.pkl", "rb") as f:
    lasso_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the input schema (expected input data)
class CarFeatures(BaseModel):
    Year: int
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: int  # 0 for Petrol, 1 for Diesel, 2 for CNG
    Seller_Type: int  # 0 for Dealer, 1 for Individual
    Transmission: int  # 0 for Manual, 1 for Automatic
    Owner: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API!"}

# Predict car price using linear regression model
@app.post("/predict/linear")
def predict_linear_price(car: CarFeatures):
    car_data = pd.DataFrame([{
        'Year': car.Year,
        'Present_Price': car.Present_Price,
        'Kms_Driven': car.Kms_Driven,
        'Fuel_Type': car.Fuel_Type,
        'Seller_Type': car.Seller_Type,
        'Transmission': car.Transmission,
        'Owner': car.Owner
    }])

    # Apply scaling if necessary (for the Linear model, no scaling required)
    car_scaled = scaler.transform(car_data)

    # Predict using the Linear Regression model
    predicted_price = linear_model.predict(car_scaled)
    return {"Predicted Price (Linear Regression)": predicted_price[0]}

# Predict car price using lasso regression model
@app.post("/predict/lasso")
def predict_lasso_price(car: CarFeatures):
    car_data = pd.DataFrame([{
        'Year': car.Year,
        'Present_Price': car.Present_Price,
        'Kms_Driven': car.Kms_Driven,
        'Fuel_Type': car.Fuel_Type,
        'Seller_Type': car.Seller_Type,
        'Transmission': car.Transmission,
        'Owner': car.Owner
    }])

    # Apply scaling (for Lasso model, scaling is required)
    car_scaled = scaler.transform(car_data)

    # Predict using the Lasso Regression model
    predicted_price = lasso_model.predict(car_scaled)
    return {"Predicted Price (Lasso Regression)": predicted_price[0]}

