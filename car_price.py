import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import os

# Check and print current directory
print("Current directory:", os.getcwd())

# Load dataset
car_dataset = pd.read_csv('car data.csv')

# Convert categorical variables to numeric
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Split data into features (X) and target variable (Y)
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Scale features for Lasso (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Save the trained Linear model
try:
    with open("linear_model.pkl", "wb") as f:
        pickle.dump(regressor, f)
    print("Linear model saved successfully.")
except Exception as e:
    print(f"Error saving linear model: {e}")

# Train Lasso Regression model
lasso_regressor = Lasso()
lasso_regressor.fit(X_train_scaled, Y_train)

# Save the trained Lasso model
try:
    with open("lasso_model.pkl", "wb") as f:
        pickle.dump(lasso_regressor, f)
    print("Lasso model saved successfully.")
except Exception as e:
    print(f"Error saving lasso model: {e}")

# Save the scaler (optional, if you used scaling)
try:
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved successfully.")
except Exception as e:
    print(f"Error saving scaler: {e}")

# Evaluate models (optional but recommended)
y_pred = regressor.predict(X_test)
print("Linear Regression Performance:")
print("Mean Squared Error:", metrics.mean_squared_error(Y_test, y_pred))
print("R^2 Score:", metrics.r2_score(Y_test, y_pred))

y_pred_lasso = lasso_regressor.predict(X_test_scaled)
print("Lasso Regression Performance:")
print("Mean Squared Error:", metrics.mean_squared_error(Y_test, y_pred_lasso))
print("R^2 Score:", metrics.r2_score(Y_test, y_pred_lasso))

# Load and test the saved Linear model (example)
try:
    with open("linear_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    print("Linear model loaded successfully.")
except Exception as e:
    print(f"Error loading linear model: {e}")

# Predict a new car price
new_car = pd.DataFrame({
    'Year': [2015],
    'Present_Price': [6.78],
    'Kms_Driven': [3000],
    'Fuel_Type': [0],  # Petrol
    'Seller_Type': [1],  # Individual
    'Transmission': [0],  # Manual
    'Owner': [0]
})

# If you used scaling, scale the new car data before prediction
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    new_car_scaled = scaler.transform(new_car)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")

predicted_price = loaded_model.predict(new_car_scaled)
print("Predicted Price using loaded model:", predicted_price)
