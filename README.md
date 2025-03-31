# Car Price Prediction API
This repository contains a machine learning model API for predicting the price of a used car based on various features like the car's year, fuel type, transmission, etc. The API is built using FastAPI and Docker, and is deployed on Render for seamless access.

# Project Overview
The Car Price Prediction API uses Linear Regression and Lasso Regression models to predict the price of used cars. The models are trained on a dataset containing various car features. The FastAPI app provides endpoints to predict the price using both regression models.

# Features:
* Linear Regression Model: Predicts car prices using a linear approach.

* Lasso Regression Model: Provides an alternative prediction using Lasso regularization.

* Scaling: The features are scaled for the Lasso model using StandardScaler.

# Tech Stack
* FastAPI: For creating the API endpoints.

* Scikit-learn: For machine learning models.

* Docker: For containerizing the application.

* Render: For deploying the Docker container to the cloud.

* Pandas: For data manipulation and handling.

# Prerequisites
To run the project locally, you'll need to install the following:

* Python 3.9 or higher

* Docker (for containerization)

* A Render account for deployment

# Requirements
* fastapi: FastAPI framework for building the API

* scikit-learn: Machine learning models and tools

* uvicorn: ASGI server to run FastAPI

* pandas: Data handling

* pickle: For serializing the models

# Troubleshooting
If you encounter an error like "No open ports detected" during deployment:

* Ensure that the EXPOSE 8000 line is present in your Dockerfile.

* Set the Start Command to uvicorn car:app --host 0.0.0.0 --port 8000 in Render.
# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Example structure:
![image](https://github.com/user-attachments/assets/7c076465-e2ae-4aa4-a3df-d204e4c62ba3)

