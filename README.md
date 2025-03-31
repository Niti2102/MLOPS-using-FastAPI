Car Price Prediction API
This repository contains a machine learning API built using FastAPI for predicting the price of a used car based on different features such as the car's year, fuel type, transmission type, etc. The API is deployed using Docker and is hosted on Render.

Project Overview
The Car Price Prediction API provides two endpoints that predict the price of a car using Linear Regression and Lasso Regression models. The API is designed to receive car details and return the predicted price.

Tech Stack
FastAPI: Framework for building the API.

Scikit-learn: Machine learning models for car price prediction.

Pandas: Data handling for pre-processing.

Docker: Containerization of the application.

Render: Cloud hosting for deployment.

pickle: For loading and saving machine learning models.

Prerequisites
You need the following tools to run this API:

Python 3.9 or higher

Docker (optional, for containerization)

Render account (optional, for cloud deployment)

Required Libraries
fastapi: FastAPI framework for building the API.

scikit-learn: For machine learning models.

uvicorn: ASGI server to run FastAPI.

pandas: Data handling.

pickle: For saving and loading models.
Troubleshooting
If you face an issue like "No open ports detected" during deployment, follow these steps:

Ensure the Dockerfile has the EXPOSE 8000 line.

Set the Start Command in Render to uvicorn car:app --host 0.0.0.0 --port 8000.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Project Structure:
bash
Copy
Edit
car-price-prediction-api/
│
├── car.py               # FastAPI app and ML model code
├── Dockerfile           # Dockerfile for containerization
├── requirements.txt     # Python dependencies
├── README.md            # Project description and usage
└── car_data.csv         # Dataset used for training the model (optional)
