import pickle

with open("car_price.pkl", "rb") as f:
    models = pickle.load(f)

print(models)  # Should print a dictionary with 'linear' and 'lasso' models
