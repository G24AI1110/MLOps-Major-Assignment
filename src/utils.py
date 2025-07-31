# src/utils.py

# src/utils.py

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def load_data(test_size=0.2, random_state=42):
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return r2, rmse

