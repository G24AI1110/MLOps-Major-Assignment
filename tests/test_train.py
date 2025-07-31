# tests/test_train.py

import pytest
from src import utils
from sklearn.linear_model import LinearRegression

def test_load_data_shapes():
    X_train, X_test, y_train, y_test = utils.load_data()
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

def test_model_instance():
    X_train, _, y_train, _ = utils.load_data()
    model = utils.train_model(X_train, y_train)
    assert isinstance(model, LinearRegression)

def test_model_performance():
    X_train, X_test, y_train, y_test = utils.load_data()
    model = utils.train_model(X_train, y_train)
    r2, rmse = utils.evaluate_model(model, X_test, y_test)
    assert r2 > 0.5  # adjust threshold if needed
