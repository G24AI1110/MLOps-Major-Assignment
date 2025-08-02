import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pickle

class QuantizedLinearModel:
    def __init__(self, coef_uint8, intercept_uint8, scales, zero_points, scale_i, zero_point_i):
        self.coef_uint8 = coef_uint8
        self.intercept_uint8 = intercept_uint8
        self.scales = scales
        self.zero_points = zero_points
        self.scale_i = scale_i
        self.zero_point_i = zero_point_i

    def predict(self, X):
        coef_float = np.array([
            (int(q) - int(zp)) * s
            for q, zp, s in zip(self.coef_uint8, self.zero_points, self.scales)
        ])
        intercept_float = (int(self.intercept_uint8) - int(self.zero_point_i)) * self.scale_i
        return X @ coef_float + intercept_float

def quantize_model(model):
    coef = model.coef_
    intercept = model.intercept_

    coef_uint8 = []
    scales = []
    zero_points = []

    for w in coef:
        w_min, w_max = min(w, 0), max(w, 0)
        scale = (w_max - w_min) / 255 if w_max != w_min else 1e-5
        zero_point = int(np.round(-w_min / scale))
        q = int(np.clip(np.round(w / scale + zero_point), 0, 255))

        coef_uint8.append(q)
        scales.append(scale)
        zero_points.append(zero_point)

    i_min, i_max = min(intercept, 0), max(intercept, 0)
    scale_i = (i_max - i_min) / 255 if i_max != i_min else 1e-5
    zero_point_i = int(np.round(-i_min / scale_i))
    intercept_uint8 = int(np.clip(np.round(intercept / scale_i + zero_point_i), 0, 255))

    return QuantizedLinearModel(coef_uint8, intercept_uint8, scales, zero_points, scale_i, zero_point_i)

def evaluate_model(model, X, y, title=""):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"{title}:\n  MSE: {mse:.4f}\n  R² Score: {r2:.4f}")
    return mse, r2

model_path = "model.joblib"
model = joblib.load(model_path)

X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

evaluate_model(model, X_test, y_test, "Original Model")
original_size = os.path.getsize(model_path) / 1024


quantized_model = quantize_model(model)
evaluate_model(quantized_model, X_test, y_test, "Quantized Model (per-weight uint8)")


with open("quantized_model.pkl", "wb") as f:
    pickle.dump(quantized_model, f)

quantized_size = os.path.getsize("quantized_model.pkl") / 1024
print(f"Original Model File Size: {original_size:.2f} KB")
print(f"Quantized Model File Size: {quantized_size:.2f} KB")
