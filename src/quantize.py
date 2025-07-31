# src/quantize.py

import joblib
import numpy as np
import os
from sklearn.metrics import r2_score
from src.utils import load_data

# Load trained model
original_model_path = "models/model.joblib"
model = joblib.load(original_model_path)

# Load test data
_, X_test, _, y_test = load_data()

# Original evaluation
original_preds = model.predict(X_test)
original_r2 = r2_score(y_test, original_preds)

# Quantize coefficients and intercept
def quantize_array(arr, scale_factor=1000):
    return np.round(arr * scale_factor).astype(np.int32), scale_factor

def dequantize_array(arr, scale_factor):
    return arr.astype(np.float32) / scale_factor

weights = model.coef_
intercept = model.intercept_

quant_weights, w_scale = quantize_array(weights)
quant_intercept, i_scale = quantize_array(np.array([intercept]))

# Save quantized weights
quant_model_path = "models/quantized_model.npz"
np.savez_compressed(
    quant_model_path,
    weights=quant_weights,
    intercept=quant_intercept,
    w_scale=w_scale,
    i_scale=i_scale
)

# Dequantize for prediction
dequant_weights = dequantize_array(quant_weights, w_scale)
dequant_intercept = dequantize_array(quant_intercept, i_scale)[0]

quant_preds = X_test @ dequant_weights + dequant_intercept
quant_r2 = r2_score(y_test, quant_preds)

# File size calculations
original_size_kb = os.path.getsize(original_model_path) / 1024
quantized_size_kb = os.path.getsize(quant_model_path) / 1024

# Print results
print(f"Original R² Score: {original_r2:.4f}")
print(f"Quantized R² Score: {quant_r2:.4f}")
print(f"R² Drop: {(original_r2 - quant_r2):.4f}")
print(f"📦 Original model size: {original_size_kb:.2f} KB")
print(f"📦 Quantized model size: {quantized_size_kb:.2f} KB")
