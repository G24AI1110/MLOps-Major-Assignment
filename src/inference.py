X_sample = np.random.rand(1, len(q_coef))
prediction = np.dot(X_sample, dequantize_array(q_coef, scale)) + (q_intercept / scale)
print("Quantized Inference Output:", prediction)