import joblib
from utils import load_data

_, X_test, _, y_test = load_data()
model = joblib.load("artifacts/model.joblib")
preds = model.predict(X_test[:5])

print("Sample Predictions:", preds)
print("Actual Values     :", y_test[:5])