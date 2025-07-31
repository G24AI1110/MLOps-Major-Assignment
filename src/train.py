# src/train.py

from src.utils import load_data, train_model, evaluate_model
import joblib
import os

def main():
    # Load and split the data
    X_train, X_test, y_train, y_test = load_data()

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    r2, rmse = evaluate_model(model, X_test, y_test)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Save the model to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "model.joblib")
    print("✅ Model saved to model.joblib")

if __name__ == "__main__":
    main()
