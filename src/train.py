# src/train.py

from src.utils import load_data, train_model, evaluate_model
import joblib
import os

def main():
    
    X_train, X_test, y_train, y_test = load_data()

    # Train the model
    model = train_model(X_train, y_train)

   
    r2, rmse = evaluate_model(model, X_test, y_test)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

   
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "model.joblib")
    print(" Model saved to model.joblib")

if __name__ == "__main__":
    main()
