#  MLOps Major Assignment — Linear Regression on California Housing Dataset

This project demonstrates a complete end-to-end MLOps pipeline using a Linear Regression model on the California Housing dataset. It integrates model training, evaluation, quantization, Dockerization, and CI/CD automation through GitHub Actions.

---

##  Project Structure

- `src/train.py`: Trains the Linear Regression model, performs manual quantization, and saves the model.
- `src/predict.py`: Loads the trained model and evaluates its performance.
- `src/utils.py`: Contains reusable functions to load the dataset and evaluate the model.
- `tests/test_train.py`: Unit tests for utility functions and training logic.
- `models/`: Directory where the trained model (`model.pkl`) is stored.
- `Dockerfile`: Builds a container image that runs predictions.
- `.github/workflows/ci.yml`: Defines the GitHub Actions workflow to automate testing and Docker build.
- `requirements.txt`: Lists Python dependencies.

---

## Setup

A Conda environment (`mlops-major-env`) is used for local development. Dependencies are managed via the `requirements.txt` file.

---

##  Model Training

The `train.py` script:
- Loads the California housing dataset.
- Trains a Linear Regression model.
- Saves the model as `model.pkl` inside the `models/` folder.
- Applies manual quantization by rounding model coefficients.
- Prints model file size before and after quantization.

---

##  Model Prediction

The `predict.py` script:
- Loads the trained model.
- Evaluates it on the test dataset.
- Prints RMSE and R² score for model accuracy.

---

##  Unit Testing

The `tests/test_train.py` script validates:
- Data loading
- Model training
- Model evaluation
Using the `pytest` framework.

---

##  Manual Quantization

Quantization is manually performed in `train.py` by rounding the model's coefficients to two decimal places. This reduces file size and floating-point precision. The script also prints the model file size before and after quantization for comparison.

---

## Dockerization

The `Dockerfile` is used to containerize the project. It defines a minimal image based on Python 3.10, installs dependencies, and runs `predict.py` to verify the trained model. Although Docker is configured, it is **not required to be executed locally**; GitHub Actions handles the builds automatically.

---

##  CI/CD with GitHub Actions

CI/CD is managed via GitHub Actions with a `.yml` workflow file. The pipeline:
- Sets up Python
- Installs dependencies
- Runs pytest
- Executes predict.py
- Builds Docker image

This ensures automated testing and deployment in a production-like environment.

---

##  Final Comparison Table

| Metric                        | Before Quantization         | After Quantization   |
| ----------------------------- | --------------------------- | -------------------- |
| Model File (`model.pkl`) Size | 1.09 KB                     |  0.79KB              |
| Impact on Accuracy (R²)       | 0.5758                      | 0.5747               |

##  Requirements

Major libraries used in the project:
- scikit-learn
- pandas
- numpy
- joblib
- pytest

---

