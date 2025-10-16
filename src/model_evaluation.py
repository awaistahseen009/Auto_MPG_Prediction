import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
from sklearn.metrics import r2_score
from config import make_logger

logger = make_logger("model_evaluation", "DEBUG")

def evaluate_model():
    # ---------------------- Load Test Data ---------------------- #
    logger.debug("Loading test data")
    test = pd.read_csv("data/test_data.csv")
    X_test = test.drop("mpg", axis=1)
    y_test = test["mpg"]

    # ---------------------- Load Trained Model ---------------------- #
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        return

    logger.debug("Loading trained model")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")

    # ---------------------- Evaluate Model ---------------------- #
    logger.debug("Evaluating model on test data")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"R2 Score on Test Data: {r2:.4f}")

    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    logger.debug(f"Sample of predictions:\n{results.head()}")

if __name__ == "__main__":
    evaluate_model()
