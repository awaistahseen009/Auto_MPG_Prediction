import pytest as test
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import r2_score
@test.mark.order(1)
def test_model_file_exists():
    model_path = Path("models/best_model.pkl")
    assert model_path.exists(), "Trained model not found!"
@test.mark.order(2)
def test_model_performance():
    model = joblib.load(open("models/best_model.pkl", "rb"))
    test_dataset = pd.read_csv("./data/clean_data/test_data.csv")
    X = test_dataset.drop("mpg", axis = 1)
    y_true = test_dataset['mpg']
    y_pred = model.predict(X)
    assert r2_score(y_true , y_pred)> 0.8, "Model R² dropped below acceptable threshold!"
