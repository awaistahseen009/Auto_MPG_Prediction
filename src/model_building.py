import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.stats import shapiro
import pingouin as pg
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import mlflow
import mlflow.sklearn
import dagshub
from dotenv import load_dotenv
load_dotenv()
# ---------------------- CONFIG ---------------------- #
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")

# Initialize DAGsHub authentication once
dagshub.auth.add_app_token(token=DAGSHUB_TOKEN)
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

# Configure MLflow manually to ensure consistent experiment linkage
MLFLOW_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("AUTO_MPG")

# ---------------------- LOGGER ---------------------- #
def make_logger(name, level="INFO"):
    import logging
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

logger = make_logger("model_building", "DEBUG")

# ---------------------- MODEL COMPARISON ---------------------- #
def compare_models(scores_a, scores_b, model_a="Model_A", model_b="Model_B"):
    diffs = np.array(scores_a) - np.array(scores_b)
    normality_p = shapiro(diffs).pvalue if len(diffs) > 3 else 1.0

    ttest_res = pg.ttest(scores_a, scores_b, paired=True)
    wilcoxon_res = pg.wilcoxon(scores_a, scores_b)

    if normality_p > 0.05 and len(scores_a) >= 10:
        test_used = "Paired T-Test"
        p_val = ttest_res["p-val"].values[0]
    else:
        test_used = "Wilcoxon Test"
        p_val = wilcoxon_res["p-val"].values[0]

    significant = p_val < 0.05
    logger.info(f"Comparison: {model_a} vs {model_b} | {test_used} p={p_val:.4f}")
    return significant, test_used

# ---------------------- TRAINING PIPELINE ---------------------- #
def train_model(cv=True):
    train = pd.read_csv("data/clean_data/train_data.csv")
    X_train, y_train = train.drop("mpg", axis=1), train["mpg"]

    regressors = [
        {"model": LinearRegression(), "params": {}},
        {"model": Ridge(), "params": {"alpha": [0.1, 1.0, 10.0]}},
        {"model": Lasso(), "params": {"alpha": [0.1, 1.0, 10.0], "max_iter": [1000]}},
        {"model": ElasticNet(), "params": {"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.5]}},
        {"model": SVR(), "params": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]}},
        {"model": RandomForestRegressor(), "params": {"n_estimators": [100, 200]}},
        {"model": DecisionTreeRegressor(), "params": {"max_depth": [10, 20], "min_samples_split": [2, 5]}}
    ]

    model_names, r2_scores, best_params, trained_models = [], [], [], []

    # ------------- MAIN RUN ------------- #
    with mlflow.start_run(run_name="Model_Training_All", nested=False) as parent_run:
        parent_run_id = parent_run.info.run_id

        for reg in regressors:
            model = reg["model"]
            param_grid = reg["params"]
            grid = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            trained_models.append(best_model)
            model_name = model.__class__.__name__
            model_names.append(model_name)
            r2_scores.append(grid.best_score_)
            best_params.append(grid.best_params_)

            # Log each model as a nested run under same experiment
            with mlflow.start_run(run_name=model_name, nested=True, parent_run_id=parent_run_id):
                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("r2_score", grid.best_score_)

            logger.info(f"{model_name} | Best Params: {grid.best_params_} | R²: {grid.best_score_:.4f}")

        # Store training results in dataframe
        df = pd.DataFrame({"Model": model_names, "r2": r2_scores, "params": best_params, "model": trained_models})

        # Compare top 2 models
        top = df.sort_values(by="r2", ascending=False).head(2)
        m1, m2 = top.iloc[0], top.iloc[1]
        significant, test_used = compare_models([m1.r2], [m2.r2], m1.Model, m2.Model)

        best_model = m1.model if m1.r2 >= m2.r2 else m2.model
        best_model_name = best_model.__class__.__name__

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        logger.info(f"Saved {best_model_name} as best model")

        # Log final best model in same parent experiment
        with mlflow.start_run(run_name=f"Best_{best_model_name}", nested=True, parent_run_id=parent_run_id):
            mlflow.log_params(m1.params if m1.Model == best_model_name else m2.params)
            mlflow.log_metric("r2_score", m1.r2 if m1.Model == best_model_name else m2.r2)
            mlflow.sklearn.log_model(best_model, best_model_name)
            mlflow.log_artifact("models/best_model.pkl")

        logger.info("✅ All models trained and logged under the same experiment.")

if __name__ == "__main__":
    train_model(cv=True)
