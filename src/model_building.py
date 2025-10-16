import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import numpy as np
from scipy.stats import shapiro
import pingouin as pg
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from config import make_logger

logger = make_logger("model_building", "DEBUG")

# ---------------------- Model Comparison Function ---------------------- #
def compare_models(scores_a, scores_b, model_a="Model_A", model_b="Model_B"):
    diffs = np.array(scores_a) - np.array(scores_b)
    normality_p = shapiro(diffs).pvalue if len(diffs) > 3 else 1.0

    ttest_res = pg.ttest(scores_a, scores_b, paired=True)
    wilcoxon_res = pg.wilcoxon(scores_a, scores_b)

    if normality_p > 0.05 and len(scores_a) >= 10:
        decision = "Normality OK - use paired t-test"
        p_val = ttest_res['p-val'].values[0]
        test_used = "Paired T-Test"
    else:
        decision = "Normality violated - use Wilcoxon test"
        p_val = wilcoxon_res['p-val'].values[0]
        test_used = "Wilcoxon Test"

    significant = p_val < 0.05
    logger.info(f"Comparison: {model_a} vs {model_b}")
    logger.info(f"Normality p-value: {normality_p:.4f}")
    logger.info(f"{test_used} p-value: {p_val:.4f} - {'Significant' if significant else 'Not Significant'}")
    logger.info(f"Decision: {decision}")
    return significant, test_used


# ---------------------- Model Training Function ---------------------- #
def train_model(cv=True):
    train = pd.read_csv("data/train_data.csv")
    X_train, y_train = train.drop("mpg", axis=1), train["mpg"]

    if cv:
        logger.debug("Training with cross fold validation")

        regressors = [
            {'model': LinearRegression(), 'params': {}},
            {'model': Ridge(), 'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}},
            {'model': Lasso(), 'params': {'alpha': [0.1, 1.0, 10.0], 'max_iter': [1000, 2000]}},
            {'model': ElasticNet(), 'params': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]}},
            {'model': SVR(), 'params': {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']}},
            {'model': RandomForestRegressor(), 'params': {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 10, 20, 30, 40]}},
            {'model': DecisionTreeRegressor(), 'params': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2, 3]}}
        ]

        model_names, r2_scores, fold_scores, best_params, trained_models = [], [], [], [], []

        for reg in regressors:
            model = reg['model']
            param_grid = reg['params']

            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            trained_models.append(best_model)

            best_index = grid_search.best_index_
            fold_r2_scores = [grid_search.cv_results_[f'split{i}_test_score'][best_index] for i in range(5)]

            model_names.append(model.__class__.__name__)
            r2_scores.append(grid_search.best_score_)
            fold_scores.append(fold_r2_scores)
            best_params.append(grid_search.best_params_)

            logger.info(f"{model.__class__.__name__}:")
            logger.info(f"Best Parameters: {grid_search.best_params_}")
            logger.info(f"Cross-Validated R2 Score: {grid_search.best_score_:.4f}")
            logger.info(f"R2 Scores per Fold: {[f'{s:.4f}' for s in fold_r2_scores]}")
            logger.info("-" * 60)

        data = pd.DataFrame({
            'Model': model_names,
            'Fold_Scores': fold_scores,
            'r2_score': r2_scores,
            'best_params': best_params,
            'trained_models': trained_models
        })

    else:
        logger.debug("Training without cross fold validation")

        regressors = [
            LinearRegression(),
            Ridge(),
            Lasso(),
            ElasticNet(),
            SVR(),
            RandomForestRegressor(),
            DecisionTreeRegressor()
        ]

        model_names, r2_scores, fold_scores, best_params, trained_models = [], [], [], [], []

        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        for model in regressors:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)

            model_names.append(model.__class__.__name__)
            r2_scores.append(r2)
            fold_scores.append([r2])
            best_params.append(None)
            trained_models.append(model)

            logger.info(f"{model.__class__.__name__}: R2 Score = {r2:.4f}")

        data = pd.DataFrame({
            'Model': model_names,
            'Fold_Scores': fold_scores,
            'r2_score': r2_scores,
            'best_params': best_params,
            'trained_models': trained_models
        })

    logger.info("Training completed")

    # ---------------------- Post-Training Model Comparison ---------------------- #
    top_models = data.groupby("Model")['r2_score'].mean().sort_values(ascending=False).iloc[:2].index.tolist()
    logger.info(f"Top 2 Models: {top_models}")

    model_a, model_b = top_models
    scores_a = data.loc[data['Model'] == model_a, 'Fold_Scores'].values[0]
    scores_b = data.loc[data['Model'] == model_b, 'Fold_Scores'].values[0]

    significant, test_used = compare_models(scores_a, scores_b, model_a, model_b)

    if cv and significant and test_used == "Paired T-Test":
        t_result = pg.ttest(scores_a, scores_b, paired=True, alternative='greater')
        logger.info(f"Performed one-sided right-tail T-Test, p-value: {t_result['p-val'].values[0]:.4f}")

    best_model_name = model_a if np.mean(scores_a) > np.mean(scores_b) else model_b
    logger.info(f"Best Model Selected: {best_model_name}")

    best_model = data.loc[data['Model'] == best_model_name, 'trained_models'].values[0]

    # ---------------------- Save Best Model ---------------------- #
    os.makedirs("models", exist_ok=True)
    logger.info("Starting model saving process")
    joblib.dump(best_model, "models/best_model.pkl")
    logger.info("Model saved successfully as models/best_model.pkl")


if __name__ == "__main__":
    train_model(cv=True)
