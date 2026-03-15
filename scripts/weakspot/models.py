"""Regression models for the weakspot experiment."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

AVAILABLE_MODELS = {
    "Ridge Regression": "ridge",
    "Random Forest": "rf",
    "Gradient Boosting": "gbm",
    "SVR": "svr",
    "MLP Neural Network": "mlp",
}


def build_model(key: str, complexity: float = 0.5, iterations: int = 100,
                random_state: int = 42):
    """
    Build a regression model.
    complexity: [0,1] – controls model capacity
    iterations: number of estimators / epochs
    """
    if key == "ridge":
        alpha = 10 ** (2 - 4 * complexity)  # [0.01, 100]
        return Pipeline([("scaler", StandardScaler()),
                         ("model", Ridge(alpha=alpha))])

    elif key == "rf":
        max_depth = max(2, int(2 + complexity * 18))  # [2, 20]
        return RandomForestRegressor(
            n_estimators=iterations, max_depth=max_depth, random_state=random_state
        )

    elif key == "gbm":
        max_depth = max(1, int(1 + complexity * 8))  # [1, 9]
        lr = 0.01 + 0.29 * complexity  # [0.01, 0.3]
        return GradientBoostingRegressor(
            n_estimators=iterations, max_depth=max_depth,
            learning_rate=lr, random_state=random_state
        )

    elif key == "svr":
        C = 10 ** (complexity * 3 - 1)  # [0.1, 100]
        return Pipeline([("scaler", StandardScaler()),
                         ("model", SVR(C=C, kernel="rbf"))])

    elif key == "mlp":
        h = max(8, int(complexity * 128))  # [8, 128]
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(h, h), max_iter=iterations,
                random_state=random_state, early_stopping=True
            ))
        ])

    else:
        raise ValueError(f"Unknown model key: {key}")


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Fit model, compute predictions and per-sample absolute errors."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    errors = np.abs(y_test - y_pred)
    metrics = {
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "R²": round(r2_score(y_test, y_pred), 4),
    }
    return y_pred, errors, metrics
