"""Logistic Regression baseline for churn prediction."""
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """class_weight='balanced' re-weights the loss inversely to class
    frequency, which matters here since Churn=Yes is only 26.5% of the
    data; an unweighted model would default to favouring the majority
    class and miss most churners."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    model.fit(X_train, y_train)
    return model


def top_coefficients(model: LogisticRegression, feature_names, n: int = 8) -> pd.DataFrame:
    coefs = pd.Series(model.coef_[0], index=feature_names)
    ranked = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    return ranked.head(n).to_frame("coefficient")
