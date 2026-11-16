"""Random Forest and XGBoost benchmarks, for comparison against the
Logistic Regression baseline and the neural network."""
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state: int = 42) -> XGBClassifier:
    """scale_pos_weight = n_negative / n_positive mirrors the same
    class-imbalance correction used as class_weight='balanced' for the
    Logistic Regression and Random Forest models, so all four models are
    compared on equal footing rather than one being tuned for imbalance
    and the others not."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="logloss",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model
