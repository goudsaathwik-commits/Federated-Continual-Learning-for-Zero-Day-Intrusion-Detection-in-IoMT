from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def build_classical_ids(model_type: str = "random_forest", seed: int = 42):
    """
    Builds classical machine learning baseline classifiers for tabular IDS.
    """
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1)
    elif model_type == "logistic_regression":
        return LogisticRegression(max_iter=500, random_state=seed)
    else:
        raise ValueError(f"Unknown classical model type: {model_type}")
