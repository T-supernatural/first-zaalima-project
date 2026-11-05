"""Fold-local weighting for the notebook's XGBoost estimator."""
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class FoldBalancedXGBClassifier(XGBClassifier):
    """Compute balanced weights from the labels given to each fit call.

    Keeping this class in an importable module makes the Joblib pipeline usable
    from the notebook, command line and Streamlit without notebook definitions.
    """

    def fit(self, X, y, **kwargs):
        kwargs["sample_weight"] = compute_sample_weight(class_weight="balanced", y=y)
        return super().fit(X, y, **kwargs)
