"""Prediction-risk estimation for PreShift."""

import numpy as np

from sklearn.ensemble import RandomForestRegressor


class RiskPredictor:
    """
    Predict downstream model risk from shift features.

    Parameters
    ----------
    n_estimators : int, default=300
        Number of trees in the random forest.

    random_state : int, default=42
        Random seed.

    max_depth : int or None, default=5
        Maximum depth of each tree.

    min_samples_leaf : int, default=2
        Minimum number of samples required at a leaf.
    """

    def __init__(
        self,
        n_estimators=300,
        random_state=42,
        max_depth=5,
        min_samples_leaf=2,
        feature_names=None
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1
        )

        self.is_fitted = False
        self._feature_names = (
            list(feature_names)
            if feature_names is not None
            else None
        )

    def feature_names(self):
        """
        Return the names of features used by the risk model.

        Returns
        -------
        list[str] or None
            Feature names supplied during initialization.
        """

        if self._feature_names is None:
            return None

        return self._feature_names.copy()
    def fit(self, shift_features, observed_risk):
        """
        Train the risk predictor.

        Parameters
        ----------
        shift_features : array-like
            Shift feature matrix.

        observed_risk : array-like
            Observed downstream prediction risk,
            such as NRMSE.

        Returns
        -------
        RiskPredictor
            Fitted predictor.
        """

        X = np.asarray(
            shift_features,
            dtype=float
        )

        y = np.asarray(
            observed_risk,
            dtype=float
        )

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(
                "shift_features must be a 1D or 2D array."
            )

        if y.ndim != 1:
            raise ValueError(
                "observed_risk must be a 1D array."
            )

        if len(X) != len(y):
            raise ValueError(
                "shift_features and observed_risk "
                "must contain the same number of samples."
            )

        if len(X) == 0:
            raise ValueError(
                "Training data cannot be empty."
            )

        if (
            self._feature_names is not None
            and len(self._feature_names) != X.shape[1]
        ):
            raise ValueError(
                "Number of feature names must match "
                "the number of feature columns."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "shift_features must contain only finite values."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                "observed_risk must contain only finite values."
            )

        self.model.fit(X, y)

        self.is_fitted = True

        return self

    def predict(self, shift_features):
        """
        Estimate downstream prediction risk.

        Parameters
        ----------
        shift_features : array-like
            Shift feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted risk values.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "RiskPredictor must be fitted before prediction."
            )

        X = np.asarray(
            shift_features,
            dtype=float
        )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(
                "shift_features must be a 1D or 2D array."
            )

        if (
            self._feature_names is not None
            and X.shape[1] != len(self._feature_names)
        ):
            raise ValueError(
                "Number of prediction features must match "
                "the number of trained features."
            )

        return self.model.predict(X)