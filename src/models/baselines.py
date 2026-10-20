"""
CerebroTwin - Baseline Physiological Models
Model 0: Mean Baseline Predictor
Model 1: Ridge Regularized Linear Predictor

CRITICAL DOMAIN RULES:
- MCA blood-flow velocity (MCAv in cm/s) is GROUND TRUTH ONLY — NEVER an input feature.
- Regularization is critical due to small cohort size (N=39 subjects).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


class MeanBaselineModel:
    """
    Model 0: Empirical Cohort Mean Predictor.
    
    Serves as the absolute lower benchmark for clinical prediction tasks.
    Predicts the training set mean across all targets (primary metrics + trajectory knots).
    """

    def __init__(self) -> None:
        self.means_: Optional[pd.Series] = None
        self.target_names_: List[str] = []
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "MeanBaselineModel":
        """
        Fit the baseline model by calculating empirical column-wise means.
        
        Args:
            X: Training features (subject_id as index).
            y: Training ground truth targets (MCAv-derived metrics and trajectory).
            
        Returns:
            self
        """
        self.feature_names_ = list(X.columns)
        if isinstance(y, pd.DataFrame):
            self.target_names_ = list(y.columns)
            self.means_ = y.mean(axis=0)
        else:
            self.target_names_ = ["target"]
            self.means_ = pd.Series([np.mean(y)], index=self.target_names_)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict training mean for all samples in X.
        
        Args:
            X: Input feature DataFrame.
            
        Returns:
            pd.DataFrame of predicted means matching X.index and self.target_names_.
        """
        if self.means_ is None:
            raise ValueError("Model 0 (MeanBaselineModel) must be fitted before predict().")

        n_samples = len(X)
        pred_mat = np.tile(self.means_.values, (n_samples, 1))
        return pd.DataFrame(pred_mat, index=X.index, columns=self.target_names_)

    def predict_trajectory(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return trajectory predictions (lower, median, upper) for knot points.
        For Model 0, lower = median = upper = training mean trajectory.
        """
        traj_cols = [c for c in self.target_names_ if c.startswith("traj_t")]
        if not traj_cols:
            traj_cols = [f"traj_t{t}" for t in range(0, 65, 5) if f"traj_t{t}" in self.target_names_]
        
        if not traj_cols:
            # Fallback 100% trajectory
            arr = np.full((len(X), 13), 100.0)
            return arr, arr, arr

        means_traj = self.means_[traj_cols].values
        traj_mat = np.tile(means_traj, (len(X), 1))
        return traj_mat, traj_mat, traj_mat

    def get_name(self) -> str:
        """Return human-readable model name."""
        return "Mean Baseline (Model 0)"


class RidgeBaselineModel:
    """
    Model 1: Ridge Regularized Linear Regression.
    
    Uses standard scaling and L2 regularization (alpha=10.0 default) to mitigate
    overfitting on the N=39 subject cohort.
    Multi-output support wraps all 6 primary targets and 13 trajectory knots.
    """

    def __init__(self, alpha: float = 10.0) -> None:
        """
        Initialize Ridge baseline model.
        
        Args:
            alpha: Regularization strength (default: 10.0).
        """
        self.alpha = float(alpha)
        self.scaler = StandardScaler()
        self.regressor = MultiOutputRegressor(Ridge(alpha=self.alpha, random_state=42))
        self.feature_names_: List[str] = []
        self.target_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "RidgeBaselineModel":
        """
        Fit scaled Ridge regression model across all targets.
        
        Args:
            X: Input feature DataFrame (subject_id as index).
            y: Ground truth target DataFrame.
            
        Returns:
            self
        """
        self.feature_names_ = list(X.columns)
        if isinstance(y, pd.DataFrame):
            self.target_names_ = list(y.columns)
            y_arr = y.values
        else:
            self.target_names_ = ["target"]
            y_arr = np.asarray(y).reshape(-1, 1)

        X_scaled = self.scaler.fit_transform(X.values)
        self.regressor.fit(X_scaled, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for all targets.
        
        Args:
            X: Input feature DataFrame.
            
        Returns:
            pd.DataFrame of predictions.
        """
        if not self.feature_names_:
            raise ValueError("Model 1 (RidgeBaselineModel) must be fitted before predict().")

        # Align columns
        X_aligned = X[self.feature_names_] if all(c in X.columns for c in self.feature_names_) else X
        X_scaled = self.scaler.transform(X_aligned.values)
        preds = self.regressor.predict(X_scaled)
        return pd.DataFrame(preds, index=X.index, columns=self.target_names_)

    def predict_trajectory(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return trajectory predictions (lower, median, upper).
        For Ridge linear regression, lower and upper can be bounded by predicted median.
        """
        preds_df = self.predict(X)
        traj_cols = [c for c in self.target_names_ if c.startswith("traj_t")]
        if not traj_cols:
            traj_cols = [f"traj_t{t}" for t in range(0, 65, 5) if f"traj_t{t}" in preds_df.columns]

        if not traj_cols:
            arr = np.full((len(X), 13), 100.0)
            return arr, arr, arr

        traj_mat = preds_df[traj_cols].values
        return traj_mat, traj_mat, traj_mat

    def get_coefficients(self) -> pd.DataFrame:
        """
        Extract standardized feature coefficients (weights) across all targets.
        
        Returns:
            pd.DataFrame with feature names as index and target names as columns.
        """
        if not hasattr(self.regressor, "estimators_"):
            raise ValueError("Model 1 must be fitted before inspecting coefficients.")

        coef_matrix = np.array([est.coef_ for est in self.regressor.estimators_]).T
        return pd.DataFrame(
            coef_matrix,
            index=self.feature_names_,
            columns=self.target_names_,
        )

    def get_name(self) -> str:
        """Return human-readable model name."""
        return "Ridge Regression (Model 1)"
