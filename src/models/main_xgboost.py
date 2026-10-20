"""
CerebroTwin - Main Production XGBoost Digital Twin Model (Model 2)
Multi-output gradient boosted tree regression with quantile uncertainty estimation
and explainable AI (SHAP) integration.

CRITICAL DOMAIN RULES:
- ALWAYS use "MCA blood-flow velocity" or "MCAv" (cm/s), NEVER "cerebral blood flow".
- MCAv is GROUND TRUTH ONLY — NEVER a runtime input feature.
- Abnormality classification: "Research Response Status: HIGH DEVIATION" — NEVER clinical diagnosis.
- N=39 subjects: Strong L1/L2 regularization and shallow depth to prevent overfitting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class XGBoostTwinModel:
    """
    Model 2: Production XGBoost Digital Twin with Quantile Uncertainty.
    
    Architecture:
      - Independent gradient boosted regression trees for all primary targets
        (pct_mca_drop, mca_nadir_cms, time_to_nadir_s, cumulative_deficit_auc,
         recovery_time_s, recovery_status) and 13 trajectory knot points (traj_t0 ... traj_t60).
      - Quantile estimation at alpha = [0.05, 0.50, 0.95] providing 90% prediction intervals.
      - Regularization: reg_alpha (L1) and reg_lambda (L2) tuned specifically for small N=39 cohort.
      - Integrated SHAP (TreeExplainer) for post-hoc physiological attribution on pct_mca_drop.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize XGBoostTwinModel with hyperparameters.
        
        Args:
            config: Optional dictionary containing 'xgb_params' or top-level params.
        """
        self.config = config or {}
        
        # Default conservative hyperparameters tuned for N=39 subjects
        self.xgb_params: Dict[str, Any] = {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.05,
            "reg_alpha": 1.5,      # L1 regularization
            "reg_lambda": 5.0,     # L2 regularization
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": 1,
        }

        # Override defaults if provided in config
        if "xgb_params" in self.config and isinstance(self.config["xgb_params"], dict):
            self.xgb_params.update(self.config["xgb_params"])
        elif any(k in self.config for k in self.xgb_params):
            for k in self.xgb_params:
                if k in self.config:
                    self.xgb_params[k] = self.config[k]

        # Quantile models dictionary: models_[quantile_name][target_name] = xgb_model
        # quantiles: 'q05', 'q50', 'q95'
        self.models_: Dict[str, Dict[str, xgb.XGBRegressor]] = {
            "q05": {},
            "q50": {},
            "q95": {},
        }
        # Fallback residual standard errors if reg:quantileerror is unsupported
        self.residual_sigma_: Dict[str, float] = {}
        self.feature_names_: List[str] = []
        self.target_names_: List[str] = []
        self.trajectory_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "XGBoostTwinModel":
        """
        Fit quantile XGBoost models across all targets.
        
        Args:
            X: Input feature DataFrame (subject_id as index, peripheral features).
            y: Target DataFrame (MCA blood-flow velocity metrics and trajectory knots).
            
        Returns:
            self
        """
        self.feature_names_ = list(X.columns)
        if isinstance(y, pd.DataFrame):
            self.target_names_ = list(y.columns)
        else:
            self.target_names_ = ["target"]
            y = pd.DataFrame(y, columns=self.target_names_, index=X.index)

        self.trajectory_cols_ = [c for c in self.target_names_ if c.startswith("traj_t")]
        X_mat = X.values

        for col in self.target_names_:
            y_col = y[col].values

            # Fit Median (0.50) model first
            med_params = self.xgb_params.copy()
            med_params["objective"] = "reg:squarederror"
            med_model = xgb.XGBRegressor(**med_params)
            med_model.fit(X_mat, y_col)
            self.models_["q50"][col] = med_model

            # Calculate residual standard error on training set for fallback
            preds_med = med_model.predict(X_mat)
            residuals = y_col - preds_med
            sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0
            self.residual_sigma_[col] = max(sigma, 1e-4)

            # Fit 0.05 and 0.95 quantiles
            for q_key, q_alpha in [("q05", 0.05), ("q95", 0.95)]:
                q_params = self.xgb_params.copy()
                try:
                    q_params["objective"] = "reg:quantileerror"
                    q_params["quantile_alpha"] = q_alpha
                    q_model = xgb.XGBRegressor(**q_params)
                    q_model.fit(X_mat, y_col)
                    self.models_[q_key][col] = q_model
                except Exception:
                    # Fallback to squared error with residual offset
                    q_params["objective"] = "reg:squarederror"
                    fallback_model = xgb.XGBRegressor(**q_params)
                    fallback_model.fit(X_mat, y_col)
                    self.models_[q_key][col] = fallback_model

        return self

    def _predict_quantile(self, X: pd.DataFrame, q_key: str) -> pd.DataFrame:
        """Internal helper to predict for a specific quantile level."""
        if not self.target_names_:
            raise ValueError("XGBoostTwinModel must be fitted before predict().")

        X_aligned = X[self.feature_names_] if all(c in X.columns for c in self.feature_names_) else X
        X_mat = X_aligned.values
        n_samples = len(X)
        pred_dict: Dict[str, np.ndarray] = {}

        for col in self.target_names_:
            model = self.models_[q_key][col]
            base_pred = model.predict(X_mat)

            # If model was fitted with squared error fallback for 0.05 or 0.95, apply normal CI offset
            is_quantile_objective = getattr(model, "objective", None) == "reg:quantileerror"
            if not is_quantile_objective and q_key != "q50":
                sigma = self.residual_sigma_.get(col, 1.0)
                z = -1.645 if q_key == "q05" else 1.645
                pred_dict[col] = base_pred + (z * sigma)
            else:
                pred_dict[col] = base_pred

        return pd.DataFrame(pred_dict, index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate median (50th percentile) predictions for all targets.
        
        Args:
            X: Input feature DataFrame.
            
        Returns:
            pd.DataFrame of predicted values matching X.index and self.target_names_.
        """
        return self._predict_quantile(X, "q50")

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate quantile predictions with uncertainty bounds (5th, 50th, 95th percentiles).
        Enforces monotonic ordering: lower_5pct <= median_50pct <= upper_95pct.
        
        Args:
            X: Input feature DataFrame.
            
        Returns:
            Tuple of (lower_5pct_df, median_50pct_df, upper_95pct_df).
        """
        df_50 = self._predict_quantile(X, "q50")
        df_05 = self._predict_quantile(X, "q05")
        df_95 = self._predict_quantile(X, "q95")

        # Enforce consistency: q05 <= q50 <= q95
        lower_vals = np.minimum(df_05.values, df_50.values)
        upper_vals = np.maximum(df_95.values, df_50.values)

        df_lower = pd.DataFrame(lower_vals, index=X.index, columns=self.target_names_)
        df_upper = pd.DataFrame(upper_vals, index=X.index, columns=self.target_names_)

        return df_lower, df_50, df_upper

    def predict_trajectory(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict continuous MCA blood-flow velocity recovery trajectory across 13 knot points
        (t=0s to t=60s at 5s intervals) with uncertainty bounds.
        
        Args:
            X: Input feature DataFrame.
            
        Returns:
            Tuple of (lower_array, median_array, upper_array), each of shape (n_samples, 13).
        """
        df_lower, df_median, df_upper = self.predict_with_uncertainty(X)
        
        knots = [f"traj_t{t}" for t in range(0, 65, 5)]
        present_knots = [k for k in knots if k in self.target_names_]

        if len(present_knots) == 13:
            lower_arr = df_lower[present_knots].values
            median_arr = df_median[present_knots].values
            upper_arr = df_upper[present_knots].values
            return lower_arr, median_arr, upper_arr

        # Synthesize fallback trajectory from pct_mca_drop and recovery_time if knots weren't in targets
        n_samples = len(X)
        t_steps = np.linspace(0, 60, 13)
        med_traj = np.ones((n_samples, 13)) * 100.0
        low_traj = np.ones((n_samples, 13)) * 95.0
        up_traj = np.ones((n_samples, 13)) * 105.0

        if "pct_mca_drop" in df_median.columns and "recovery_time_s" in df_median.columns:
            drops = df_median["pct_mca_drop"].values
            recs = np.maximum(df_median["recovery_time_s"].values, 5.0)
            
            for i in range(n_samples):
                drop_val = drops[i]
                rec_val = recs[i]
                # Triangular/exponential profile
                nadir_val = max(10.0, 100.0 - drop_val)
                # t=0 -> 100%, t=5s -> nadir, t=rec -> 95%, t=60 -> 100%
                med_traj[i] = np.interp(t_steps, [0.0, 5.0, rec_val, 60.0], [100.0, nadir_val, 95.0, 100.0])
                low_traj[i] = med_traj[i] - 5.0
                up_traj[i] = med_traj[i] + 5.0

        return low_traj, med_traj, up_traj

    def get_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Compute SHAP values using TreeExplainer for the primary target: 'pct_mca_drop'.
        Provides patient-level physiological explainability.
        
        Args:
            X: Input feature DataFrame.
            
        Returns:
            Tuple of (shap_values_matrix, expected_value).
            shap_values_matrix has shape (n_samples, n_features).
        """
        target_name = "pct_mca_drop"
        if target_name not in self.models_["q50"]:
            # Fallback to first available target
            target_name = self.target_names_[0]

        model = self.models_["q50"][target_name]
        X_aligned = X[self.feature_names_] if all(c in X.columns for c in self.feature_names_) else X

        if not HAS_SHAP:
            # Fallback if shap is not available: linear surrogate approximation
            n_samples = len(X)
            n_features = len(self.feature_names_)
            importances = model.feature_importances_
            surrogate = np.tile(importances, (n_samples, 1)) * (X_aligned.values - np.mean(X_aligned.values, axis=0))
            return surrogate, float(np.mean(model.predict(X_aligned.values)))

        explainer = shap.TreeExplainer(model)
        shap_res = explainer(X_aligned)

        if hasattr(shap_res, "values"):
            shap_values = shap_res.values
            expected_val = float(shap_res.base_values[0]) if hasattr(shap_res.base_values, "__getitem__") else float(shap_res.base_values)
        else:
            shap_values = explainer.shap_values(X_aligned)
            expected_val = explainer.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = float(expected_val[0])
            else:
                expected_val = float(expected_val)

        return np.asarray(shap_values), expected_val

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract Gini feature importance across all primary median models.
        
        Returns:
            pd.DataFrame indexed by feature name, with columns for each target
            and a sorted 'mean_importance' column.
        """
        if not self.models_["q50"]:
            raise ValueError("Model must be fitted before extracting feature importances.")

        primary_targets = [t for t in self.target_names_ if not t.startswith("traj_t")]
        if not primary_targets:
            primary_targets = self.target_names_

        data: Dict[str, np.ndarray] = {}
        for target in primary_targets:
            data[target] = self.models_["q50"][target].feature_importances_

        df_importance = pd.DataFrame(data, index=self.feature_names_)
        df_importance["mean_importance"] = df_importance.mean(axis=1)
        return df_importance.sort_values(by="mean_importance", ascending=False)

    def get_name(self) -> str:
        """Return human-readable model name."""
        return "XGBoost Digital Twin (Model 2)"
