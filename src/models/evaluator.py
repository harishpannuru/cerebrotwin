"""
CerebroTwin - Rigorous Subject-Level Model Evaluation Framework

CRITICAL DOMAIN RULES:
1. Subject-level splitting ONLY (LeaveOneGroupOut or GroupKFold grouped strictly by subject_id).
   No subject may ever appear in both training and test sets simultaneously.
2. N=39 subjects: 20 T2D, 19 Controls.
3. Abnormality flag: "Research Response Status: HIGH DEVIATION" — NEVER clinical diagnosis.
"""

import copy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


class ModelEvaluator:
    """
    Evaluation suite enforcing strict subject-level cross-validation
    and comprehensive multi-target hemodynamic metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize ModelEvaluator.
        
        Args:
            config: Optional evaluation configuration dictionary.
        """
        self.config = config or {}

    def compute_metrics(
        self, y_true: pd.DataFrame, y_pred: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute standard regression metrics per target and across the continuous trajectory.
        
        Metrics:
          - MAE (Mean Absolute Error)
          - RMSE (Root Mean Squared Error)
          - R2 (Coefficient of Determination)
          - MAPE (Mean Absolute Percentage Error, %)
          - Overall Trajectory RMSE (across all 13 knot points)
          
        Args:
            y_true: Ground truth target DataFrame.
            y_pred: Predicted target DataFrame.
            
        Returns:
            Nested dictionary: {target_name: {metric_name: value}}
        """
        metrics: Dict[str, Dict[str, float]] = {}
        common_cols = [c for c in y_true.columns if c in y_pred.columns]

        for col in common_cols:
            yt = y_true[col].values.astype(float)
            yp = y_pred[col].values.astype(float)

            # Avoid NaN or Inf
            valid_mask = ~(np.isnan(yt) | np.isnan(yp) | np.isinf(yt) | np.isinf(yp))
            if np.sum(valid_mask) < 2:
                metrics[col] = {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "mape": 0.0}
                continue

            yt_v = yt[valid_mask]
            yp_v = yp[valid_mask]

            mae = float(mean_absolute_error(yt_v, yp_v))
            rmse = float(np.sqrt(mean_squared_error(yt_v, yp_v)))

            # Handle zero variance in r2_score
            if np.var(yt_v) < 1e-8:
                r2 = 0.0
            else:
                r2 = float(r2_score(yt_v, yp_v))

            # MAPE with safe denominator
            denom = np.where(np.abs(yt_v) < 1e-3, 1e-3, np.abs(yt_v))
            mape = float(np.mean(np.abs((yt_v - yp_v) / denom)) * 100.0)

            metrics[col] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
            }

        # Overall trajectory metric across all 13 knots (traj_t0 to traj_t60)
        traj_cols = [c for c in common_cols if c.startswith("traj_t")]
        if len(traj_cols) > 0:
            yt_traj = y_true[traj_cols].values.flatten()
            yp_traj = y_pred[traj_cols].values.flatten()
            v_mask = ~(np.isnan(yt_traj) | np.isnan(yp_traj))
            if np.sum(v_mask) > 1:
                t_mae = float(mean_absolute_error(yt_traj[v_mask], yp_traj[v_mask]))
                t_rmse = float(np.sqrt(mean_squared_error(yt_traj[v_mask], yp_traj[v_mask])) )
                t_var = np.var(yt_traj[v_mask])
                t_r2 = float(r2_score(yt_traj[v_mask], yp_traj[v_mask])) if t_var > 1e-8 else 0.0
                metrics["overall_trajectory"] = {
                    "mae": t_mae,
                    "rmse": t_rmse,
                    "r2": t_r2,
                    "mape": float(np.mean(np.abs((yt_traj[v_mask] - yp_traj[v_mask]) / np.maximum(np.abs(yt_traj[v_mask]), 1e-3))) * 100.0),
                }

        return metrics

    def evaluate_loso(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        groups: Union[pd.Series, np.ndarray, List[Any]],
    ) -> Dict[str, Any]:
        """
        Leave-One-Subject-Out (LOSO) Cross-Validation.
        Strictly guarantees complete subject-level isolation.
        
        Args:
            model: Model instance (must implement fit and predict).
            X: Feature matrix with subject_id as index.
            y: Target matrix.
            groups: Subject identifier series/array for group grouping.
            
        Returns:
            Dictionary containing aggregated metrics, per-fold metrics,
            out-of-fold predictions, and CV metadata.
        """
        groups_arr = np.asarray(groups)
        logo = LeaveOneGroupOut()
        fold_predictions: List[pd.DataFrame] = []
        fold_metrics: List[Dict[str, Any]] = []

        model_name = getattr(model, "get_name", lambda: model.__class__.__name__)()

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups_arr)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            held_out_subject = str(groups_arr[test_idx[0]])

            # Deep-copy model to guarantee isolation
            fold_model = copy.deepcopy(model)
            fold_model.fit(X_train, y_train)
            pred_test = fold_model.predict(X_test)

            if not isinstance(pred_test, pd.DataFrame):
                pred_test = pd.DataFrame(pred_test, index=X_test.index, columns=y.columns)

            fold_predictions.append(pred_test)
            single_fold_metric = self.compute_metrics(y_test, pred_test)
            fold_metrics.append({
                "fold": fold_idx,
                "held_out_subject": held_out_subject,
                "metrics": single_fold_metric,
            })

        all_preds = pd.concat(fold_predictions, axis=0).loc[X.index]
        aggregate_metrics = self.compute_metrics(y, all_preds)

        return {
            "model_name": model_name,
            "cv_method": "LeaveOneSubjectOut",
            "n_folds": len(fold_metrics),
            "metrics": aggregate_metrics,
            "fold_metrics": fold_metrics,
            "predictions": all_preds,
            "y_true": y,
        }

    def evaluate_group_kfold(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        groups: Union[pd.Series, np.ndarray, List[Any]],
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        Group K-Fold Cross-Validation.
        Ensures all records belonging to a subject are confined to either train or test fold.
        
        Args:
            model: Model instance (must implement fit and predict).
            X: Feature matrix.
            y: Target matrix.
            groups: Subject identifier array.
            n_splits: Number of CV folds (default: 5).
            
        Returns:
            Dictionary containing aggregated and fold metrics.
        """
        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        actual_splits = min(n_splits, len(unique_groups))
        
        gkf = GroupKFold(n_splits=actual_splits)
        fold_predictions: List[pd.DataFrame] = []
        fold_metrics: List[Dict[str, Any]] = []

        model_name = getattr(model, "get_name", lambda: model.__class__.__name__)()

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups_arr)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fold_model = copy.deepcopy(model)
            fold_model.fit(X_train, y_train)
            pred_test = fold_model.predict(X_test)

            if not isinstance(pred_test, pd.DataFrame):
                pred_test = pd.DataFrame(pred_test, index=X_test.index, columns=y.columns)

            fold_predictions.append(pred_test)
            single_fold_metric = self.compute_metrics(y_test, pred_test)
            fold_metrics.append({
                "fold": fold_idx,
                "n_test_subjects": len(np.unique(groups_arr[test_idx])),
                "metrics": single_fold_metric,
            })

        all_preds = pd.concat(fold_predictions, axis=0).loc[X.index]
        aggregate_metrics = self.compute_metrics(y, all_preds)

        return {
            "model_name": model_name,
            "cv_method": f"GroupKFold_{actual_splits}",
            "n_folds": actual_splits,
            "metrics": aggregate_metrics,
            "fold_metrics": fold_metrics,
            "predictions": all_preds,
            "y_true": y,
        }

    def generate_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Format cross-validation results into a clean, human-readable summary table.
        
        Args:
            results: Results dictionary returned by evaluate_loso or evaluate_group_kfold.
            
        Returns:
            pd.DataFrame with targets as rows and (MAE, RMSE, R2, MAPE) as columns.
        """
        metrics_dict = results.get("metrics", {})
        rows: List[Dict[str, Any]] = []

        # Standard display order
        preferred_targets = [
            "pct_mca_drop",
            "mca_nadir_cms",
            "time_to_nadir_s",
            "cumulative_deficit_auc",
            "recovery_time_s",
            "recovery_status",
            "overall_trajectory",
        ]

        # Gather targets
        order = [t for t in preferred_targets if t in metrics_dict]
        for t in metrics_dict:
            if t not in order and not t.startswith("traj_t"):
                order.append(t)

        for target in order:
            m = metrics_dict[target]
            rows.append({
                "Target Variable": target,
                "MAE": round(m.get("mae", 0.0), 3),
                "RMSE": round(m.get("rmse", 0.0), 3),
                "R²": round(m.get("r2", 0.0), 3),
                "MAPE (%)": round(m.get("mape", 0.0), 2),
            })

        df_report = pd.DataFrame(rows)
        return df_report

    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Produce a side-by-side benchmark comparison of multiple models
        (e.g., Mean Baseline, Ridge Linear Regression, XGBoost Twin).
        
        Args:
            results_list: List of results dictionaries from multiple model evaluations.
            
        Returns:
            pd.DataFrame comparing key hemodynamic forecasting targets across models.
        """
        rows: List[Dict[str, Any]] = []

        key_metrics = [
            ("pct_mca_drop", "MAE", "pct_mca_drop (MAE %)"),
            ("pct_mca_drop", "RMSE", "pct_mca_drop (RMSE %)"),
            ("pct_mca_drop", "r2", "pct_mca_drop (R²)"),
            ("mca_nadir_cms", "MAE", "mca_nadir_cms (MAE cm/s)"),
            ("mca_nadir_cms", "RMSE", "mca_nadir_cms (RMSE cm/s)"),
            ("recovery_time_s", "MAE", "recovery_time_s (MAE s)"),
            ("cumulative_deficit_auc", "MAE", "cumulative_deficit (MAE)"),
            ("overall_trajectory", "RMSE", "overall_trajectory (RMSE %)"),
        ]

        for res in results_list:
            m_name = res.get("model_name", "Unknown Model")
            cv = res.get("cv_method", "CV")
            metrics = res.get("metrics", {})
            row = {"Model": m_name, "Validation": cv}

            for target_key, metric_key, col_header in key_metrics:
                val = metrics.get(target_key, {}).get(metric_key.lower(), np.nan)
                row[col_header] = round(val, 3) if not np.isnan(val) else "N/A"

            rows.append(row)

        return pd.DataFrame(rows)
