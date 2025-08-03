import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EvaluationMetrics:
    """Custom evaluation metrics for energy price forecasting"""

    @staticmethod
    def peak_trough_accuracy(y_true, y_pred):
        """
        y_true: pandas Series (indexed by datetime)
        y_pred: numpy array or list (same length/ordering as y_true)
        """
        # Align y_pred to the index of y_true
        y_pred_series = pd.Series(y_pred, index=y_true.index)

        daily_groups = y_true.groupby(y_true.index.date)
        accuracies = []

        for date, group in daily_groups:
            if len(group) == 0:
                continue
            # True min/max times
            true_min_hour = group.idxmin().hour
            true_max_hour = group.idxmax().hour

            # Pred min/max times
            pred_group = y_pred_series.loc[group.index]
            pred_min_hour = pred_group.idxmin().hour
            pred_max_hour = pred_group.idxmax().hour

            min_acc = 1 if true_min_hour == pred_min_hour else 0
            max_acc = 1 if true_max_hour == pred_max_hour else 0

            accuracies.append((min_acc + max_acc) / 2)

        return np.mean(accuracies)

    @staticmethod
    def spread_analysis(
        y_true: pd.Series, threshold: float = 200
    ) -> Dict[str, float]:
        """
        Task 3.ii: Count instances where daily spreads exceed threshold
        """
        daily_spreads = y_true.groupby(y_true.index.date).agg(["min", "max"])
        daily_spreads["spread"] = daily_spreads["max"] - daily_spreads["min"]

        high_spread_count = (daily_spreads["spread"] > threshold).sum()
        total_days = len(daily_spreads)

        return {
            "high_spread_days": high_spread_count,
            "total_days": total_days,
            "percentage": (
                (high_spread_count / total_days) * 100
                if total_days > 0
                else 0.0
            ),
            "avg_spread": daily_spreads["spread"].mean(),
            "max_spread": daily_spreads["spread"].max(),
        }

    @staticmethod
    def comprehensive_metrics(
        y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""

        # Align forecasting with true values
        if len(y_pred) != len(y_true):
            min_len = min(len(y_true), len(y_pred))
            y_true_aligned = y_true.iloc[-min_len:]
            y_pred_aligned = y_pred[-min_len:]
        else:
            y_true_aligned = y_true
            y_pred_aligned = y_pred

        # Standard regression metrics
        mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
        mse = mean_squared_error(y_true_aligned, y_pred_aligned)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_aligned, y_pred_aligned)

        # Custom energy market metrics
        peak_trough_acc = EvaluationMetrics.peak_trough_accuracy(
            y_true_aligned, y_pred_aligned
        )
        spread_stats = EvaluationMetrics.spread_analysis(y_true_aligned)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Peak_Trough_Accuracy": peak_trough_acc,
            "High_Spread_Days": spread_stats["high_spread_days"],
            "High_Spread_Percentage": spread_stats["percentage"],
            "Average_Daily_Spread": spread_stats["avg_spread"],
            "Max_Daily_Spread": spread_stats["max_spread"],
        }
