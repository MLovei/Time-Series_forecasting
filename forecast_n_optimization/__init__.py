"""Enerheads Quantitative Challenge - Main Package"""

__version__ = "1.0.0"

# Expose main classes at package level for simple imports
from .forecasting.prediction_models import (
    EnergyPriceForecaster,
    display_summary_md,
)
from .forecasting.visuals import (
    feature_importance,
    plot_seaborn_comparison,
    plot_seaborn_scatter,
)
from .utils.data_loader import DataLoader
from .utils.evaluation import EvaluationMetrics

# Also expose submodules for more specific imports if needed
from . import forecasting
from . import utils

__all__ = [
    "DataLoader",
    "EvaluationMetrics",
    "EnergyPriceForecaster",
    "display_summary_md",
    "feature_importance",
    "plot_seaborn_comparison",
    "plot_seaborn_scatter",
    "forecasting",
    "utils",
]
