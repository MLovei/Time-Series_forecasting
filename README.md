# 📊 ENERHEADS - Energy Market Price Forecasting

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/Poetry-1.0+-orange.svg)](https://python-poetry.org/)
[![License](https://img.shields.io/badge/License-CC-green.svg)](LICENSE)

A comprehensive solution for energy market price forecasting, developed for the Enerheads Quantitative Challenge. This project focuses on predicting energy market prices using meteorological and market data to support energy trading decisions in the Baltic region.

## 🌟 Features

### 📊 Energy Price Forecasting
- **Day-ahead Nord Pool price prediction** using meteorological and market data
- **Intraday mFRR price forecasting** for balancing market activations
- Advanced feature engineering with temporal and rolling features
- XGBoost and Random Forest models with time-series cross-validation
- Comprehensive evaluation metrics for price prediction accuracy

### 📈 Analytics & Visualization
- Interactive Jupyter notebooks for analysis
- Feature importance analysis and model performance visualization
- Comprehensive evaluation metrics including:
  - Peak/valley price prediction accuracy
  - Price spread analysis (>200 EUR/MWh thresholds)
  - Statistical confidence intervals
  - Model performance visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Poetry (for dependency management)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MLovei/Time-Series_forecasting.git
   ```

2. **Install dependencies using Poetry**
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**
   ```bash
   poetry shell
   ```

### Usage

#### Energy Price Forecasting

```python
from forecast_n_optimization import EnergyPriceForecaster, DataLoader

# Initialize the forecaster
forecaster = EnergyPriceForecaster()

# Prepare day-ahead data
X_da, y_da, summary_da = forecaster.prepare_day_ahead_data("Vilnius")

# Train day-ahead model
results_da = forecaster.train_day_ahead_model(X_da, y_da)

# Prepare intraday data
X_id, y_id, summary_id = forecaster.prepare_intraday_data("Vilnius")

# Train intraday model
results_id = forecaster.train_intraday_model(X_id, y_id)
```

#### Interactive Analysis

```bash
# Run the forecasting notebook
jupyter notebook notebooks/prc_forecasting.ipynb
```

## 📁 Project Structure

```
ENERHEADS/
├── forecast_n_optimization/
│   └── enerheads-quant-challenge/
│       ├── config/
│       │   ├── __init__.py
│       │   └── config.py              # Configuration constants
│       ├── data/                      # Market and weather datasets
│       │   ├── market_data.csv        # Nord Pool and mFRR data
│       │   └── weather_location_*.csv # Meteorological data for 10 locations
│       ├── forecast_n_optimization/   # Main package
│       │   ├── forecasting/
│       │   │   ├── prediction_models.py # Energy price forecasting models
│       │   │   └── visuals.py          # Visualization utilities
│       │   └── utils/
│       │       ├── data_loader.py      # Data loading and preprocessing
│       │       └── evaluation.py       # Evaluation metrics
│       ├── notebooks/
│       │   └── prc_forecasting.ipynb   # Price forecasting analysis
│       ├── pyproject.toml             # Poetry configuration
│       └── README.md                  # This file
```

## 📊 Data Sources

### Market Data
- **Nord Pool day-ahead prices**: `10YLT-1001A0008Q_DA_eurmwh`
- **mFRR activation prices**: `LT_up_sa_cbmp` / `LT_down_sa_cbmp`
- **Balancing market data**: mFRR activations and clearing prices
- Source: [Baltic Transparency Dashboard](https://baltic.transparency-dashboard.eu/)

### Weather Data
- **10 Lithuanian locations**: Alytus, Kaunas, Klaipeda, Marijampole, Panevezys, Siauliai, Taurage, Telsiai, Utena, Vilnius
- **Day-ahead forecasts**: 24-hour ahead predictions (suffix: `previous_day1`)
- **Intraday forecasts**: 1-hour ahead predictions
- Source: [OpenMeteo API](https://open-meteo.com/en/docs)

## 🔧 Key Components

### EnergyPriceForecaster
Main class for energy price forecasting with methods for:
- Data preparation with temporal and rolling features
- XGBoost model training with time-series cross-validation
- Feature importance analysis
- Model performance evaluation

### DataLoader
Utility class for:
- Loading market and weather data
- Feature extraction (day-ahead vs intraday)
- Data alignment and preprocessing
- Lagged feature creation

### EvaluationMetrics
Comprehensive evaluation including:
- Peak/valley price prediction accuracy
- Price spread analysis
- Statistical confidence intervals
- Model performance visualization

## 📈 Model Performance

The forecasting models achieve:
- **Day-ahead Nord Pool prices**: Accurate prediction of daily price patterns
- **Intraday mFRR prices**: Real-time balancing market price forecasts
- **Feature importance**: Weather variables and temporal patterns drive predictions
- **Evaluation metrics**: Comprehensive analysis of prediction accuracy and economic impact

## 🛠️ Development

### Adding New Features
1. Follow the existing package structure
2. Add tests for new functionality
3. Update documentation and notebooks
4. Ensure compatibility with existing models

### Code Style
This project follows PEP 8 guidelines and uses type hints throughout.

## 📚 Dependencies

### Core Dependencies
- **pandas** (^2.3.1): Data manipulation and analysis
- **numpy** (^2.3.2): Numerical computing
- **scikit-learn** (^1.7.1): Machine learning algorithms
- **xgboost** (^3.0.3): Gradient boosting for forecasting

### Visualization & Analysis
- **matplotlib** (^3.10.5): Plotting and visualization
- **seaborn** (^0.13.2): Statistical data visualization
- **jupyter** (^1.1.1): Interactive notebooks

### Utilities
- **tqdm** (^4.67.1): Progress bars
- **tabulate** (^0.9.0): Pretty-printed tables
- **protobuf** (^6.31.1): Protocol buffers
- **setuptools** (^80.9.0): Package development tools

## 📄 License

This project is licensed under Creative Commons - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Martynas Loveikis** - [martynas.love@gmail.com](mailto:martynas.love@gmail.com)

For questions, suggestions, or collaboration opportunities:
- Email: [martynas.love@gmail.com](mailto:martynas.love@gmail.com)
- GitHub Issues: [Create an issue](https://github.com/yourusername/ENERHEADS/issues)

---