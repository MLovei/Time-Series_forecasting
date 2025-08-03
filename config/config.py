# Configuration file for Enerheads Quantitative Challenge

# Data paths
DATA_DIR = "../data"
RESULTS_DIR = "../results"
MODELS_DIR = "../models"

# Data files
MARKET_DATA_FILE = "market_data.csv"
WEATHER_LOCATIONS = [
    "weather_location_Vilnius.csv",
    "weather_location_Kaunas.csv",
    "weather_location_Klaipeda.csv",
    "weather_location_Marijampole.csv",
    "weather_location_Panevezys.csv",
    "weather_location_Siauliai.csv",
    "weather_location_Taurage.csv",
    "weather_location_Telsiai.csv",
    "weather_location_Utena.csv",
    "weather_location_Alytus.csv",
]

# Target variables
NORD_POOL_TARGET = "10YLT-1001A0008Q_DA_eurmwh"
MFRR_UP_TARGET = "LT_up_sa_cbmp"
MFRR_DOWN_TARGET = "LT_down_sa_cbmp"

# Model parameters
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
N_ESTIMATORS = 100

# Evaluation parameters
SPREAD_THRESHOLD_EUR = 200
