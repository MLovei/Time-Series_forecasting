import pandas as pd
from typing import List, Dict
import os

from config.config import *


class DataLoader:
    """Class for loading and preprocessing data"""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir

    def load_market_data(self) -> pd.DataFrame:
        """Load market data with proper datetime index"""
        filepath = os.path.join(self.data_dir, MARKET_DATA_FILE)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df

    def load_weather_data(self, location: str = "Vilnius") -> pd.DataFrame:
        """Load weather data for a specific location"""
        filename = f"weather_location_{location}.csv"
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df

    def load_all_weather_data(self) -> Dict[str, pd.DataFrame]:
        """Load weather data for all locations"""
        weather_data = {}
        for location_file in WEATHER_LOCATIONS:
            location = location_file.replace("weather_location_", "").replace(
                ".csv", ""
            )
            weather_data[location] = self.load_weather_data(location)
        return weather_data

    def get_day_ahead_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Extract day-ahead weather features (previous_day1 suffix)"""
        day_ahead_cols = [
            col for col in weather_df.columns if "previous_day1" in col
        ]
        return weather_df[day_ahead_cols]

    def get_intraday_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Extract intraday weather features (no suffix)"""
        intraday_cols = [
            col for col in weather_df.columns if "previous_day1" not in col
        ]
        return weather_df[intraday_cols]

    def create_lagged_features(
        self, df: pd.DataFrame, lag_minutes: int = 30
    ) -> pd.DataFrame:
        """Create lagged features for market data"""
        # Convert lag from minutes to number of periods (assuming 15-minute intervals)
        lag_periods = lag_minutes // 15
        lagged_df = df.shift(lag_periods)
        return lagged_df

    def align_data(self, *dfs: pd.DataFrame) -> List[pd.DataFrame]:
        """Align multiple dataframes by common datetime index"""
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)

        aligned = []
        for df in dfs:
            aligned.append(df.loc[common_index])

        return aligned
