import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from IPython.display import Markdown, display
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from config.config import *
from ..utils.data_loader import DataLoader
from ..utils.evaluation import EvaluationMetrics


class EnergyPriceForecaster:
    """Main class for energy price forecasting"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.day_ahead_model = None
        self.intraday_model = None
        self.scaler_day_ahead = StandardScaler()
        self.scaler_intraday = StandardScaler()

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month
        return df

    def _add_rolling_features(
        self, market: pd.DataFrame, window_hours: int = 3
    ) -> pd.DataFrame:
        target = market[NORD_POOL_TARGET]
        roll = target.rolling(window=window_hours, min_periods=1)
        return pd.DataFrame(
            {
                f"da_roll_mean_{window_hours}h": roll.mean(),
                f"da_roll_std_{window_hours}h": roll.std().fillna(0),
            }
        )

    def prepare_day_ahead_data(
        self, location: str = "Vilnius"
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        dl = self.data_loader
        mkt = dl.load_market_data()
        wth = dl.load_weather_data(location)

        Xw = dl.get_day_ahead_features(wth)
        y = mkt[NORD_POOL_TARGET].replace(-9999, np.nan)
        y = y.fillna(mkt[NORD_POOL_TARGET]).clip(-500, 4000)
        y_outlier = ((y < -100) | (y > 1000)).astype(int)

        Xw = self._add_temporal_features(Xw)
        rolling = self._add_rolling_features(mkt, 3).reindex(Xw.index).ffill()
        if "wind_speed_80m_previous_day1" in Xw.columns:
            Xw["wind_time_inter"] = (
                Xw["wind_speed_80m_previous_day1"] * Xw["hour"]
            )

        Xa = pd.concat([Xw, rolling], axis=1)
        Xa["price_extreme"] = y_outlier.reindex(Xa.index).fillna(0)
        ya = y.reindex(Xa.index)
        Xa, ya = dl.align_data(Xa, ya)

        df = pd.concat([Xa, ya.rename("target")], axis=1)
        df = df[df["target"].notna()].ffill().dropna()

        X_final = df.drop(columns="target")
        y_final = df["target"]

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Raw market rows",
                    "Raw weather rows",
                    "Input features",
                    "Final rows",
                    "Date start",
                    "Date end",
                    "Target min",
                    "Target max",
                    "Target mean",
                ],
                "Value": [
                    mkt.shape[0],
                    wth.shape[0],
                    X_final.shape[1],
                    len(df),
                    df.index.min(),
                    df.index.max(),
                    y_final.min(),
                    y_final.max(),
                    y_final.mean(),
                ],
            }
        )
        return X_final, y_final, summary

    def prepare_intraday_data(
        self, location: str = "Vilnius"
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        dl = self.data_loader
        wth = dl.load_weather_data(location)
        mkt = dl.load_market_data()
        Xw = wth.copy()
        Xn = mkt[[NORD_POOL_TARGET]]

        y = (
            mkt[MFRR_UP_TARGET]
            .replace(-9999, np.nan)
            .fillna(mkt[MFRR_DOWN_TARGET])
            .clip(-500, 4000)
        )
        y_outlier = ((y < -100) | (y > 1000)).astype(int)

        Xw = self._add_temporal_features(Xw)
        rolling = self._add_rolling_features(mkt, 3).reindex(Xw.index).ffill()
        if "temperature_2m" in Xw.columns:
            Xw["temp_time_inter"] = Xw["temperature_2m"] * Xw["hour"]

        mask = mkt.isnull().mean() < 0.5
        good_cols = [
            c
            for c in mkt.columns[mask]
            if c not in [MFRR_UP_TARGET, MFRR_DOWN_TARGET]
        ]
        Xl = dl.create_lagged_features(mkt[good_cols], lag_minutes=30)

        X_comb = pd.concat([Xw, Xn, rolling, Xl], axis=1)
        X_comb = X_comb.loc[:, ~X_comb.columns.duplicated()]
        X_comb["price_extreme"] = y_outlier.reindex(X_comb.index).fillna(0)
        Xa, ya = dl.align_data(X_comb, y)

        df = pd.concat([Xa, ya.rename("target")], axis=1)
        df = df[df["target"].notna()].ffill().dropna()

        X_final = df.drop(columns="target")
        y_final = df["target"]

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Raw weather rows",
                    "Raw market rows",
                    "Weather features",
                    "Lagged features",
                    "Final rows",
                    "Date start",
                    "Date end",
                    "Target min",
                    "Target max",
                    "Target mean",
                ],
                "Value": [
                    wth.shape[0],
                    mkt.shape[0],
                    Xw.shape[1],
                    Xl.shape[1],
                    len(df),
                    df.index.min(),
                    df.index.max(),
                    y_final.min(),
                    y_final.max(),
                    y_final.mean(),
                ],
            }
        )
        return X_final, y_final, summary

    def train_day_ahead_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        n = len(X)
        split_idx = int(n * 0.8)
        X_cv, y_cv = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        tscv = TimeSeriesSplit(n_splits=5)
        cv_results = []
        for train_idx, val_idx in tqdm(
            tscv.split(X_cv),
            total=tscv.get_n_splits(),
            desc="Day-Ahead CV folds",
        ):
            X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
            y_tr, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

            X_tr_s = self.scaler_day_ahead.fit_transform(X_tr)
            X_val_s = self.scaler_day_ahead.transform(X_val)

            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
            )
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_val_s)
            cv_results.append(
                EvaluationMetrics.comprehensive_metrics(y_val, y_pred)
            )

        # Aggregate CV metrics, skipping NaNs
        with np.errstate(invalid="ignore"):
            cv_mean = {
                k: np.nanmean([r[k] for r in cv_results]) for k in cv_results[0]
            }
            cv_std = {
                k: np.nanstd([r[k] for r in cv_results]) for k in cv_results[0]
            }

        # Retrain on full CV set
        X_full_s = self.scaler_day_ahead.fit_transform(X_cv)
        final_model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        final_model.fit(X_full_s, y_cv)

        X_test_s = self.scaler_day_ahead.transform(X_test)
        y_test_pred = final_model.predict(X_test_s)
        test_metrics = EvaluationMetrics.comprehensive_metrics(
            y_test, y_test_pred
        )

        summary_rows = [
            ("CV Mean MAE", cv_mean["MAE"]),
            ("CV Std MAE", cv_std["MAE"]),
            ("CV Mean RMSE", cv_mean["RMSE"]),
            ("CV Std RMSE", cv_std["RMSE"]),
        ]
        summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

        return {
            "model": final_model,
            "scaler": self.scaler_day_ahead,
            "cv_mean_metrics": cv_mean,
            "cv_std_metrics": cv_std,
            "test_metrics": test_metrics,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
            "summary": summary,
        }

    def train_intraday_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        n = len(X)
        split_idx = int(n * 0.8)
        X_cv, y_cv = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        tscv = TimeSeriesSplit(n_splits=5)
        cv_results = []
        for train_idx, val_idx in tqdm(
            tscv.split(X_cv),
            total=tscv.get_n_splits(),
            desc="Intraday CV folds",
        ):
            X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
            y_tr, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

            X_tr_s = self.scaler_intraday.fit_transform(X_tr)
            X_val_s = self.scaler_intraday.transform(X_val)

            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
            )
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_val_s)
            cv_results.append(
                EvaluationMetrics.comprehensive_metrics(y_val, y_pred)
            )

        with np.errstate(invalid="ignore"):
            cv_mean = {
                k: np.nanmean([r[k] for r in cv_results]) for k in cv_results[0]
            }
            cv_std = {
                k: np.nanstd([r[k] for r in cv_results]) for k in cv_results[0]
            }

        X_full_s = self.scaler_intraday.fit_transform(X_cv)
        final_model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        final_model.fit(X_full_s, y_cv)

        X_test_s = self.scaler_intraday.transform(X_test)
        y_test_pred = final_model.predict(X_test_s)
        test_metrics = EvaluationMetrics.comprehensive_metrics(
            y_test, y_test_pred
        )

        summary_rows = [
            ("CV Mean MAE", cv_mean["MAE"]),
            ("CV Std MAE", cv_std["MAE"]),
            ("CV Mean RMSE", cv_mean["RMSE"]),
            ("CV Std RMSE", cv_std["RMSE"]),
        ]
        summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

        return {
            "model": final_model,
            "scaler": self.scaler_intraday,
            "cv_mean_metrics": cv_mean,
            "cv_std_metrics": cv_std,
            "test_metrics": test_metrics,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
            "summary": summary,
        }

    def train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> Dict[str, Any]:
        """
        XGBoost with TimeSeriesSplit CV for boost‐round selection, then final fit,
        returning CV means/stds, test metrics, and a summary DataFrame.
        """
        # 1) Split off final 20% for testing
        n = len(X)
        split_idx = int(n * 0.8)
        X_cv, y_cv = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        # 2) CV folds for out‐of‐fold RMSE & boost‐round tuning
        tscv = TimeSeriesSplit(n_splits=n_splits)
        dtrain = xgb.DMatrix(X_cv, label=y_cv)

        params = {
            "objective": "reg:squarederror",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "seed": RANDOM_STATE,
            "eval_metric": "rmse",
        }

        # run CV to pick best number of rounds
        cv_df = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            folds=tscv,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
            as_pandas=True,
        )
        best_rounds = len(cv_df)

        # collect per‐fold metrics manually for summary
        cv_results = []
        for train_idx, val_idx in tscv.split(X_cv):
            X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
            y_tr, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

            m = xgb.XGBRegressor(**params, n_estimators=best_rounds)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = m.predict(X_val)

            cv_results.append(
                EvaluationMetrics.comprehensive_metrics(y_val, y_pred)
            )

        # 3) Aggregate CV metrics
        with np.errstate(invalid="ignore"):
            cv_mean = {
                k: np.nanmean([r[k] for r in cv_results]) for k in cv_results[0]
            }
            cv_std = {
                k: np.nanstd([r[k] for r in cv_results]) for k in cv_results[0]
            }

        # 4) Fit final model on full CV block
        final_model = xgb.XGBRegressor(**params, n_estimators=best_rounds)
        final_model.fit(X_cv, y_cv, verbose=False)

        # 5) Final test evaluation
        y_test_pred = final_model.predict(X_test)
        test_metrics = EvaluationMetrics.comprehensive_metrics(
            y_test, y_test_pred
        )

        # 6) Build summary DataFrame
        summary_rows = [
            ("CV Mean MAE", cv_mean["MAE"]),
            ("CV Std MAE", cv_std["MAE"]),
            ("CV Mean RMSE", cv_mean["RMSE"]),
            ("CV Std RMSE", cv_std["RMSE"]),
            ("Test MAE", test_metrics["MAE"]),
            ("Test RMSE", test_metrics["RMSE"]),
            ("Best Rounds", best_rounds),
        ]
        summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

        return {
            "model": final_model,
            "n_boost_rounds": best_rounds,
            "cv_mean_metrics": cv_mean,
            "cv_std_metrics": cv_std,
            "test_metrics": test_metrics,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
            "summary": summary,
        }


def display_summary_md(summary_df: pd.DataFrame, title: str):
    """Display a pandas summary DataFrame as a Markdown table."""
    md = f"### {title}\n\n" + summary_df.to_markdown(index=False)
    display(Markdown(md))
