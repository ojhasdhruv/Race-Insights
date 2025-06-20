# src/models/deg_model.py
"""
Light-weight tyre-degradation helpers.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pit/out laps and obvious slow (> 120 s) laps."""
    mask = (df["pit"] == 0) & (df["milliseconds"] < 120_000)
    return df.loc[mask].copy()


def fit_linear_deg(df: pd.DataFrame):
    """Fit linear model (tyre_age → lap time [s]). Return model, slope."""
    X = df["tyre_age"].values.reshape(-1, 1)
    y = df["milliseconds"].values / 1000
    model = LinearRegression().fit(X, y)
    slope = float(model.coef_[0])
    return model, slope


def undercut_lap(slope: float, pit_loss: float = 22) -> int:
    """
    Return the tyre age (in laps) at which the lap-time difference
    equals the pit-lane loss.
    """
    if slope <= 0:
        return np.inf  # tyres not degrading
    return int(np.ceil(pit_loss / slope))
