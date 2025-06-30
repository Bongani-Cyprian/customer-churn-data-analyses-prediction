# feature_engineering.py

############################################################################################

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Tenure Bucket (12-month windows)
        bins = [0, 12, 24, 48, np.inf]
        labels = ["0-12", "13-24", "25-48", "49+"]
        X["tenure_bucket"] = pd.cut(X["tenure"], bins=bins, labels=labels, right=True)
        
        # 2. Service Count (count number of "Yes" among core services)
        service_cols = [
            "phone_service", "multiple_lines", "internet_service",
            "online_security", "online_backup", "device_protection",
            "tech_support", "streaming_tv", "streaming_movies"
        ]
        X["service_count"] = X[service_cols].apply(lambda row: (row == "Yes").sum(), axis=1)
        
        # 3. Average Charge per Activated Service
        X["avg_charge_per_service"] = (
            X["monthly_charges"] / X["service_count"].replace(0, np.nan)
        ).fillna(0)
        
        # 4. Long-Term Contract Flag
        X["is_long_contract"] = X["contract"].isin(["One year", "Two year"]).astype(int)
        
        # 5. Paperless Billing Flag
        X["is_paperless"] = (X["paperless_billing"] == "Yes").astype(int)
        
        # Convert remaining object columns to categorical.
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = X[col].astype("category")
        
        return X
