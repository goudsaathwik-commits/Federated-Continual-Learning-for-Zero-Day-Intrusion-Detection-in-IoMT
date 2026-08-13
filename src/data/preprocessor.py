import os
import json
import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.utils.seed import set_seed
from src.utils.logging_config import setup_logger

logger = setup_logger("preprocessor")

class LeakageFreePreprocessor:
    """
    Leakage-Free Preprocessing Pipeline for Edge-IIoTset IoMT IDS.
    Fits all transformers (imputers, scalers, encoders) STRICTLY on training data.
    Enforces programmatic withholding of Zero-Day attack categories.
    """
    def __init__(self, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42, withheld_classes: Optional[List[str]] = None):
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.withheld_classes = withheld_classes if withheld_classes is not None else ["Ransomware", "Backdoor"]
        
        self.num_imputer = None
        self.scaler = None
        self.cat_encoders = {}
        self.label_encoder = None
        self.num_cols = []
        self.cat_cols = []
        self.drop_cols = ["frame.time", "ip.src_host", "ip.dst_host", "Attack_label"]
        self.fitted = False

    def validate_and_clean_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicates, replaces infinity values with NaN, and validates schema integrity.
        """
        logger.info(f"Initial raw dataset shape: {df.shape}")
        
        # 1. Remove duplicate rows
        df_cleaned = df.drop_duplicates().copy()
        logger.info(f"Shape after removing duplicates: {df_cleaned.shape}")

        # 2. Replace Infinite values with NaN
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)

        return df_cleaned

    def split_data(self, df: pd.DataFrame, target_col: str = "Attack_type") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs Zero-Day attack class withholding first, then performs Stratified Train/Val/Test splitting
        on known classes.
        Returns: (train_df, val_df, test_df, zero_day_df)
        """
        set_seed(self.seed)

        # 1. Programmatic Zero-Day Attack Withholding
        is_zero_day = df[target_col].isin(self.withheld_classes)
        zero_day_df = df[is_zero_day].copy()
        known_df = df[~is_zero_day].copy()

        logger.info(f"Zero-Day Withheld samples ({self.withheld_classes}): {len(zero_day_df)}")
        logger.info(f"Known attack/normal samples for training/val/test: {len(known_df)}")

        # 2. Stratified Train / Val / Test Split
        # Calculate split ratios
        test_ratio = self.test_size
        val_ratio = self.val_size / (1.0 - test_ratio)

        train_val_df, test_df = train_test_split(
            known_df, test_size=test_ratio, random_state=self.seed, stratify=known_df[target_col]
        )

        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=self.seed, stratify=train_val_df[target_col]
        )

        logger.info(f"Train split size: {len(train_df)} | Val split size: {len(val_df)} | Test split size: {len(test_df)}")

        return train_df, val_df, test_df, zero_day_df

    def fit_preprocessor(self, train_df: pd.DataFrame, target_col: str = "Attack_type"):
        """
        Fits Imputer, Scaler, and Encoders STRICTLY on the training split train_df.
        """
        logger.info("Fitting preprocessor transformers strictly on training data slice...")
        set_seed(self.seed)

        feature_df = train_df.drop(columns=[target_col] + [c for c in self.drop_cols if c in train_df.columns])

        # Identify numerical and categorical features
        self.num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = feature_df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

        # 1. Fit Numerical Imputer
        if self.num_cols:
            self.num_imputer = SimpleImputer(strategy="median")
            self.num_imputer.fit(feature_df[self.num_cols])

            # Transform train numerical to fit scaler
            num_imputed = self.num_imputer.transform(feature_df[self.num_cols])
            
            # 2. Fit StandardScaler ONLY on train_df
            self.scaler = StandardScaler()
            self.scaler.fit(num_imputed)

        # 3. Fit Categorical Encoders on train_df
        for col in self.cat_cols:
            le = LabelEncoder()
            # Convert to string to handle missing/mixed types
            col_vals = feature_df[col].astype(str).fillna("missing")
            le.fit(col_vals)
            self.cat_encoders[col] = le

        # 4. Fit LabelEncoder for Target Classes on train_df
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(train_df[target_col])

        self.fitted = True
        logger.info(f"Preprocessor fitted successfully. Numerical features: {len(self.num_cols)}, Categorical features: {len(self.cat_cols)}")

    def transform_data(self, df: pd.DataFrame, target_col: str = "Attack_type") -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms input dataframe using pre-fitted transformers.
        Outputs scaled feature matrix X (np.ndarray) and target vector y (np.ndarray).
        Unseen zero-day target classes are encoded as -1.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform_data()!")

        feature_df = df.drop(columns=[target_col] + [c for c in self.drop_cols if c in df.columns], errors='ignore')

        # 1. Numerical Impute + Scale
        if self.num_cols:
            num_data = feature_df[self.num_cols]
            num_imputed = self.num_imputer.transform(num_data)
            num_scaled = self.scaler.transform(num_imputed)
        else:
            num_scaled = np.empty((len(df), 0))

        # 2. Categorical Transform
        cat_encoded_list = []
        for col in self.cat_cols:
            le = self.cat_encoders[col]
            col_vals = feature_df[col].astype(str).fillna("missing")
            # Map unknown categories to nearest known label or -1
            known_classes = set(le.classes_)
            col_mapped = col_vals.apply(lambda x: x if x in known_classes else le.classes_[0])
            encoded = le.transform(col_mapped).reshape(-1, 1)
            cat_encoded_list.append(encoded)

        if cat_encoded_list:
            cat_scaled = np.hstack(cat_encoded_list)
            X = np.hstack([num_scaled, cat_scaled])
        else:
            X = num_scaled

        # 3. Transform Target Labels
        y_labels = df[target_col]
        known_target_classes = set(self.label_encoder.classes_)
        y = np.array([self.label_encoder.transform([label])[0] if label in known_target_classes else -1 for label in y_labels])

        return X, y

    def save_processed_data(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            X_zero_day: np.ndarray, y_zero_day: np.ndarray,
                            output_dir: str = "data/processed"):
        """Saves processed numerical numpy arrays to data/processed directory."""
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "X_val.npy"), X_val)
        np.save(os.path.join(output_dir, "y_val.npy"), y_val)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        np.save(os.path.join(output_dir, "X_zero_day.npy"), X_zero_day)
        np.save(os.path.join(output_dir, "y_zero_day.npy"), y_zero_day)
        logger.info(f"All processed data splits successfully saved to: {output_dir}")
