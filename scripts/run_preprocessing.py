import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import EdgeIIoTLoader
from src.data.preprocessor import LeakageFreePreprocessor
from src.utils.seed import set_seed

if __name__ == "__main__":
    set_seed(42)
    print("Executing Phase 4: Leakage-Safe Preprocessing Pipeline...")

    # 1. Load data
    loader = EdgeIIoTLoader(seed=42)
    df = loader.load_dataset(num_samples_fallback=10000)

    # 2. Instantiate Leakage-Free Preprocessor
    preprocessor = LeakageFreePreprocessor(seed=42, withheld_classes=["Ransomware", "Backdoor"])
    
    # 3. Clean raw data
    cleaned_df = preprocessor.validate_and_clean_raw(df)

    # 4. Split data (Zero-day withholding first, then Stratified split)
    train_df, val_df, test_df, zero_day_df = preprocessor.split_data(cleaned_df)

    # 5. Fit preprocessor STRICTLY on training split
    preprocessor.fit_preprocessor(train_df)

    # 6. Transform all data splits
    X_train, y_train = preprocessor.transform_data(train_df)
    X_val, y_val = preprocessor.transform_data(val_df)
    X_test, y_test = preprocessor.transform_data(test_df)
    X_zero_day, y_zero_day = preprocessor.transform_data(zero_day_df)

    # 7. Save processed data
    preprocessor.save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, X_zero_day, y_zero_day, output_dir="data/processed")

    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")
    print(f"X_zero_day shape: {X_zero_day.shape} | y_zero_day shape: {y_zero_day.shape}")
    print("Preprocessing completed cleanly without leakage!")
