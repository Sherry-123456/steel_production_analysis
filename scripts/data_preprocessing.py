import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




def check_data_consistency(df):
    """
    Perform basic data consistency checks:
    - Clean column names
    - Ensure numeric columns are correctly typed
    - Handle invalid numeric values
    """
    df = df.copy()

    # 1. clean column names
    df.columns = df.columns.str.strip()

    # 2. try converting object columns to numeric if possible
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                converted = pd.to_numeric(df[col])
                df[col] = converted
                print(f"Column '{col}' converted to numeric type.")
            except ValueError:
                # keep as categorical
                pass

    # 3. remove obviously invalid numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        invalid_count = (df[col] < 0).sum()
        if invalid_count > 0:
            df.loc[df[col] < 0, col] = np.nan
            print(f"Column '{col}': {invalid_count} invalid negative values set to NaN.")

    return df


def remove_duplicates(df):
    """Remove duplicate rows from dataset."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows.")
    return df


def fill_missing_values(df):
    """Fill missing values in numeric columns using median."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            count = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {count} missing values in '{col}' with median {median_val}")
    return df


def handle_outliers_iqr(df):
    """Replace outliers using IQR bounds."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        mask = (df[col] < lower) | (df[col] > upper)
        outliers = mask.sum()

        if outliers > 0:
            replacement = df[col].median()
            df.loc[mask, col] = replacement
            print(f"Column '{col}': replaced {outliers} outliers with {replacement}")

    return df


def encode_categorical(df):
    """One-hot encode categorical features."""
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        print(f"Encoding {len(cat_cols)} categorical columns.")
        df = pd.get_dummies(df, drop_first=True)
    return df




def split_dataset(df, target="output", test_size=0.2, val_size=0.2, random_state=42):
    """Split dataset into train / validation / test sets."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state
    )

    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Standardize features using training data statistics."""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled




def preprocess_data(df):
    df = remove_duplicates(df)
    df = check_data_consistency(df)
    df = fill_missing_values(df)
    df = handle_outliers_iqr(df)
    df = encode_categorical(df)
    return df



if __name__ == "__main__":
    df = pd.read_csv("../data/normalized_train_data.csv")

    print("Starting data preprocessing...")
    df_cleaned = preprocess_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df_cleaned)

    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

    print("Preprocessing completed successfully.")


