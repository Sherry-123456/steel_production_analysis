import pandas as pd
from pathlib import Path


def load_data(filename):
    """
    Load steel production data from the project's data directory.

    Parameters
    ----------
    filename : str
        Name of the CSV file stored in the data folder.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing steel production data.
    """
    # locate project root directory
    project_root = Path(__file__).absolute().parents[1]
    data_path = project_root / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    print("Data loaded successfully")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    # simple test run
    data = load_data("normalized_train_data.csv")
    print(data.head())


