import pandas as pd

def load_data(url: str) -> pd.DataFrame:
    """Load dataset from a given URL."""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None


def calculate_mean(df: pd.DataFrame, column: str) -> float:
    """Calculate mean of a column."""
    if df is None or column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset")
    return df[column].mean()


def find_max(df: pd.DataFrame, column: str):
    """Find maximum value in a column."""
    if df is None or column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset")
    return df[column].max()


def filter_by_species(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Filter rows based on species."""
    if df is None or 'species' not in df.columns:
        raise ValueError("Column 'species' not found in dataset")
    return df[df['species'] == species]


if __name__ == "__main__":
    # Dataset URL
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

    # Load data
    df = load_data(url)

    # Perform operations
    try:
        avg_sepal = calculate_mean(df, 'sepal_length')
        max_petal = find_max(df, 'petal_width')
        setosa_data = filter_by_species(df, 'setosa')

        print("Average Sepal Length:", avg_sepal)
        print("Max Petal Width:", max_petal)
        print("\nFirst 5 rows of Setosa species:\n")
        print(setosa_data.head())

    except ValueError as e:
        print("Error:", e)