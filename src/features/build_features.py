import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def compute_gender_weights(gender_train):
    """
    Return per-sample weights that upweight the minority gender so both
    groups contribute equally to the loss
    """
    gender_arr = np.asarray(gender_train)
    counts = {g: (gender_arr == g).sum() for g in np.unique(gender_arr)}
    n_groups = len(counts)
    n_total = len(gender_arr)
    weights = np.array([
        n_total / (n_groups * counts[g]) for g in gender_arr
    ])
    return weights


def smote_balance_gender(X_train_scaled, y_train, gender_train, random_state=42):
    """
    Oversample the minority gender in the training set using SMOTE
    """
    from imblearn.over_sampling import SMOTE

    gender_arr = np.asarray(gender_train)
    y_arr = np.asarray(y_train)

    # Carry y along as an extra feature so SMOTE interpolates it too
    Xy = np.column_stack([X_train_scaled, y_arr])

    smote = SMOTE(random_state=random_state)
    Xy_balanced, _ = smote.fit_resample(Xy, gender_arr)

    X_balanced = Xy_balanced[:, :-1]
    # Round the interpolated y back to binary
    y_balanced = np.round(Xy_balanced[:, -1]).astype(int)

    return X_balanced, y_balanced

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # filter extreme blood pressure values
    df = df[(df["ap_hi"] > 40) & (df["ap_hi"] < 370)]
    df = df[(df["ap_lo"] > 0) & (df["ap_lo"] < 250)]

    # drop id to prevent data leakage
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features: BMI and age in years"""
    df = df.copy()

    # Create BMI: weight (kg) / (height (cm) / 100)^2
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Convert age from days to years
    df['age_years'] = df['age'] / 365.25

    # Drop raw features that are now represented in engineered features
    df = df.drop(columns=['age', 'height', 'weight'])

    return df

def split_and_scale(
    df: pd.DataFrame,
    target_col: str = "cardio",
    test_size: float = 0.2,
    random_state: int = 42,
    use_feature_engineering: bool = False,
) -> tuple:
    """
    Split and scale data for model training

    Args:
        df: DataFrame with features
        target_col: Name of target column
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        use_feature_engineering: If True, apply BMI and age_years engineering

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Apply feature engineering if requested
    if use_feature_engineering:
        df = engineer_features(df)
    # For baseline: keep all original features as-is

    y = df[target_col]
    x = df.drop(columns=[target_col])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler, x_train

if __name__ == "__main__":
    df = pd.read_csv("../../data/raw/cardio_train.csv", sep=";")
    print(f"Loaded data shape: {df.shape}")

    df_cleaned = clean_data(df)
    print(f"Cleaned data shape: {df_cleaned.shape}")

    x_train, x_test, y_train, y_test, scaler, _ = split_and_scale(df_cleaned)
    print(f"Train-test split completed:")
    print(f"  X_train shape: {x_train.shape}")
    print(f"  X_test shape: {x_test.shape}")
    print(f"  y_train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  y_test distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
