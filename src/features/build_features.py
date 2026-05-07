import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def aggressive_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stricter cleaning used by the advanced stacking model. Removes physiologically
    impossible blood pressure, height and weight values, enforces ap_hi > ap_lo,
    and drops exact duplicates. Returns a copy.
    """
    df = df.copy()
    df = df[(df['ap_hi'].between(80, 250)) & (df['ap_lo'].between(40, 180))]
    df = df[df['ap_hi'] > df['ap_lo']]
    df = df[df['height'].between(140, 210)]
    df = df[df['weight'].between(35, 200)]
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df = df.drop_duplicates()
    return df


def engineer_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rich clinical feature set used by the advanced stacking model. Adds BMI,
    pulse pressure, MAP, ACC/AHA blood-pressure category, BMI category, age
    decade, and a few well-motivated interactions. Drops the raw age/height/
    weight columns so the model trains on the engineered representation.
    """
    df = df.copy()
    df['age_years']      = df['age'] / 365.25
    df['bmi']            = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['map']            = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3.0

    # ACC/AHA 2017 blood-pressure category
    def _bp_cat(row):
        s, d = row['ap_hi'], row['ap_lo']
        if s < 120 and d < 80: return 0  # Normal
        if s < 130 and d < 80: return 1  # Elevated
        if s < 140 or d < 90:  return 2  # Stage 1
        if s < 180 or d < 120: return 3  # Stage 2
        return 4                          # Crisis
    df['bp_cat'] = df.apply(_bp_cat, axis=1)

    bmi = df['bmi']
    df['bmi_cat'] = np.select(
        [bmi < 18.5, bmi < 25, bmi < 30, bmi < 35, bmi < 40],
        [0, 1, 2, 3, 4],
        default=5,
    )

    df['age_decade']   = (df['age_years'] // 10).astype(int)
    df['hypertensive'] = ((df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)).astype(int)
    df['obese']        = (df['bmi'] >= 30).astype(int)
    df['chol_gluc']    = df['cholesterol'] * df['gluc']
    df['smoke_alco']   = df['smoke'] * df['alco']
    df['age_chol']     = df['age_years'] * df['cholesterol']
    df['age_bp']       = df['age_years'] * df['ap_hi']
    df['bmi_age']      = df['bmi'] * df['age_years']

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
    
    x_train, x_test, y_train, y_test, scaler = split_and_scale(df_cleaned)
    print(f"Train-test split completed:")
    print(f"  X_train shape: {x_train.shape}")
    print(f"  X_test shape: {x_test.shape}")
    print(f"  y_train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  y_test distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
