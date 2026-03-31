import pandas as pd
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


def split_and_scale(
    df: pd.DataFrame,
    target_col: str = "cardio",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    y = df[target_col]
    x = df.drop(columns=[target_col])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler
