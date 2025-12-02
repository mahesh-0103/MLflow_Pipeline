import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    """
    Perform basic validation checks:
    - No missing values
    - No empty dataframe
    - All numeric columns contain valid numbers
    """

    # 1. Check if DF is empty
    if df.empty:
        raise ValueError("[VALIDATION ERROR] Dataframe is empty.")

    # 2. Check for missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("[VALIDATION ERROR] Data contains missing values.")

    # 3. Optional: ensure no infinite values
    if not df.replace([float('inf'), -float('inf')], pd.NA).notna().all().all():
        raise ValueError("[VALIDATION ERROR] Infinite values detected.")

    print("[VALIDATION] Passed basic validation.")
    return True
