# src/data_pipeline/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from typing import Tuple, List

# Define default max features to keep for performance 
MAX_FEATURES_TO_KEEP = 1000 
MAX_SAMPLE_SIZE = 20000 

def preprocess(df: pd.DataFrame, target_col: str, max_sample_size: int = MAX_SAMPLE_SIZE):
    
    # --- 0. Initialize Final Variables (CRITICAL FIX for UnboundLocalError) ---
    # These will be overwritten, but initializing prevents the UnboundLocalError
    # if the feature selection/concatenation steps are skipped due to zero features.
    X_train_final = pd.DataFrame()
    X_val_final = pd.DataFrame()
    X_test_final = pd.DataFrame()
    # --- END CRITICAL FIX ---

    # --- NEW: SAMPLING FOR PERFORMANCE ---
    if len(df) > max_sample_size and max_sample_size > 0:
        print(f"[PREPROCESS] Sampling down from {len(df)} rows to {max_sample_size} for speed.")
        df = df.sample(n=max_sample_size, random_state=42).reset_index(drop=True)
    # --- END SAMPLING ---

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Target Encoding (if classification)
    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
    
    if not is_regression and not pd.api.types.is_integer_dtype(y):
        print("[PREPROCESS] Encoding non-numeric classification target.")
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)

    # Identify numeric and non-numeric columns
    numeric_cols: List[str] = X.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols: List[str] = X.select_dtypes(exclude=['number']).columns.tolist()

    # 1. Split Data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print(f"[PREPROCESS] Split sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    
    
    # --- Feature Processing and Combination ---
    
    df_list_train, df_list_val, df_list_test = [], [], []

    # A. Numeric Feature Processing (Scaling/Imputation)
    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        
        X_train_num = X_train[numeric_cols]
        X_val_num = X_val[numeric_cols]
        X_test_num = X_test[numeric_cols]

        X_train_imputed = imputer.fit_transform(X_train_num)
        X_train_scaled = scaler.fit_transform(X_train_imputed)

        X_val_scaled = scaler.transform(imputer.transform(X_val_num))
        X_test_scaled = scaler.transform(imputer.transform(X_test_num))

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=X_train.index)
        X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=numeric_cols, index=X_val.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_cols, index=X_test.index)
        
        df_list_train.append(X_train_scaled_df)
        df_list_val.append(X_val_scaled_df)
        df_list_test.append(X_test_scaled_df)

    # B. Non-Numeric Feature Processing (OHE)
    if non_numeric_cols:
        X_train_non_num = X_train[non_numeric_cols].astype('category')
        X_val_non_num = X_val[non_numeric_cols].astype('category')
        X_test_non_num = X_test[non_numeric_cols].astype('category')
        
        X_train_ohe = pd.get_dummies(X_train_non_num, drop_first=True, dummy_na=False)
        X_val_ohe = pd.get_dummies(X_val_non_num, drop_first=True, dummy_na=False)
        X_test_ohe = pd.get_dummies(X_test_non_num, drop_first=True, dummy_na=False)
        
        train_ohe_cols = X_train_ohe.columns
        X_val_ohe = X_val_ohe.reindex(columns=train_ohe_cols, fill_value=0)
        X_test_ohe = X_test_ohe.reindex(columns=train_ohe_cols, fill_value=0)
        
        df_list_train.append(X_train_ohe)
        df_list_val.append(X_val_ohe)
        df_list_test.append(X_test_ohe)

    # Combine features (This step NOW defines X_train_final, etc.)
    if df_list_train:
        # Concatenate and reset index once at the end
        X_train_final = pd.concat(df_list_train, axis=1).reset_index(drop=True)
        X_val_final = pd.concat(df_list_val, axis=1).reset_index(drop=True)
        X_test_final = pd.concat(df_list_test, axis=1).reset_index(drop=True)
    else:
        # Safety check: If no features were processed, return empty DataFrames
        return X_train_final, X_val_final, X_test_final, y_train, y_val, y_test


    # 3. Univariate Feature Selection (SelectKBest)
    final_feature_count = X_train_final.shape[1] # Line 52: This is now safe
    k_features = min(MAX_FEATURES_TO_KEEP, final_feature_count)

    if final_feature_count > k_features:
        print(f"[PREPROCESS] Applying SelectKBest: reducing features from {final_feature_count} to {k_features}.")
        selector = SelectKBest(score_func=f_classif, k=k_features)
        
        # SelectKBest requires non-sparse matrix.
        X_train_for_selection = X_train_final.values if np.issubdtype(X_train_final.values.dtype, np.floating) else X_train_final.astype(np.float64).values
        selector.fit(X_train_for_selection, y_train)
        
        selected_cols = X_train_final.columns[selector.get_support()]
        
        X_train_final = X_train_final[selected_cols]
        X_val_final = X_val_final[selected_cols]
        X_test_final = X_test_final[selected_cols]
        
    print(f"[PREPROCESS] Final post-selection features: {X_train_final.shape[1]}")


    # 4. Memory Optimization (Final Step)
    if numeric_cols:
        X_train_final[numeric_cols] = X_train_final[numeric_cols].astype(np.float32)
        # ... (rest of memory optimization) ...

    # Final cleanup
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("[PREPROCESS] Completed splitting, imputation, and scaling of numeric features.")
    
    return X_train_final, X_val_final, X_test_final, y_train, y_val, y_test