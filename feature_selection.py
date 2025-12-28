import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold

from log_code import setup_logging
logger = setup_logging('feature_selection_heart')

reg_con = VarianceThreshold(threshold=0.0)
reg_quasi = VarianceThreshold(threshold=0.01)

class FEATURE_SELECTION:
    def complete_feature_selection(X_train_num, X_test_num, y_train):
        try:
            logger.info(f"Initial Train Shape: {X_train_num.shape}")
            logger.info(f'{X_train_num.head(10)}')
            logger.info(f"Initial Test Shape : {X_test_num.shape}")
            logger.info(f'{X_test_num.head(10)}')

            # ---------------- Constant Feature Removal ----------------
            reg_con.fit(X_train_num)
            const_features = X_train_num.columns[~reg_con.get_support()]
            logger.info(f"Constant Features Removed: {list(const_features)}")

            X_train_fs = pd.DataFrame(
                reg_con.transform(X_train_num),
                columns=X_train_num.columns[reg_con.get_support()]
            )

            X_test_fs = pd.DataFrame(
                reg_con.transform(X_test_num),
                columns=X_test_num.columns[reg_con.get_support()]
            )

            # ---------------- Quasi-Constant Feature Removal ----------------
            reg_quasi.fit(X_train_fs)
            quasi_features = X_train_fs.columns[~reg_quasi.get_support()]
            logger.info(f"Quasi-Constant Features Removed: {list(quasi_features)}")

            X_train_fs = pd.DataFrame(
                reg_quasi.transform(X_train_fs),
                columns=X_train_fs.columns[reg_quasi.get_support()]
            )

            X_test_fs = pd.DataFrame(
                reg_quasi.transform(X_test_fs),
                columns=X_test_fs.columns[reg_quasi.get_support()]
            )

            logger.info(f"After Variance Filter Train Shape: {X_train_fs.shape}")
            logger.info(f"After Variance Filter Test Shape : {X_test_fs.shape}")

            # ---------------- Pearson Correlation (Numerical vs Target) ----------------
            corr_values = {}
            p_values = {}

            for col in X_train_fs.columns:
                corr, p_val = pearsonr(X_train_fs[col], y_train)
                corr_values[col] = corr
                p_values[col] = p_val

            corr_df = pd.DataFrame({
                'correlation': corr_values,
                'p_value': p_values
            }).sort_values(by='p_value')

            logger.info(f"Feature correlation summary:\n{corr_df}")

            # ---------------- Drop weak features (p-value > 0.05) ----------------
            weak_features = corr_df[corr_df['p_value'] > 0.05].index.tolist()
            logger.info(f"Weak Features Removed (p > 0.05): {weak_features}")

            X_train_final = X_train_fs.drop(columns=weak_features)
            X_test_final = X_test_fs.drop(columns=weak_features)

            logger.info(f"Final Train Shape: {X_train_final.shape}")
            logger.info(f'{X_train_final.head(10)}')
            logger.info(f"Final Test Shape : {X_test_final.shape}")
            logger.info(f'{X_test_final.head(10)}')

            return X_train_final, X_test_final

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error at line {error_line.tb_lineno}: {error_msg}")
