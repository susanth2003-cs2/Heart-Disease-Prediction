import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('var_out')

from scipy import stats

class VT_OUT:
    def variable_transformation_outlier(X_train, X_test):
        try:
            logger.info(f"Initial Train Shape: {X_train.shape}")
            logger.info(f'{X_train.head(10)}')
            logger.info(f"Initial Test Shape : {X_test.shape}")
            logger.info(f'{X_test.head(10)}')

            PLOT_PATH = "all_plots"
            # ================= BEFORE PLOTS =================
            for col in X_train.columns:
                plt.figure()
                X_train[col].plot(kind='kde', color='r')
                plt.title(f'KDE-{col}')
                plt.savefig(f'{PLOT_PATH}/kde_before_{col}.png')
                plt.close()
            for col in X_train.columns:
                plt.figure()
                sns.boxplot(x=X_train[col])
                plt.title(f'Boxplot-{col}')
                plt.savefig(f'{PLOT_PATH}/boxplot_before_{col}.png')
                plt.close()

            # ================= TRANSFORMATION + CAPPING =================
            # Continuous columns identified from EDA
            cap_cols = ['chol', 'thalach', 'trestbps', 'oldpeak']
            log_cols = ['chol', 'oldpeak']

            for col in cap_cols:
                Q1 = X_train[col].quantile(0.25)
                Q3 = X_train[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                # Apply capping using TRAIN statistics
                X_train[col] = X_train[col].clip(lower, upper)
                X_test[col] = X_test[col].clip(lower, upper)

                logger.info(f'Outlier capping applied on {col}')

            # Log transformation (only positive-skewed columns)
            for col in log_cols:
                X_train[col] = np.log1p(X_train[col])
                X_test[col] = np.log1p(X_test[col])

                logger.info(f'Log transformation applied on {col}')

            # ================= AFTER PLOTS =================
            for col in X_train.columns:
                plt.figure()
                X_train[col].plot(kind='kde', color='g')
                plt.title(f'KDE-{col}')
                plt.savefig(f'{PLOT_PATH}/kde_after_{col}.png')
                plt.close()
            for col in X_train.columns:
                plt.figure()
                sns.boxplot(x=X_train[col])
                plt.title(f'Boxplot-{col}')
                plt.savefig(f'{PLOT_PATH}/boxplot_after_{col}.png')
                plt.close()

            return X_train, X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error at line {error_line.tb_lineno}: {error_msg}')
