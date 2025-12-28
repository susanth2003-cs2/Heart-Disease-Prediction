'''
In this file we are going to load the data and other ML pipeline techniques
which are needed
'''
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
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from random_sample import RSITechnique
from var_out import VT_OUT
from feature_selection import FEATURE_SELECTION
from data_balance import BALANCING_DATA
from model_training import common
from sklearn.preprocessing import StandardScaler
import pickle

class HEART_DISEASE_PREDICT:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info('Data loaded succesfully')

            logger.info(f'Total Rows in the data : {self.df.shape[0]}')
            logger.info(f'Total columns in the data : {self.df.shape[1]}')
            logger.info(f'Checking Null Values : {self.df.isnull().sum()}')

            self.X = self.df.iloc[:, :-1]  # independent
            self.y = self.df.iloc[:, -1].astype(int)  # dependent

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)

            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')

            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def missing_values(self):
        try:
            logger.info(f'Missing Values')
            logger.info(f"X_train columns: {self.X_train.columns}")
            logger.info(f"X_test columns: {self.X_test.columns}")

            if self.X_train.isnull().sum().any() > 0 or self.X_test.isnull().sum().any() > 0:
                self.X_train, self.X_test = RSITechnique.random_sample_imputation_technique(self.X_train, self.X_test)
            else:
                logger.info(f'No Missing Values')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def vt_out(self):
        try:
            logger.info('Variable Transformation and Outlier Detection')
            for i in self.X_train.columns:
                logger.info(f'{self.X_train[i].dtype}')
            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')

            self.X_train, self.X_test = VT_OUT.variable_transformation_outlier(self.X_train, self.X_test)

            logger.info(f'{self.X_train.columns} --> {self.X_train.shape}')
            logger.info(f'{self.X_test.columns} --> {self.X_test.shape}')

            logger.info('Variable Transformation Completed')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
    '''
    def fs(self):
        try:
            logger.info(f'Before : {self.X_train_num.columns} --> {self.X_train_num.shape}')
            logger.info(f'Before : {self.X_test_num.columns} --> {self.X_test_num.shape}')
            self.X_train_num,self.X_test_num = FEATURE_SELECTION.complete_feature_selection(self.X_train_num,self.X_test_num,self.y_train)
            logger.info(f'After : {self.X_train_num.columns} --> {self.X_train_num.shape}')
            logger.info(f'After : {self.X_test_num.columns} --> {self.X_test_num.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
    '''
    def balance(self):
        try:
            logger.info(f"Before SMOTE - Class Distribution: {self.y_train.value_counts().to_dict()}")
            self.X_train,self.y_train=BALANCING_DATA.balance_data(self.X_train,self.y_train)
            logger.info(f"After SMOTE - Class Distribution: {self.y_train.value_counts().to_dict()}")
            logger.info(f"Balanced X_train shape: {self.X_train.shape}")
            logger.info(f"Balanced y_train shape: {self.y_train.shape}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def data_scaling(self):
        try:
            logger.info(f'{self.X_train.shape}')
            logger.info(f'{self.X_test.shape}')
            logger.info(f'Before \n:{self.X_train}')
            logger.info(f'Before \n:{self.X_test}')
            scale_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

            sc = StandardScaler()
            sc.fit(self.X_train[scale_cols])

            self.X_train[scale_cols] = sc.transform(self.X_train[scale_cols])
            self.X_test[scale_cols] = sc.transform(self.X_test[scale_cols])

            # Save scaler for inference
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(sc, f)

            logger.info(f'{self.X_train.shape}')
            logger.info(f'{self.X_test.shape}')
            logger.info(f'Before \n:{self.X_train}')
            logger.info(f'Before \n:{self.X_test}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def models(self):
        try:
            logger.info(f'Training Started')
            common(self.X_train,self.y_train,self.X_test,self.y_test)
            logger.info(f'Training Completed')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')


if __name__ == "__main__":
    try:
        obj = HEART_DISEASE_PREDICT('C:\\Users\\Rajesh\\Downloads\\Mini Projects\\Heart Disease Prediction\\heart.csv')
        obj.missing_values()
        obj.vt_out()
        #obj.fs()
        obj.balance()
        obj.data_scaling()
        obj.models()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
