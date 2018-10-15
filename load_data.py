import pandas as pd
import numpy as np
from sklearn import preprocessing

import time
import matplotlib.pyplot as plt
from numpy import newaxis

def load_data(data_path,P,step):
    num_logs = P+step
    df = pd.read_csv(data_path)

    data_np = np.zeros((len(df),num_logs))
    data_df_combined = pd.DataFrame(data_np)
    data_df_combined.loc[:,0] = df['Henry Hub Natural Gas Spot Price Dollars per Million Btu'].data

    for i in range(1, num_logs):
        data_df_combined.loc[:,i] = data_df_combined.shift(-i)

    data_df_combined_clean = data_df_combined.dropna()
    data_df_combined_clean = data_df_combined_clean.reset_index()
    data_df_combined_clean.drop('index',axis=1,inplace=True)
    data_combined_standardized = preprocessing.scale(data_df_combined_clean)

    train_split = round(0.8 * data_combined_standardized.shape[0])
    val_split = round(0.9 * data_combined_standardized.shape[0])
    print("all len",data_combined_standardized.shape[0])
    print("train_split",train_split)

    X = data_combined_standardized[:,:P]
    Y = data_combined_standardized[:,P:]

    X_train = X[:train_split]
    Y_train = Y[:train_split]
    X_test = X[train_split:]
    Y_test = Y[train_split:]

    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
    Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],1))

    return X_train,Y_train,X_test,Y_test,data_df_combined_clean


