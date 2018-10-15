from load_data import load_data
from model import Wavelet_LSTM
from train import train,test
import numpy as np

def main():
    data_path = "./Data/GasPrice.csv"
    P = 12  #sequence length
    step = 1 #ahead predict steps

    X_train,Y_train,X_test,Y_test,data_df_combined_clean = load_data(data_path,P=P,step=step)
    print(X_train.shape)
    print(Y_train.shape)
    
    model = Wavelet_LSTM(P,32,1)
    model = model.double()
    train(model,X_train,Y_train,epochs=20)
    test(model,X_test,Y_test,data_df_combined_clean)


if __name__ == "__main__":
    main()