import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from utils import plot_results,ToVariable,use_cuda,TorchDataLoader

def train(model,x_train,y_train,epochs=10,batch_size=32,alpha=0.3,beta=0.3):
    optimizer = optim.RMSprop(model.parameters())
    criterion = nn.MSELoss()

    torch_dataloader = TorchDataLoader(batch_size)
    train_loader = torch_dataloader.torch_dataloader(x_train,y_train)

    x_len = x_train.shape[1]
    for epoch in range(0,epochs):
        for batch,(X,Y) in enumerate(train_loader):
            h1,c1,h2,c2,h3,c3 = model.init_state(X.shape[0])
            out_put,h1,c1,h2,c2,h3,c3 = model(X,h1,c1,h2,c2,h3,c3)

            W_mWDN1_H = model.mWDN1_H.weight.data
            W_mWDN1_L = model.mWDN1_L.weight.data
            W_mWDN2_H = model.mWDN2_H.weight.data
            W_mWDN2_L = model.mWDN2_L.weight.data
            L_loss = torch.norm((W_mWDN1_L-model.cmp_mWDN1_L),2)+torch.norm((W_mWDN2_L-model.cmp_mWDN2_L),2)
            H_loss = torch.norm((W_mWDN1_H-model.cmp_mWDN1_H),2)+torch.norm((W_mWDN2_H-model.cmp_mWDN2_H),2)

            optimizer.zero_grad()
            loss = criterion(out_put[:,-1,:], Y[:,-1,:]) + alpha*L_loss + beta*H_loss
            loss.backward()
            optimizer.step()
            print('Epoch: ', epoch+1, '| Batch: ',batch+1, '| Loss: ',loss.detach())

    torch.save(model, 'model/model.pkl')

def test(model,x_test,y_test,data_df_combined_clean):
    model = torch.load('model/model.pkl')
    model.eval()
    x_test = ToVariable(x_test).double()
    h1,c1,h2,c2,h3,c3= model.init_state(x_test.shape[0])    
    seq_len = x_test.shape[1]

    pred_dat,h1,c1,h2,c2,h3,c3 =  model(x_test,h1,c1,h2,c2,h3,c3)
        
    pred_dat=np.array(pred_dat.detach().numpy())

    #De-standardize predictions
    preds_unstd = pred_dat * data_df_combined_clean.iloc[:,-1].std() + data_df_combined_clean.iloc[:,-1].mean()
    y_test_unstd = y_test * data_df_combined_clean.iloc[:,-1].std() + data_df_combined_clean.iloc[:,-1].mean()

    mrse = np.sqrt(((preds_unstd[:,-1,:] - y_test_unstd[:,-1,:]) ** 2)).mean(axis=0)
    print('The mean square error is: %f' % mrse)

    plot_results(preds_unstd[:,-1,:],y_test_unstd[:,-1,:])
