import pickle
import numpy as np
import torch
path="sampling/data_n9825opt.pkl"
with open(path,"rb") as w:
    data=pickle.load(w)
x0=[]
u_prev=[]
u0=[]
for m in range(len(data['x0'])):
    #for k in range(len(data['x0'][0])):
        x0.append(data['x0'][m])
        u_prev.append(data['u_prev'][m])
        u0.append(data['u0'][m])
dataset=[torch.Tensor(np.concatenate((np.asarray(x0).squeeze(),np.asarray(u_prev).squeeze()),axis=1))]
dataset=dataset+[torch.Tensor(np.asarray(u0)).squeeze()]

path="datasets/dataset_10000.pt"
torch.save(dataset,path)
#with open(path,"wb") as w:
#    data=pickle.dump(dataset,w)