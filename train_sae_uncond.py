import torch
import os
from sae import sae
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import wandb
ACT_PATH = "/home/yifulu/work/sae/SiT_IN10_20lay_t0.1_uncond"
epoch = 50
BS = 1024
lr = 4e-4
l1_coeff = 8e-5
expand = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(entity="yflpersonal",project="sae_SiT_IN10_20lay_t0.1",
            name=f"epo_{epoch}_bs_{BS}_lr_{lr}_l1_{l1_coeff}_expand_{expand}_ty_uncond",)
class act_data(Dataset):
    def __init__(self,act_path):
        files = os.listdir(act_path)
        acts = []
        for file in files:
            acts.append(torch.load(os.path.join(act_path,file)))
        self.data = torch.cat(acts)
        self.data = self.data.view(-1,self.data.shape[-1])
        print(self.data.shape)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        return self.data[idx]
    def _dim(self):
        return self.data.shape[-1]



dataset = act_data(ACT_PATH)
dataloader = DataLoader(dataset,batch_size=BS,shuffle=True,num_workers=4,pin_memory=True,prefetch_factor=8)

model = sae(expand,dataset._dim()).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

mse_loss = nn.MSELoss()
i = 0
for epo in range(epoch):
    model.train()
    for batch in (dataloader):
        batch = batch.to(device)
        feature, hidden, x_hat = model(batch)
        recon = mse_loss(batch,x_hat)
        l1_loss = torch.norm(feature,1,dim=-1).mean()
        loss = recon + l1_coeff*l1_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.set_decoder_norm_to_unit_norm()
        i += 1
        
        if i%1000 == 0:
            print(f"step {i} loss {loss.item()} mse {recon.item()} l1 {l1_loss.item()}")
            wandb.log({'epoch': epo, 'loss':loss.item(), 
                        'mse': recon.item(), "l1": l1_loss.item()})
torch.save(model.state_dict(),f"epo_{epoch}_bs_{BS}_lr_{lr}_l1_{l1_coeff}_expand_{expand}_ty_sae_uncond.pth")
print("Done")
