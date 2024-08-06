import torch
import torch.nn as nn
import torch.nn.functional as F

class sae(nn.Module):
    def __init__(self,expand_ratio,channels):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.channels = channels
        self.act = F.relu
        self.b_enc = nn.Parameter(
            torch.zeros(int(expand_ratio*channels))
        )
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    channels, int(expand_ratio*channels)
                )
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                   int(expand_ratio*channels), channels
                )
            )
        )
        self.W_enc.data = self.W_dec.data.T # weight tying for now
        # methdods which change b_dec as a function of the dataset are implemented after init.
        self.b_dec = nn.Parameter(
            torch.zeros(channels)
        )
    
    def encode(self,x):
        hidden = x @ self.W_enc + self.b_enc
        feature = self.act(hidden)
        return feature, hidden
    
    def decode(self,feature):
        x_hat = feature@ self.W_dec + self.b_dec
        return x_hat
    
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    def forward(self,x):
        feature, hidden = self.encode(x)
        x_hat = self.decode(feature)
        return feature, hidden, x_hat