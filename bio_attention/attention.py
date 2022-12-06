import torch
from torch import nn, einsum
import torch.nn.functional as F

class VanillaSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim = -2) 
        
    def forward(self, q, k, v):
        h = q.shape[-1]
        q = q * (h ** -.5)
        
        A = einsum('b q n h, b k n h -> b q k n', q, k)
        A = self.softmax(A)

        z = einsum('b q k n, b k n h -> b q n h', A, v)
        return z

class RandomSelfAttention(nn.Module):
    def __init__(self, n_random_keys = 64):
        super().__init__()
        self.softmax = nn.Softmax(dim = -2)
        self.n = n_random_keys
        
    def forward(self, q, k, v):
        b, s, nh, h = k.shape
        s2 = q.shape[1]
        
        q = q * (h ** -.5)
        
        indices_select = torch.randint(0, s, (b, s2, self.n)).to(q.device)
        
        indexer = torch.arange(b).view(b, 1, 1)
        k = k[indexer, indices_select]
        v = v[indexer, indices_select]
        
        A = einsum('b q n h, b q k n h -> b q k n', q, k)
        A = self.softmax(A)

        z = einsum('b q k n, b q k n h -> b q n h', A, v)
        
        return z
    
class WindowAttention(nn.Module):
    def __init__(self, window):
        super().__init__()
        assert window % 2 == 1, 'Window size should be an odd integer.'
        
        self.softmax = nn.Softmax(dim = -2)
        self.w = int((window-1)/2)
    
    def forward(self, q, k, v):
        assert k.shape[1] == q.shape[1], 'q and k should have same input length.'
        b, s, nh, h = k.shape
        
        q = q * (h ** -.5)
        
        k = F.pad(k, (0,)*4 + (self.w,)*2).unfold(1, s, 1)
        v = F.pad(v, (0,)*4 + (self.w,)*2).unfold(1, s, 1)
        
        A = einsum('b q n h, b k n h q -> b q k n', q, k)
        
        mask = torch.zeros((nh, s), device = k.device).bool()
        mask = F.pad(mask, (self.w,)*2, value = True).unfold(1, s, 1)
        mask = mask.transpose(0,-1).unsqueeze(0)
        
        mask_value = -torch.finfo(A.dtype).max
        A.masked_fill_(mask, mask_value)
        
        A = self.softmax(A)

        z = einsum('b q k n, b k n h q -> b q n h', A, v)
        return z