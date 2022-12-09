import torch
from torch import nn, einsum
import torch.nn.functional as F
import math

def compl_mod(m, n):
    return int(n * math(m/n) - m)   

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
        
        self.softmax = nn.Softmax(dim = -1)
        self.w = int((window-1)/2)
        
        self.k_ch = window*2
        self.q_ch = self.k_ch - window + 1
        
        u = torch.triu(torch.full((self.q_ch, self.k_ch), True))
        l = torch.tril(torch.full((self.q_ch, self.k_ch), True), self.w*2)
        self.mask = torch.logical_and(u, l).cuda()
    
    def forward(self, q, k, v):
        assert k.shape[1] == q.shape[1], 'q and k should have same input length.'
        b, s, nh, h = k.shape
        
        q = q * (h ** -.5)
        
        pad_q = compl_mod(s, self.q_ch)
        pad_k = compl_mod((s + self.w*2), self.q_ch)
        
        q = F.pad(k, (0,)*5 + (pad_q,)).unfold(1, self.q_ch, self.q_ch)
        k = F.pad(k, (0,)*4 + (self.w, self.w + pad_k)).unfold(1, self.k_ch, self.q_ch)
        v = F.pad(v, (0,)*4 + (self.w, self.w + pad_k)).unfold(1, self.k_ch, self.q_ch)
        
        A = einsum('b c n h q, b c n h k -> b n c q k ', q, k)
        
        mask_value = -torch.finfo(A.dtype).max
        A[:,:].masked_fill_(~self.mask, mask_value)
        A = self.softmax(A)

        z = einsum('b n c q k, b c n h k -> b n c q h', A, v)
        z = z.view(b,nh, -1, h)[:,:,:-pad_q].permute(0,2,1,3)
        
        return z