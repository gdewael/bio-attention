# 2D Sliding Window Attention

<img src="./2Dslidingwindow-attention.svg" width="500">

Stand-alone PyTorch implementation of 2D sliding window attention.

## Contents

`2Dslidingwindow_attn.py` contains three PyTorch modules: `RelPositionalWindowEmbedding`, `MultiDimWindowAttention`, and `MultiDimWindowTransformerLayer`. The modules have been programmed in a way so that they can be used to do 1D sliding window attention, as well as $\ge$ 2-dimensional sliding window attention. In the multidimensional case, sliding window attention is applied over the first dimension following the batch dimension and full self-attention is applied over all the others.

Sliding windows are efficiently obtained using the `unfold` operation.

Note that positional encodings are applied for the dimension in which sliding windows are applied. To inform the model of position in other dimensions, this should be encoded in the input itself.

## Usage

```python

from sliding_window_attn import MultiDimWindowTransformerLayer

# one layer:
layer = MultiDimWindowTransformerLayer(
    hidden_dim=64, # number of input & output hidden dimensions (int)
    head_dim=8, # hidden dimensionality of each SA head (int)
    n_head=8, # number of SA heads (int)
    ff_dim=256, # number of feed-forward hidden dimensions (int)
    window=21, # window size of sliding window, should be odd. (int) (default=21)
    dropout=0.20, # dropout rate on the self-attention matrix (float) (default=0.20)
    activation='relu', # activation used in feed-forward, either 'relu' or 'gelu' (str) (default='relu')
    layernorm=True # whether to apply layernorm after attn+res and ff+res (bool) (default=True)
)

# model consisting of 4 layers:
model = nn.Sequential([MultiDimWindowTransformerLayer(64, 8, 8, 256),
                       MultiDimWindowTransformerLayer(64, 8, 8, 256),
                       MultiDimWindowTransformerLayer(64, 8, 8, 256),
                       MultiDimWindowTransformerLayer(64, 8, 8, 256)])



# 2D sequence input:
# batch size = 1
# sequence dim1 length = 512 (sliding window SA)
# sequence dim2 length = 4 (full SA)
# hidden = 64
x = torch.randn(1, 512, 4, 64)
pos = torch.cumsum(torch.randint(1, 7, (1, 512)), 1)

x, pos = model((x, pos))
```

The same model can also be used for 1D sequence inputs:
```python
# batch size = 1
# sequence dim1 length = 512 (sliding window SA)
# hidden = 64
x = torch.randn(1, 512, 64)
pos = torch.cumsum(torch.randint(1, 7, (1, 512)), 1)

x, pos = model((x, pos))
```


Or even 3D (or more) sequence input:
```python
# batch size = 1
# sequence dim1 length = 512 (sliding window SA)
# sequence dim2 length = 4 (full SA)
# sequence dim3 length = 3 (full SA)
# hidden = 64
x = torch.randn(1, 512, 4, 3, 64)
pos = torch.cumsum(torch.randint(1, 7, (1, 512)), 1)

x, pos = model((x, pos))

```

Note that computational complexity will scale quadratically with each added dimension.
For example: the attention matrix (per head) for the above 1D example is: $512 \times 21$.
For the 2D example this becomes: $512 \cdot 4 \times 21 \cdot 4$.
And for the 3D example: $512 \cdot 4 \cdot 3 \times 21 \cdot 4 \cdot 3$.