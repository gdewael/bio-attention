<div align="center">
<h1>bio-attention</h1>

Simple implementations of attention modules adapted for the biological data domain.

[![PyPi Version](https://img.shields.io/pypi/v/bio-attention.svg)](https://pypi.python.org/pypi/bio-attention/)
[![GitHub license](https://img.shields.io/github/license/gdewael/bio-attention)](https://github.com/gdewael/bio-attention/blob/main/LICENSE)

</div>



# :construction: THIS CODE IS BEING ACTIVELY DEVELOPED :construction:

Don't look for stability here (yet).

## Install
Since PyTorch is a dependency of `bio-attention`, we recommend [installing PyTorch](https://pytorch.org/get-started/locally/) independently first, as your system may require a specific version (e.g. CUDA drivers).

After PyTorch installation, `bio-attention` can be installed using `pip`
```bash
pip install bio-attention
```

## Usage

## Package roadmap

- [x] Embedding layers
  - [x] Continuous
  - [x] Discrete
  - [x] Binary
  - [x] Bin
- [~] Positional encoding schemes
  - [x] Sinusoidal
  - [x] Embedding
  - [x] Continuous
  - [x] Rotary
  - [x] AliBi
  - [x] DPB
  - [x] XL
  - [ ] Test support for multi-dimensional inputs
- [~] Attention modules
  - [x] Vanilla
  - [x] Windowed
  - [x] Random
  - [x] Performer
  - [x] Axial
  - [x] Flatten
  - [x] Encoder
  - [x] Decoder
  - [ ] Cross
- [ ] Tests
- [ ] Typing
- [ ] Docs


# LEGACY documentation
## THIS REPO USED TO BE A 2D SLIDING WINDOW ATTENTION REPO

### 2D Sliding Window Attention

<img src="./bio-attention/img/2Dslidingwindow-attention.png" width="750">

Stand-alone PyTorch implementation of 2D sliding window attention. Introduced by and part of CpG Transformer located at this [repo](https://github.com/gdewael/cpg-transformer) and detailed in our [preprint paper](https://www.biorxiv.org/content/10.1101/2021.06.08.447547v1).

### Contents

`sliding_window_attn.py` contains three PyTorch modules: `RelPositionalWindowEmbedding`, `MultiDimWindowAttention`, and `MultiDimWindowTransformerLayer`. The modules have been programmed in a way so that they can be used to do 1D sliding window attention, as well as >= 2-dimensional sliding window attention. In the multidimensional case, sliding window attention is applied over the first dimension following the batch dimension and full self-attention is applied over all the others.

Sliding windows are efficiently obtained using the `unfold` operation.

Positional embeddings are relative sinusoidal ones as described in [Transformer-XL](https://arxiv.org/abs/1901.02860). Note that positional encodings are applied for the dimension in which sliding windows are applied. To inform the model of position in other dimensions, this should be encoded in the input itself.

### Usage

```python

from sliding_window_attn import MultiDimWindowTransformerLayer

# one layer:
layer = MultiDimWindowTransformerLayer(
    hidden_dim=64,     # number of input & output hidden dimensions (int)
    head_dim=8,        # hidden dimensionality of each SA head (int)
    n_head=8,          # number of SA heads (int)
    ff_dim=256,        # number of feed-forward hidden dimensions (int)
    window=21,         # window size of sliding window, should be odd. (int) (default=21)
    dropout=0.20,      # dropout rate on the self-attention matrix (float) (default=0.20)
    activation='relu', # activation used in feed-forward, either 'relu' or 'gelu' (str) (default='relu')
    layernorm=True     # whether to apply layernorm after attn+res and ff+res (bool) (default=True)
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
# if all positional indices follow on eachother by one: pos = torch.arange(512).unsqueeze(0)

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
For example: the attention matrix (per head) for the above 1D example is: `512 * 21`.
For the 2D example this becomes: `(512*4) * (21*4)`.
And for the 3D example: `(512*4*3) * (21*4*3)`.

## Citation

If you find this repository useful in your research, please cite our [paper](https://www.biorxiv.org/content/10.1101/2021.06.08.447547v1).
```bibtex
@article{dewaele2021cpg,
	author = {Gaetan De Waele and Jim Clauwaert and Gerben Menschaert and Willem Waegeman},
	title = {CpG Transformer for imputation of single-cell methylomes},
	year = {2021},
	doi = {10.1101/2021.06.08.447547},
	URL = {https://www.biorxiv.org/content/early/2021/06/09/2021.06.08.447547}
}
```
