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
  - [x] Test support for multi-dimensional inputs
- [~] Attention modules
  - [x] Vanilla
  - [x] Windowed
  - [x] Random
  - [x] Performer
  - [x] Encoder
  - [x] Decoder
  - [ ] Cross
  - [x] Support for multi-dim inputs
- [ ] Tests
- [ ] Typing
- [ ] Docs
