<div align="center">
<h1>bio-attention</h1>

Simple implementations of attention modules adapted for the biological data domain.

[![PyPi Version](https://img.shields.io/pypi/v/bio-attention.svg)](https://pypi.python.org/pypi/bio-attention/)
[![GitHub license](https://img.shields.io/github/license/gdewael/bio-attention)](https://github.com/gdewael/bio-attention/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/bio-attention/badge/?version=latest&style=flat-default)](https://bio-attention.readthedocs.io/en/latest/index.html)

</div>

## Why use this package?

There are already plenty of excellent implementations out there that allow you to test out the countless variants of transformers [[1]](https://github.com/facebookresearch/xformers), [[2]](https://github.com/lucidrains/x-transformers).
This repository primarily separates itself from the previous in that it contains positional encodings schemes adapted to allow for irregularly-spaced positions in sequences.
This is useful in, for example: (1) the mass spectral domain (proteomics, metabolomics, ...), where transformers operate on sets of peaks, (2) any kind of (epi)genomic data that measures sites of interests on the genome that are irregularly-spaced (such as WGBS/CpG sites, ATAC-seq/chromatin accessibility, ...).
Additionally, the attention definitions in this repository are compatible with multi-dimensional data, such as the MSAs used in some protein language models, and AlphaFold.

## Install
Since PyTorch is a dependency of `bio-attention`, we recommend [installing PyTorch](https://pytorch.org/get-started/locally/) independently first, as your system may require a specific version (e.g. CUDA drivers).

After PyTorch installation, `bio-attention` can be installed using `pip`
```bash
pip install bio-attention
```

## Note

This package used to be a 2D sliding window attention package. The current formulation of the package does not allow for this type of attention anymore (instead, I recommend to perform axial attention with alternating sliding window attention across one axis and full self-attention across the other). If you want to use 2D sliding window attention, check out the [old version of this repo](https://github.com/gdewael/bio-attention/tree/ac3cb87906a2ff7adf9de393a5d2bbd3bf11eef3).

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
  - [x] Cross
  - [x] Support for multi-dim inputs
- [ ] Add a warning if non-increasing positional indices are used with a decoder attention
- [ ] Add docs clarifying that clf tokens are automatically accounted for if no pos is provided for them
- [ ] Tests
- [x] Typing
- [x] Docs
