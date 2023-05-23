import torch
from torch import nn

class DiscreteEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        cls=False,
        cls_axis=-2,
        init_cls_as_mask=False,
    ):
        """Embedding module for discrete data. Uses `nn.Embedding` with added support for masking tokens cls tokens.
        Masking tokens are recognized as torch.nans. Therefore, input should be float dtype.

        Args:
            num_embeddings (int): number of discrete classes.
            embedding_dim (int): number of hidden dimensions of the embeddings
            cls (bool, optional): whether to include cls token. Defaults to True.
            cls_axis (int, optional): which axis in the input to add cls token to. Defaults to -1.
            init_cls_as_mask (bool, optional): initialize the cls token to be equal to the masking token. Defaults to False.
        """
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings + 1, embedding_dim)

        if cls:
            self.cls = CLS(
                embedding_dim,
                cls_axis=cls_axis,
                init=(
                    None
                    if not init_cls_as_mask
                    else self.embedder.weight.data[0].clone()
                ),
            )
        else:
            self.cls = nn.Identity()

    def forward(self, x):
        x = x + 1
        x[torch.isnan(x)] = 0
        x = self.embedder(x.long())
        x = self.cls(x)
        return x


class ContinuousEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        depth = 1,
        norm = False,
        cls=False,
        cls_axis=-2,
        init_cls_as_bias=False,
        init_mask_as_bias=False,
        init_cls_as_mask=False,
    ):
        """Embedding module for continuous data. Uses `nn.Linear` with added support for masking tokens cls tokens.
        Masking tokens are recognized as torch.nans. Therefore, input should be float dtype.

        Args:
            embedding_dim (int): number of hidden dimensions of the embeddings
            cls (bool, optional): whether to include cls token. Defaults to True.
            cls_axis (int, optional): which axis in the input to add cls token to. Defaults to -1.
            init_cls_as_bias (bool, optional): initialize the cls token to be equal to the bias in the `nn.Linear`. Defaults to False.
            init_mask_as_bias (bool, optional): initialize the masking token to be equal to the bias in the `nn.Linear`. Defaults to False.
            init_cls_as_mask (bool, optional): initialize the cls token to be equal to the masking token. Defaults to False.
        """
        super().__init__()
        assert not (init_cls_as_bias and init_cls_as_mask)
        self.embedder = nn.Linear(1, embedding_dim)

        layers = []
        layers.append(nn.Linear(1, embedding_dim))
        layers.append(nn.LayerNorm(embedding_dim) if norm else nn.Identity())

        for _ in range(depth - 1):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(embedding_dim, embedding_dim))
            layers.append(nn.LayerNorm(embedding_dim) if norm else nn.Identity())
        self.embedder = nn.Sequential(*layers)

        with torch.no_grad():
            bias = self.embedder(torch.tensor([[0.]])).unsqueeze(0)

        if init_mask_as_bias:
            self.mask_embedding = nn.Parameter(bias)
        else:
            self.mask_embedding = nn.Parameter(torch.empty(embedding_dim))
            nn.init.uniform_(self.mask_embedding, -1, 1)

        if cls:
            if init_cls_as_bias:
                init = bias
            elif init_cls_as_mask:
                init = self.mask_embedding.data.clone()
            else:
                init = None
            self.cls = CLS(embedding_dim, cls_axis=cls_axis, init=init)
        else:
            self.cls = nn.Identity()

    def forward(self, x):
        out = self.embedder(x.unsqueeze(-1))
        out[torch.isnan(x)] = self.mask_embedding
        out = self.cls(out)
        return out


class BinaryEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        lambda_fn=lambda x: x == 0,
        cls=False,
        cls_axis=-2,
        init_cls_as_mask=False,
    ):
        """Embedding module that binarizes input data and then embeds them using DiscreteEmbedding. Supports masking tokens as NaNs
        For efficiency, consider tokenizing data before training using this module and using DiscreteEmbedding in the actual model.

        Args:
            embedding_dim (int): number of hidden dimensions of the embeddings
            lambda_fn (Callable, optional): function that binarizes data. Defaults to lambdax:x==0.
        """
        super().__init__()
        self.embedder = DiscreteEmbedding(
            2,
            embedding_dim,
            cls=cls,
            cls_axis=cls_axis,
            init_cls_as_mask=init_cls_as_mask,
        )
        self.lambda_fn = lambda_fn

    def forward(self, x):
        nans = torch.isnan(x)
        x = self.lambda_fn(x).float()
        x[nans] = torch.nan
        return self.embedder(x)


class BinEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        bins,
        left_closed=True,
        cls=False,
        cls_axis=-2,
        init_cls_as_mask=False,
    ):
        """Embedding module that bins input data and then embeds them using DiscreteEmbedding. Supports masking tokens as NaNs.
        For efficiency, consider tokenizing data before training using this module and using DiscreteEmbedding in the actual model.

        Args:
            embedding_dim (int): number of hidden dimensions of the embeddings
            bins (torch.Tensor): 1-D tensor that delineates bins. Should be monotonically increasing.
            left_closed (bool): Whether the bins should start including the lower bound or not. Defaults to True
        """
        super().__init__()
        self.embedder = DiscreteEmbedding(
            len(bins),
            embedding_dim,
            cls=cls,
            cls_axis=cls_axis,
            init_cls_as_mask=init_cls_as_mask,
        )
        self.left_closed = left_closed
        self.bins = bins

    def forward(self, x):
        nans = torch.isnan(x)
        bins = self.bin(x).float()
        bins[nans] = torch.nan
        return self.embedder(bins)

    def bin(self, x):
        diff = x[..., None] - self.bins
        z = diff >= 0 if self.left_closed else diff > 0
        z = torch.cat([z, torch.zeros_like(z)[..., [-1]]], dim=-1)
        return torch.clamp(z.int().argmin(2) - 1, min=0)


class CLS(nn.Module):
    def __init__(self, dim, cls_axis=-2, init=None):
        """Module for adding a CLS token to an input

        Args:
            dim (int): number of hidden dimensions of the embeddings
            cls_axis (int, optional): which axis in the input to add cls token to. Defaults to -1.
            init (int, optional): initialize the weights of the CLS token as ... Defaults to None, for uniform [-1, 1] initialization.
        """
        super().__init__()
        if init is not None:
            self.cls_embedding = nn.Parameter(init)
        else:
            self.cls_embedding = nn.Parameter(torch.empty(dim))
            nn.init.uniform_(self.cls_embedding, -1, 1)

        self.cls_axis = cls_axis

    def forward(self, x):
        shp = list(x.shape)
        shp[self.cls_axis] = 1
        x = torch.cat([self.cls_embedding.expand(*shp), x], self.cls_axis)
        return x
