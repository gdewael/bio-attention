import torch
from torch import nn
from typing import Callable, Optional


class DiscreteEmbedding(nn.Module):
    """Embedding module for discrete data. Uses `nn.Embedding` with added support for masking tokens and cls tokens.
    Masking tokens are recognized as torch.nans. Therefore, input should be float dtype.
    WARNING: Because this module expects a float dtype for what should be integers, there is a limit to the vocab size you can effectively accurately represent.
    In practice, vocab sizes up to 1 million in size should be faithfully represented with float32. For float16, expect issues.


    Parameters
    ----------
    num_embeddings : int
        number of discrete classes.
    embedding_dim : int
        umber of hidden dimensions of the embeddings
    cls : bool, optional
        whether to include cls token, by default False
    cls_axis : int, optional
        which axis in the *embedded* input to add cls token to, by default -2
    init_cls_as_mask : bool, optional
        initialize the cls token to be equal to the masking token, by default False
        If False, cls will be initialized as N(0,1), just like all other tokens.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        cls: bool = False,
        cls_axis: int = -2,
        init_cls_as_mask: bool = False,
    ):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings + 1, embedding_dim)

        if cls:
            self.cls = CLS(
                embedding_dim,
                cls_axis=cls_axis,
                init=(
                    torch.empty(embedding_dim).normal_(0, 1)
                    if not init_cls_as_mask
                    else self.embedder.weight.data[0].clone()
                ),
            )
        else:
            self.cls = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L)

        Returns
        -------
        torch.Tensor
            (B, *, L, H) if cls = False
            (B, *, L+1, H) if cls = True
        """
        x = x + 1
        x[torch.isnan(x)] = 0
        x = self.embedder(x.long())
        x = self.cls(x)
        return x


class ContinuousEmbedding(nn.Module):
    """Embedding module for continuous data. Uses `nn.Linear` with added support for masking tokens cls tokens.
    Masking tokens are recognized as torch.nans. Therefore, input should be float dtype.

    Parameters
    ----------
    embedding_dim : int
        number of hidden dimensions of the embeddings
    depth : int, optional
        depth of the embedding module.
        Will follow this structure:
        Linear -> (Norm if norm) { -> Swish -> Linear -> (Norm if norm) } * (depth-1)
        By default 1, for a linear embedding.
    norm : bool, optional
        Whether to use LayerNorms in the embedding net, by default False
    cls : bool, optional
        whether to include cls token, by default False
    cls_axis : int, optional
        which axis in the *embedded* input to add cls token to, by default -2
    init_cls_as_bias : bool, optional
        initialize the cls token to be equal to the bias in the embedding network, by default False
        Mutually excluse with init_cls_as_mask
        If a multi-layer embedding net is taken, the bias is still defined as the output of said network with a zero input.
        If both this and init_cls_as_mask are False, cls will be initialized as U(-1,1), just like the bias by default.
    init_mask_as_bias : bool, optional
        initialize the mask token to be equal to the bias in the embedding network, by default False
        If a multi-layer embedding net is taken, the bias is still defined as the output of said network with a zero input.
        If False, mask will be initialized as U(-1,1), just like the bias by default.
    init_cls_as_mask : bool, optional
        initialize the cls token to be equal to the masking token, by default False.
        Mutually excluse with init_cls_as_bias
        If both this and init_cls_as_bias are False, cls will be initialized as U(-1,1), just like the bias by default.
    """

    def __init__(
        self,
        embedding_dim: int,
        depth: int = 1,
        norm: bool = False,
        cls: bool = False,
        cls_axis: int = -2,
        init_cls_as_bias: bool = False,
        init_mask_as_bias: bool = False,
        init_cls_as_mask: bool = False,
    ):
        super().__init__()
        assert not (init_cls_as_bias and init_cls_as_mask)

        layers = []
        layers.append(nn.Linear(1, embedding_dim))
        layers.append(nn.LayerNorm(embedding_dim) if norm else nn.Identity())

        for _ in range(depth - 1):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(embedding_dim, embedding_dim))
            layers.append(nn.LayerNorm(embedding_dim) if norm else nn.Identity())
        self.embedder = nn.Sequential(*layers)

        with torch.no_grad():
            bias = self.embedder(torch.tensor([[0.0]])).unsqueeze(0)

        if init_mask_as_bias:
            self.mask_embedding = nn.Parameter(bias)
        else:
            self.mask_embedding = nn.Parameter(
                torch.empty(embedding_dim).uniform_(-1, 1)
            )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L)

        Returns
        -------
        torch.Tensor
            (B, *, L, H) if cls = False
            (B, *, L+1, H) if cls = True
        """
        nans = torch.isnan(x)
        x[nans] = 0
        out = self.embedder(x.unsqueeze(-1))
        out[nans] = self.mask_embedding.to(out)
        out = self.cls(out)
        return out


class BinaryEmbedding(nn.Module):
    """Embedding module that binarizes input data and then embeds them using DiscreteEmbedding.
    Supports masking tokens as NaNs.
    For efficiency, consider tokenizing data before training using this module and using DiscreteEmbedding in the actual model.

    Parameters
    ----------
    embedding_dim : int
        number of hidden dimensions of the embeddings
    lambda_fn : Callable, optional
        function that binarizes data, by default lambdax:x==0
    cls : bool, optional
        whether to include cls token, by default False
    cls_axis : int, optional
        which axis in the *embedded* input to add cls token to, by default -2
    init_cls_as_mask : bool, optional
        initialize the cls token to be equal to the masking token, by default False
        If False, cls will be initialized as N(0,1), just like all other tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        lambda_fn: Callable = lambda x: x == 0,
        cls: bool = False,
        cls_axis: int = -2,
        init_cls_as_mask: bool = False,
    ):
        super().__init__()
        self.embedder = DiscreteEmbedding(
            2,
            embedding_dim,
            cls=cls,
            cls_axis=cls_axis,
            init_cls_as_mask=init_cls_as_mask,
        )
        self.lambda_fn = lambda_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L)

        Returns
        -------
        torch.Tensor
            (B, *, L, H) if cls = False
            (B, *, L+1, H) if cls = True
        """
        nans = torch.isnan(x)
        x = self.lambda_fn(x).float()
        x[nans] = torch.nan
        return self.embedder(x)


class BinEmbedding(nn.Module):
    """Embedding module that bins input data and then embeds them using DiscreteEmbedding.
    Supports masking tokens as NaNs.
    For efficiency, consider tokenizing data before training using this module and using DiscreteEmbedding in the actual model.

    Parameters
    ----------
    embedding_dim : int
        number of hidden dimensions of the embeddings
    bins : torch.Tensor
        1-D tensor that delineates bins. Should be monotonically increasing.
    left_closed : bool, optional
        Whether the bins should start including the lower bound or not, by default True
    cls : bool, optional
        whether to include cls token, by default False
    cls_axis : int, optional
        which axis in the *embedded* input to add cls token to, by default -2
    init_cls_as_mask : bool, optional
        initialize the cls token to be equal to the masking token, by default False
        If False, cls will be initialized as N(0,1), just like all other tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        bins: torch.Tensor,
        left_closed: bool = True,
        cls: bool = False,
        cls_axis: int = -2,
        init_cls_as_mask: bool = False,
    ):
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

    def forward(self, x) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L)

        Returns
        -------
        torch.Tensor
            (B, *, L, H) if cls = False
            (B, *, L+1, H) if cls = True
        """
        nans = torch.isnan(x)
        bins = self.bin(x).float()
        bins[nans] = torch.nan
        return self.embedder(bins)

    def bin(self, x):
        diff = x[..., None] - self.bins
        z = diff >= 0 if self.left_closed else diff > 0
        z = torch.cat([z, torch.zeros_like(z)[..., [-1]]], dim=-1)
        return torch.clamp(z.int().argmin(-1) - 1, min=0)


class CLS(nn.Module):
    """Module for adding a CLS token to an input

    Parameters
    ----------
    dim : int
        number of hidden dimensions of the embeddings
    cls_axis : int, optional
        which axis in the *embedded* input to add cls token to, by default -2
    init : Optional[torch.Tensor], optional
        initial CLS token weights, by default None, for uniform [-1, 1] initialization.
    """

    def __init__(
        self,
        dim: int,
        cls_axis: int = -2,
        init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if init is not None:
            self.cls_embedding = nn.Parameter(init)
        else:
            self.cls_embedding = nn.Parameter(torch.empty(dim))
            nn.init.uniform_(self.cls_embedding, -1, 1)

        self.cls_axis = cls_axis

    def forward(self, x) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L, H)

        Returns
        -------
        torch.Tensor
            (B, *, L+1, H) if cls_axis = -2
        """
        shp = list(x.shape)
        shp[self.cls_axis] = 1
        x = torch.cat([self.cls_embedding.expand(*shp), x], self.cls_axis)
        return x
