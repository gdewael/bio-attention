import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
import math
from typing import Optional, Tuple


class Base(nn.Module):
    def __init__(self, divide=1.0, learned=False):
        super().__init__()
        self.learned = learned
        if not learned:
            self.div = divide
        else:
            self.div = nn.Parameter(torch.tensor(math.log(divide)))

        self.forward = self.mod_x

    def mod_x(self, x, pos=None, **kwargs):
        return x

    def mod_qkv(self, q, k, v, pos=None, **kwargs):
        return q, k, v

    def mod_mask(self, mask, q, k, v, pos=None, **kwargs):
        return mask

    def apply_pos_division(self, x):
        if not self.learned:
            return x / self.div
        else:
            return x / self.div.exp().clamp(min=1e-8)

    @staticmethod
    def default_pos_x(x):
        return torch.arange(x.size(-2), device=x.device).expand(*x.shape[:-1])


class Sinusoidal(Base):
    """Sinusoidal positional embedding as in Vaswani et al. (2017)
    Supports specifying positions, masking, and division of positional range.

    Parameters
    ----------
    dim : int
        Hidden size of the embeddings
    divide : float, optional
        divide positions by this factor, useful for large (or small) numerical ranges, by default 1.0
    learned_div : bool, optional
        Whether to learn the division factor, if True, div value initialization is as provided by divide argument, by default False
    """

    def __init__(self, dim: int, divide: float = 1.0, learned_div: bool = False):
        super().__init__(divide=divide, learned=learned_div)
        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def mod_x(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Modify X

        Parameters
        ----------
        x : torch.Tensor
            (B,*,L,H)
        pos : Optional[torch.Tensor], optional
            (B,*,L) or (B,*,L-x), by default None for computing positions from 0 to L-1
            If sequence length is smaller than x, will pad tokens on the lefthand side to not have any positional encodings added.
        mask : Optional[torch.Tensor], optional
            (B,*,L) or (B,*,L-x), by default None
            A boolean mask can be used to explicitly indicate which positions should not have positional encodings added.
            If sequence length is smaller than x, will pad tokens on the lefthand side to have a positional encoding added.

        Returns
        -------
        torch.Tensor
            (B,*,L,H)
        """
        shp = list(x.shape)
        l = shp[-2]
        assert shp[-1] == self.dim

        pos_emb = self.default_pos_x(x).to(self.inv_freq) if pos is None else pos
        shp[-2] = pos_emb.shape[-1]

        sinusoid_inp = self.apply_pos_division(pos_emb.unsqueeze(-1) * self.inv_freq)
        pos_emb = (
            torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
            .view(*shp)
            .to(x.dtype)
        )

        if mask is not None:
            assert mask.dtype == torch.bool
            mask = F.pad(mask, (pos_emb.shape[-2] - mask.shape[-1], 0), value=True)
            pos_emb = pos_emb * mask.unsqueeze(-1)

        if l != shp[-2]:
            pos_emb = F.pad(pos_emb, (0, 0, l - shp[-2], 0))

        return x + pos_emb


class LearnedVocab(Base):
    """Learned vocab as in Devlin et al. (2018)
    Supports specifying positions.
    Only works for discrete positional indices.

    Parameters
    ----------
    dim : int
        Hidden size of the embeddings
    max_seq_len : int
        Maximum sequence length or vocab size of the learned embeddings
    """

    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def mod_x(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Modify X

        Parameters
        ----------
        x : torch.Tensor
            (B,*,L,H)
        pos : Optional[torch.Tensor], optional
            (B,*,L) or (B,*,L-x), by default None for computing positions from 0 to L-1
            If sequence length is smaller than x, will pad tokens on the lefthand side to not have any positional encodings added.
        mask : Optional[torch.Tensor], optional
            (B,*,L) or (B,*,L-x), by default None
            A boolean mask can be used to explicitly indicate which positions should not have positional encodings added.
            If sequence length is smaller than x, will pad tokens on the lefthand side to have a positional encoding added.

        Returns
        -------
        torch.Tensor
            (B,*,L,H)
        """
        shp = list(x.shape)
        assert shp[-1] == self.emb.embedding_dim
        pos = self.default_pos_x(x).long() if pos is None else pos

        pos_emb = self.emb(pos)

        if mask is not None:
            assert mask.dtype == torch.bool
            mask = F.pad(mask, (pos_emb.shape[-2] - mask.shape[-1], 0), value=True)
            pos_emb = pos_emb * mask.unsqueeze(-1)

        l = pos_emb.shape[-2]
        if l != shp[-2]:
            pos_emb = F.pad(pos_emb, (0, 0, shp[-2] - l, 0))

        return x + pos_emb


class LearnedContinuous(Base):
    """Learned embeddings with a continuity between absolute positional indices, as learned by a series of linear layers.
    Supports specifying positions, and division of positional range.

    Parameters
    ----------
    dim : int
        Hidden size of the embeddings
    depth : int, optional
        Number of hidden layers in the positional embedding network.
        Will follow this structure:
        Linear -> (Norm if norm) { -> Swish -> Linear -> (Norm if norm) } * (depth-1)
        By default 1, for a linear embedding.
    norm : bool, optional
        Whether to use LayerNorms in the embedding net, by default False
    divide : float, optional
        divide positions by this factor, useful for large (or small) numerical ranges, by default 1.0
    learned_div : bool, optional
        Whether to learn the division factor, if True, div value initialization is as provided by divide argument, by default False
    """

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        norm: bool = False,
        divide: float = 1.0,
        learned_div: bool = False,
    ):
        super().__init__(divide=divide, learned=learned_div)

        self.mlp = nn.ModuleList([])
        self.mlp.append(
            nn.Sequential(
                nn.Linear(1, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
            )
        )

        for _ in range(depth - 1):
            self.mlp.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim) if norm else nn.Identity(),
                )
            )

    def mod_x(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Modify X

        Parameters
        ----------
        x : torch.Tensor
            (B,*,L,H)
        pos : Optional[torch.Tensor], optional
            (B,*,L) or (B,*,L-x), by default None for computing positions from 0 to L-1
            If sequence length is smaller than x, will pad tokens on the lefthand side to not have any positional encodings added.
        mask : Optional[torch.Tensor], optional
            (B,*,L) or (B,*,L-x), by default None
            A boolean mask can be used to explicitly indicate which positions should not have positional encodings added.
            If sequence length is smaller than x, will pad tokens on the lefthand side to have a positional encoding added.

        Returns
        -------
        torch.Tensor
            (B,*,L,H)
        """
        shp = list(x.shape)

        pos = self.default_pos_x(x).to(x) if pos is None else pos

        pos_emb = self.apply_pos_division(pos).unsqueeze(-1)
        for layer in self.mlp:
            pos_emb = layer(pos_emb)

        if mask is not None:
            assert mask.dtype == torch.bool
            mask = F.pad(mask, (pos_emb.shape[-2] - mask.shape[-1], 0), value=True)
            pos_emb = pos_emb * mask.unsqueeze(-1)

        l = pos_emb.shape[-2]
        if l != shp[-2]:
            pos_emb = F.pad(pos_emb, (0, 0, shp[-2] - l, 0))

        return x + pos_emb


class Rotary(Base):
    """Rotary embedding as in RoFormer / RoPE
    Supports specifying positions and division of positional range.

    Parameters
    ----------
    head_dim : int
        Hidden dimensions per head
    n_dims : Optional[int], optional
        number of dimensions (per head) to apply rotations on.
            Can be used to control how strong the positional bias should be.
            By default None to use all dims.
    divide : float, optional
        divide positions by this factor, useful for large (or small) numerical ranges, by default 1.0
    learned_div : bool, optional
        Whether to learn the division factor, if True, div value initialization is as provided by divide argument, by default False
    """

    def __init__(
        self,
        head_dim: int,
        n_dims: Optional[int] = None,
        divide: float = 1.0,
        learned_div: bool = False,
    ):
        super().__init__(divide=divide, learned=learned_div)

        thetas = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        if n_dims is not None:
            assert n_dims % 2 == 0
            thetas[(n_dims // 2) :] = 0
        self.register_buffer("thetas", thetas)

    def mod_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_q_k: Optional[torch.Tensor] = None,
        pos_q: Optional[torch.Tensor] = None,
        pos_k: Optional[torch.Tensor] = None,
        self_attn_mode: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Modify Q, K and V

        Parameters
        ----------
        q : torch.Tensor
            (B, *, L1, NH, H)
        k : torch.Tensor
            (B, *, L2, NH, H)
        v : torch.Tensor
            (B, *, L2, NH, H)
        pos_q_k : Optional[torch.Tensor], optional
            (B, *, L). Positions of q and k in self attention mode. Requires L1 = L2
            By default None for computing positions from 0 to L-1
        pos_q : Optional[torch.Tensor], optional
            (B, *, L1). Positions of q in cross attention mode
            By default None for computing positions from 0 to L1-1
        pos_k : Optional[torch.Tensor], optional
            (B, *, L2). Positions of k in cross attention mode
            By default None for computing positions from 0 to L2-1
        self_attn_mode : bool, optional
            Whether to use the same positions for q and k (pos_q_k) or use different positions for each (pos_q) and (pos_k)
            by default True

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            q, k, v [(B, *, L1, NH, H), (B, *, L2, NH, H), (B, *, L2, NH, H)]
        """
        if self_attn_mode:
            assert q.shape[-3] == k.shape[-3]
            pos_q_k = self.default_pos_x(q[..., 0, :]) if pos_q_k is None else pos_q_k
            sin, cos = self.get_rotations(pos_q_k)

            l_sin, l_q = sin.shape[-3], q.shape[-3]
            if l_sin != l_q:
                sin, cos = map(
                    lambda t: F.pad(t, (0, 0, 0, 0, l_q - l_sin, 0), (sin, cos))
                )

            q = q * cos.to(q) + self.rotate_every_two(q) * sin.to(q)
            k = k * cos.to(k) + self.rotate_every_two(k) * sin.to(k)
        else:
            pos_q = self.default_pos_x(q[..., 0, :]) if pos_q is None else pos_q
            pos_k = self.default_pos_x(k[..., 0, :]) if pos_k is None else pos_k
            sin_q, cos_q = self.get_rotations(pos_q)
            sin_k, cos_k = self.get_rotations(pos_k)

            l_sin, l_q = sin_q.shape[-3], q.shape[-3]
            if l_sin != l_q:
                sin_q, cos_q = map(
                    lambda t: F.pad(t, (0, 0, 0, 0, l_q - l_sin, 0), (sin_q, cos_q))
                )
            l_sin, l_k = sin_k.shape[-3], k.shape[-3]
            if l_sin != l_k:
                sin_k, cos_k = map(
                    lambda t: F.pad(t, (0, 0, 0, 0, l_k - l_sin, 0), (sin_k, cos_k))
                )

            q = q * cos_q.to(q) + self.rotate_every_two(q) * sin_q.to(q)
            k = k * cos_k.to(k) + self.rotate_every_two(k) * sin_k.to(k)

        return q, k, v

    def get_rotations(self, pos):
        mthetas = (
            self.apply_pos_division(pos).to(self.thetas)[..., None] * self.thetas
        )  # B, *, h
        sin, cos = map(
            lambda t: repeat(t, "b ... h  -> b ... (h j)", j=2),
            (mthetas.sin(), mthetas.cos()),
        )
        sin, cos = map(lambda t: t.unsqueeze(-2), (sin, cos))
        return sin, cos

    @staticmethod
    def rotate_every_two(x):
        x = x.clone()
        x = rearrange(x, "... (d j) -> ... d j", j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d j -> ... (d j)")


class ALiBi(Base):
    """Attention with linear biases as in Press et al. 2022
    Supports specifying positions, division of positional range, using only a fraction of the heads, and asymmetric bias for bidirectional cases.

    Parameters
    ----------
    n_heads : int
        Number of heads
    use_n_heads : bool, optional
        Number of heads to use. Can be used to control how strong the positional bias should be, by default None to use all heads
    asymmetric : bool, optional
        Whether to use assymetric positional biases to differentiate between negative or positive relative positions.
        Implemented according to solution #3 proposed here https://github.com/ofirpress/attention_with_linear_biases/issues/5.
        By default False
    divide : float, optional
        divide positions by this factor, useful for large (or small) numerical ranges, by default 1.0
    learned_div : bool, optional
        Whether to learn the division factor, if True, div value initialization is as provided by divide argument, by default False
    """

    def __init__(
        self,
        n_heads: int,
        use_n_heads: bool = None,
        asymmetric: bool = False,
        divide: float = 1.0,
        learned_div: bool = False,
    ):
        super().__init__(divide=divide, learned=learned_div)

        self.asymmetric = asymmetric
        use_n_heads = n_heads if use_n_heads is None else use_n_heads
        if not asymmetric:
            slopes = torch.tensor(
                self._get_slopes(use_n_heads) + [0] * (n_heads - use_n_heads)
            )
        else:
            slopes = self._get_slopes(use_n_heads)
            slopes = [slopes[(i // 2) * 2] for i in range((len(slopes)))]
            slopes = torch.tensor(slopes + [0] * (n_heads - use_n_heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes)

        self.div = divide

    def mod_mask(
        self,
        mask: Optional[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_q_k: Optional[torch.Tensor] = None,
        pos_q: Optional[torch.Tensor] = None,
        pos_k: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Modify mask

        Parameters
        ----------
        mask : Optional[torch.Tensor]
            (B, *, NH, L1, L2), can pass None for no pre-existing mask.
        q : torch.Tensor
            (B, *, L1, NH, H)
        k : torch.Tensor
            (B, *, L2, NH, H)
        v : torch.Tensor
            (B, *, L2, NH, H)
        pos_q_k : Optional[torch.Tensor], optional
            (B, *, L). Positions of q and k in self attention mode. Requires L1 = L2
            By default None for using pos_q and pos_k instead
        pos_q : Optional[torch.Tensor], optional
            (B, *, L1). Positions of q in cross attention mode
            Ignored if pos_q_k is specified
            By default None for computing positions from 0 to L1-1
        pos_k : Optional[torch.Tensor], optional
            (B, *, L2). Positions of k in cross attention mode
            Ignored if pos_q_k is specified
            By default None for computing positions from 0 to L2-1

        Returns
        -------
        torch.Tensor
            (B, *, NH, L1, L2)
        """
        if pos_q_k is not None:
            pos_q = pos_k = pos_q_k

        elif (pos_q is None) and (pos_k is None):
            pos_q = self.default_pos_x(q[..., 0, :]) if pos_q is None else pos_q
            pos_k = self.default_pos_x(k[..., 0, :]) if pos_k is None else pos_k

        relative_pos = pos_q[..., :, None] - pos_k[..., None, :]
        bias = self.apply_pos_division(relative_pos.unsqueeze(-3) * self.slopes)

        if self.asymmetric:
            bias = self.asymmetric_bias(bias)
        bias = torch.abs(bias)

        if mask is not None:
            l1, l2 = q.shape[-3], k.shape[-3]
            mask_l1, mask_l2 = mask.shape[-2], mask.shape[-1]
            if (l1 != mask_l1) or (l2 != mask_l2):
                mask = F.pad(mask, (l2 - mask_l2, 0, l1 - mask_l1, 0))

        return (0 if mask is None else mask) - bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    @staticmethod
    def asymmetric_bias(bias):
        bias[:, ::2][bias[:, ::2] > 0] = torch.finfo(bias.dtype).max
        bias[:, 1::2][bias[:, 1::2] < 0] = torch.finfo(bias.dtype).max
        return bias


class DPB(Base):
    """Dynamic positional bias.
    Computes a bias to every element of the attention matrix based on their relative position via a parameterized MLP
    Described in https://arxiv.org/abs/2108.00154, https://arxiv.org/abs/2111.09883,
    https://github.com/lucidrains/x-transformers#dynamic-positional-bias
    Supports specifying positions and division of positional range

    Parameters
    ----------
    dim : int
        number of hidden dimensions of the MLP
    n_heads : int
        Number of heads in the model
    depth : int, optional
        Number of hidden layers in the positional embedding network.
        Will follow this structure:
        Linear -> (Norm if norm) { -> Swish -> Linear -> (Norm if norm) } * (depth-1)
        By default 1, for a linear embedding.
    norm : bool, optional
        Whether to use LayerNorms in the embedding net, by default False
    divide : float, optional
        divide positions by this factor, useful for large (or small) numerical ranges, by default 1.0
    learned_div : bool, optional
        Whether to learn the division factor, if True, div value initialization is as provided by divide argument, by default False
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        depth: int = 1,
        norm: bool = False,
        divide: float = 1.0,
        learned_div: bool = False,
    ):
        super().__init__(divide=divide, learned=learned_div)
        self.mlp = nn.ModuleList([])
        self.mlp.append(
            nn.Sequential(
                nn.Linear(1, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
            )
        )

        for _ in range(depth - 1):
            self.mlp.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim) if norm else nn.Identity(),
                )
            )

        self.mlp.append(nn.Linear(dim, n_heads))

        self.div = divide

    def mod_mask(
        self,
        mask: Optional[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_q_k: Optional[torch.Tensor] = None,
        pos_q: Optional[torch.Tensor] = None,
        pos_k: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Modify mask

        Parameters
        ----------
        mask : Optional[torch.Tensor]
            (B, *, NH, L1, L2), can pass None for no pre-existing mask.
        q : torch.Tensor
            (B, *, L1, NH, H)
        k : torch.Tensor
            (B, *, L2, NH, H)
        v : torch.Tensor
            (B, *, L2, NH, H)
        pos_q_k : Optional[torch.Tensor], optional
            (B, *, L). Positions of q and k in self attention mode. Requires L1 = L2
            By default None for using pos_q and pos_k instead
        pos_q : Optional[torch.Tensor], optional
            (B, *, L1). Positions of q in cross attention mode
            Ignored if pos_q_k is specified
            By default None for computing positions from 0 to L1-1
        pos_k : Optional[torch.Tensor], optional
            (B, *, L2). Positions of k in cross attention mode
            Ignored if pos_q_k is specified
            By default None for computing positions from 0 to L2-1

        Returns
        -------
        torch.Tensor
            (B, *, NH, L1, L2)
        """
        if pos_q_k is not None:
            pos_q = pos_k = pos_q_k
        elif (pos_q is None) and (pos_k is None):
            pos_q = self.default_pos_x(q[..., 0, :]) if pos_q is None else pos_q
            pos_k = self.default_pos_x(k[..., 0, :]) if pos_k is None else pos_k

        bias = self.apply_pos_division(pos_q[..., :, None] - pos_k[..., None, :])[
            ..., None
        ].to(q)

        for layer in self.mlp:
            bias = layer(bias)
        bias = bias.transpose(-1, -2).transpose(-2, -3).to(q)

        if mask is not None:
            l1, l2 = q.shape[-3], k.shape[-3]
            mask_l1, mask_l2 = mask.shape[-2], mask.shape[-1]
            if (l1 != mask_l1) or (l2 != mask_l2):
                mask = F.pad(mask, (l2 - mask_l2, 0, l1 - mask_l1, 0))

        return (0 if mask is None else mask) - bias


class XL(Base):
    """Relative positional biases as in Transformer-XL (Dai et al 2019)
    Supports specifying positions and division of positional range

    Parameters
    ----------
    dim : int
        number of hidden of x (total, not per head)
    n_heads : int
        Number of heads in the model
    divide : float, optional
        divide positions by this factor, useful for large (or small) numerical ranges, by default 1.0
    learned_div : bool, optional
        Whether to learn the division factor, if True, div value initialization is as provided by divide argument, by default False
    """

    def __init__(
        self, dim: int, n_heads: int, divide: float = 1.0, learned_div: bool = False
    ):
        super().__init__(divide=divide, learned=learned_div)
        self.embed_lin = nn.Linear(dim, dim)
        self.bias_r_w = self._init_bias(
            nn.Parameter(torch.Tensor(n_heads, dim // n_heads))
        )
        self.bias_r_r = self._init_bias(
            nn.Parameter(torch.Tensor(n_heads, dim // n_heads))
        )
        self.n_heads = n_heads
        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.div = divide

    def mod_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify Q, K and V

        Parameters
        ----------
        q : torch.Tensor
            (B, *, L1, NH, H)
        k : torch.Tensor
            (B, *, L2, NH, H)
        v : torch.Tensor
            (B, *, L2, NH, H)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            q, k, v [(B, *, L1, NH, H), (B, *, L2, NH, H), (B, *, L2, NH, H)]
        """
        return q + self.bias_r_w, k, v

    def mod_mask(
        self,
        mask: Optional[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_q_k: Optional[torch.Tensor] = None,
        pos_q: Optional[torch.Tensor] = None,
        pos_k: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Modify mask

        Parameters
        ----------
        mask : Optional[torch.Tensor]
            (B, *, NH, L1, L2), can pass None for no pre-existing mask.
        q : torch.Tensor
            (B, *, L1, NH, H)
        k : torch.Tensor
            (B, *, L2, NH, H)
        v : torch.Tensor
            (B, *, L2, NH, H)
        pos_q_k : Optional[torch.Tensor], optional
            (B, *, L). Positions of q and k in self attention mode. Requires L1 = L2
            By default None for using pos_q and pos_k instead
        pos_q : Optional[torch.Tensor], optional
            (B, *, L1). Positions of q in cross attention mode
            Ignored if pos_q_k is specified
            By default None for computing positions from 0 to L1-1
        pos_k : Optional[torch.Tensor], optional
            (B, *, L2). Positions of k in cross attention mode
            Ignored if pos_q_k is specified
            By default None for computing positions from 0 to L2-1

        Returns
        -------
        torch.Tensor
            (B, *, NH, L1, L2)
        """
        if pos_q_k is not None:
            pos_q = pos_k = pos_q_k
        elif (pos_q is None) and (pos_k is None):
            pos_q = self.default_pos_x(q[..., 0, :]) if pos_q is None else pos_q
            pos_k = self.default_pos_x(k[..., 0, :]) if pos_k is None else pos_k

        relative_pos = (pos_q[..., :, None] - pos_k[..., None, :]).to(
            self.inv_freq.dtype
        )

        sinusoid_inp = self.apply_pos_division(relative_pos[..., None] * self.inv_freq)
        pos_emb = torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

        pos_emb = rearrange(pos_emb, "... d j -> ... (d j)")

        r = rearrange(
            self.embed_lin(pos_emb),
            "... (nh h) -> ... nh h",
            nh=self.n_heads,
            h=q.shape[-1],
        )

        l1, l2 = q.shape[-3], k.shape[-3]
        r_l1, r_l2 = r.shape[-4], r.shape[-3]
        if (l1 != r_l1) or (l2 != r_l2):
            r = F.pad(r, (0, 0, 0, 0, l2 - r_l2, 0, l1 - r_l1, 0))

        q_r = q + self.bias_r_w
        BD = einsum(q_r, r, "... q n h, ... q k n h -> ... n q k")

        if mask is not None:
            l1, l2 = q.shape[-3], k.shape[-3]
            mask_l1, mask_l2 = mask.shape[-2], mask.shape[-1]
            if (l1 != mask_l1) or (l2 != mask_l2):
                mask = F.pad(mask, (l2 - mask_l2, 0, l1 - mask_l1, 0))

        return (0 if mask is None else mask) + BD

    def _init_bias(self, bias):
        bound = 1 / bias.size(1) ** 0.5
        return nn.init.uniform_(bias, -bound, bound)
