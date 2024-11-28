import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F
import math
from bio_attention import positional
from typing import Optional, Literal, Union
from torch.nn.attention import sdpa_kernel, SDPBackend

def compl_mod(m, n):
    return int(n * math.ceil(m / n) - m)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class GLU(nn.Module):
    def __init__(self, dim, ff_dim, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim, ff_dim * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class Attention(nn.Module):
    """Scaled-dot product attention operator.
    For more information on kernels: see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Parameters
    ----------
    dropout : float, optional
        dropout rate in self attention matrix, by default 0.0
    enable_math : bool, optional
        allow PyTorch C++ implementation, by default True
    enable_flash : bool, optional
        allow FlashAttention implementation, by default True
    enable_mem_efficient : bool, optional
        allow Memory-Efficient implementation, by default True
    """

    def __init__(
        self,
        dropout: float = 0.0,
        enable_math: bool = True,
        enable_flash: bool = True,
        enable_mem_efficient: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.context_manager = []
        if enable_math:
            self.context_manager.append(SDPBackend.MATH)
        if enable_flash:
            self.context_manager.append(SDPBackend.FLASH_ATTENTION)
        if enable_mem_efficient:
            self.context_manager.append(SDPBackend.EFFICIENT_ATTENTION)

        self.use_context_manager = not all(
            [enable_math, enable_flash, enable_mem_efficient]
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        q : torch.Tensor
            (B, *, L1, NH, H)
        k : torch.Tensor
            (B, *, L2, NH, H)
        v : torch.Tensor
            (B, *, L2, NH, H)
        mask : Optional[torch.Tensor], optional
            (B, *, NH, L1, L2), by default None
        causal : bool, optional
            Perform causal attention. Unlike default pytorch implementation, both a mask and causal can be used jointly, by default False

        Returns
        -------
        torch.Tensor
            (B, *, L1, NH, H)
        """
        q_shape = q.shape
        q, k, v = map(
            lambda t: rearrange(t, "b ... x n h -> (b ...) x n h").permute(0, 2, 1, 3),
            (q, k, v),
        )  # (B...), NH, L, H
        if mask is not None:
            mask = rearrange(mask, "b ... n q k -> (b ...) n q k")  # (B...), NH, L1, L2

        is_causal = False
        if causal and (mask is not None):
            causal_mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device)
            causal_mask = causal_mask.triu(diagonal=0).expand(
                q.shape[0], q.shape[1], -1, -1
            )
            mask = mask.masked_fill(
                causal_mask, -float("inf")
            )
        elif causal:
            is_causal = True

        if self.use_context_manager:
            with sdpa_kernel(self.context_manager):
                return (
                    F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=(self.dropout if self.training else 0),
                        attn_mask=mask,
                        is_causal=is_causal,
                    )
                    .permute(0, 2, 1, 3)
                    .view(*q_shape)
                )
        else:
            return (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=(self.dropout if self.training else 0),
                    attn_mask=mask,
                    is_causal=is_causal,
                )
                .permute(0, 2, 1, 3)
                .view(*q_shape)
            )


class RandomAttention(nn.Module):
    """Scaled-dot product attention operator that only randomly attends on a number of keys per query.
    Supports two versions: one that materializes the full matrix and, hence, scales quadratically with sequence length.
    In essence, this is default attention with random masks.
    The other version is memory efficient, scaling linearly with sequence length, but has a lower base efficiency because of the extra steps taken.

    Parameters
    ----------
    n_random_keys : int, optional
        number of keys every query should attend to, by default 64
    dropout : float, optional
        dropout rate in self attention matrix, by default 0.0
    materialize_full : bool, optional
        whether to materialize full attention matrix, by default False
    """

    def __init__(
        self,
        n_random_keys: int = 64,
        dropout: float = 0.0,
        materialize_full: bool = False,
        **kwargs,
    ):
        super().__init__()
        if not materialize_full:
            self.softmax = nn.Softmax(dim=-2)
            self.dropout = nn.Dropout(dropout)

            self.forward = self.forward_indexed
        else:
            self.dropout = dropout
            self.forward = self.forward_naive
        self.n = n_random_keys

    def forward_indexed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Memory-efficient forward pass
        NOTE: for the moment, causal attention with input masks are not implemented.

        Parameters
        ----------
        q : torch.Tensor
            (B, *, L1, NH, H)
        k : torch.Tensor
            (B, *, L2, NH, H)
        v : torch.Tensor
            (B, *, L2, NH, H)
        mask : Optional[torch.Tensor], optional
            (B, *, NH, L1, L2), by default None
        causal : bool, optional
            Perform causal attention, by default False

        Returns
        -------
        torch.Tensor
            (B, *, L1, NH, H)
        """
        assert not (causal and (mask is not None))

        q_shape = q.shape
        q, k, v = map(lambda t: rearrange(t, "b ... x n h -> (b ...) x n h"), (q, k, v))
        if mask is not None:
            mask = rearrange(mask, "b ... n q k -> (b ...) n q k")

        b, s2, nh, h = k.shape
        s1 = q.shape[1]

        mask = (
            torch.ones(b, nh, s1, s2, dtype=torch.bool).tril(diagonal=0)
            if causal
            else mask
        )
        if mask is not None:
            mask = (
                (~mask).float().masked_fill(~mask, -float("inf"))
                if mask.dtype == torch.bool
                else mask
            )
            mask = mask.to(q)

        q = q * (h**-0.5)

        indices_select = torch.randint(0, s1, (b, s2, self.n)).to(q.device)
        indexer = torch.arange(b).view(b, 1, 1)
        if mask is not None:
            mask = mask.permute(0, 2, 3, 1)[
                indexer, torch.arange(s1)[None, :, None], indices_select
            ].permute(0, 3, 1, 2)

        k = k[indexer, indices_select]
        v = v[indexer, indices_select]

        A = einsum("b q n h, b q k n h -> b q k n", q, k) + (
            0 if mask is None else mask.permute(0, 2, 3, 1)
        )
        A = self.softmax(A)

        A = self.dropout(A)
        z = einsum("b q k n, b q k n h -> b q n h", A, v)

        return z.view(*q_shape)

    def forward_naive(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Naive (random masking) forward pass
        NOTE: for the moment, is incompatible with input masks

        Parameters
        ----------
        q : torch.Tensor
            (B, *, L1, NH, H)
        k : torch.Tensor
            (B, *, L2, NH, H)
        v : torch.Tensor
            (B, *, L2, NH, H)
        mask : Optional[torch.Tensor], optional
            (B, *, NH, L1, L2), by default None
        causal : bool, optional
            Perform causal attention, by default False

        Returns
        -------
        torch.Tensor
            (B, *, L1, NH, H)
        """
        q_shape = q.shape
        q, k, v = map(
            lambda t: rearrange(t, "b ... x n h -> (b ...) x n h").permute(0, 2, 1, 3),
            (q, k, v),
        )
        if mask is not None:
            mask = rearrange(mask, "b ... n q k -> (b ...) n q k")

        b, nh, s2, h = k.shape
        s1 = q.shape[-2]

        mask = (
            (torch.rand(b, 1, s1, s2) < (self.n / s2))
            .expand(-1, nh, -1, -1)
            .to(k.device)
        )

        return (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=(self.dropout if self.training else 0),
                attn_mask=mask,
                is_causal=causal,
            )
            .permute(0, 2, 1, 3)
            .view(*q_shape)
        )


class WindowAttention(nn.Module):
    """Scaled-dot product attention operator that only attends on a local window of keys per query.
    Supports two versions: one that materializes the full matrix and, hence, scales quadratically with sequence length.
    In essence, this is default attention with a mask.
    The other version is memory efficient, scaling linearly with sequence length, but has a lower base efficiency because of the extra steps taken.

    Parameters
    ----------
    window : int, optional
        Window size, analogous to kernel size in convolutions, should be odd, by default 15
    dropout : float, optional
        dropout rate in self attention matrix, by default 0.0
    materialize_full : bool, optional
        whether to materialize full attention matrix, by default False
    """

    def __init__(
        self,
        window: int = 15,
        dropout: float = 0.0,
        materialize_full: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert window % 2 == 1, "Window size should be an odd integer."

        self.w = int((window - 1) / 2)

        if not materialize_full:
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)
            self.k_ch = window * 2
            self.q_ch = window + 1

            u = torch.triu(torch.full((self.q_ch, self.k_ch), True))
            self.mask = ~torch.logical_and(u, torch.flip(u, [0, 1]))
            self.mask_k_left = torch.clone(self.mask)
            self.mask_k_left[:, : self.w] = True
            self.forward = self.forward_sliced

        else:
            self.dropout = dropout
            self.forward = self.forward_naive

    def forward_sliced(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Memory-efficient forward pass
        NOTE: q and k need to have the same sequence length L1 = L2
        NOTE: is incompatible with user-defined masks for the time being.

        Parameters
        ----------
        q : torch.Tensor
            (B, *, L1, NH, H)
        k : torch.Tensor
            (B, *, L2, NH, H)
        v : torch.Tensor
            (B, *, L2, NH, H)
        mask : Optional[torch.Tensor], optional
            (B, *, NH, L1, L2), by default None
        causal : bool, optional
            Perform causal attention, by default False

        Returns
        -------
        torch.Tensor
            (B, *, L1, NH, H)
        """

        assert mask is None

        assert k.shape[-3] == q.shape[-3], "q and k should have same input length."

        q_shape = q.shape
        q, k, v = map(lambda t: rearrange(t, "b ... x n h -> (b ...) x n h"), (q, k, v))

        b, s, nh, h = k.shape

        q = q * (h**-0.5)

        q_pad = compl_mod(s, self.q_ch)
        k_pad = compl_mod((s + self.w * 2) - self.k_ch, self.q_ch)

        q = F.pad(q, (0,) * 5 + (q_pad,)).unfold(1, self.q_ch, self.q_ch)
        k = F.pad(k, (0,) * 4 + (self.w, self.w + k_pad)).unfold(
            1, self.k_ch, self.q_ch
        )
        v = F.pad(v, (0,) * 4 + (self.w, self.w + k_pad)).unfold(
            1, self.k_ch, self.q_ch
        )

        A = einsum("b c n h q, b c n h k -> b n c q k ", q, k)

        mask_value = -torch.finfo(A.dtype).max
        mask_k_right = torch.clone(self.mask.to(A.device))
        mask_k_right[:, -(self.w + k_pad) :] = True
        if q.shape[1] > 1:
            mask = torch.stack(
                [self.mask_k_left.to(A.device)]
                + [self.mask.to(A.device)] * (q.shape[1] - 2)
                + [mask_k_right]
            )
        else:
            mask = torch.logical_or(
                self.mask_k_left.to(A.device), mask_k_right
            ).unsqueeze(0)
        if causal:
            mask = ~(~mask).tril(diagonal=self.w)

        A.masked_fill_(mask, mask_value)
        A = self.softmax(A)
        A = self.dropout(A)

        z = einsum("b n c q k, b c n h k -> b n c q h", A, v)
        z = z.view(b, nh, -1, h)[:, :, :s].permute(0, 2, 1, 3).view(*q_shape)

        return z

    def forward_naive(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Naive (masking) forward pass
        NOTE: q and k need to have the same sequence length L1 = L2
        NOTE: is incompatible with user-defined masks for the time being.

        Parameters
        ----------
        q : torch.Tensor
            (B, *, L1, NH, H)
        k : torch.Tensor
            (B, *, L2, NH, H)
        v : torch.Tensor
            (B, *, L2, NH, H)
        mask : Optional[torch.Tensor], optional
            (B, *, NH, L1, L2), by default None
        causal : bool, optional
            Perform causal attention, by default False

        Returns
        -------
        torch.Tensor
            (B, *, L1, NH, H)
        """
        assert mask is None
        assert k.shape[-3] == q.shape[-3], "q and k should have same input length."

        q_shape = q.shape
        q, k, v = map(
            lambda t: rearrange(t, "b ... x n h -> (b ...) x n h").permute(0, 2, 1, 3),
            (q, k, v),
        )

        b, nh, s2, h = k.shape
        s1 = q.shape[-2]

        u = torch.triu(torch.full((s1, s2), True), diagonal=-self.w)
        mask = (
            torch.logical_and(u, torch.flip(u, [0, 1]))
            .expand(b, nh, -1, -1)
            .to(k.device)
        )
        if causal:
            mask = torch.tril(mask)
        return (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=(self.dropout if self.training else 0),
                attn_mask=mask,
                is_causal=False,
            )
            .permute(0, 2, 1, 3)
            .view(*q_shape)
        )


class AttnLayer(nn.Module):
    """Self-attention layer performing
    (1) projection to q, k and v.
    (2) attention.
    (3) collapsing heads back to same shape as x.
    (4) Final linear project to z.

    Parameters
    ----------
    dim : int
        input and output hidden dimension of x
    attn : Union[Attention, RandomAttention, WindowAttention]
        attention operator. Default, random and windowed attention are implemented.
    nh : int, optional
        number of heads, dim should be divisible by this number, by default 4
    plugin : Optional[positional.Base], optional
        positional bias plugin, for options see the list of implemented biases, by default None
    """

    def __init__(
        self,
        dim: int,
        attn: Union[Attention, RandomAttention, WindowAttention],
        nh: int = 4,
        plugin: Optional[positional.Base] = None,
    ):
        super().__init__()
        assert dim % nh == 0, "dim should be divisible by number of heads"

        self.lin = nn.Linear(dim, 3 * dim)
        self.attn = attn
        self.nh = nh
        self.to_out = nn.Linear(dim, dim)

        self.plugin = plugin if plugin is not None else positional.Base()

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        **mod_kwargs,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L, H)
        pos : Optional[torch.Tensor], optional
            (B, *, L) or (B, *, L-x), by default None
            If pos has a smaller sequence length than x, it is assumed x has extra tokens added in the beginning of its sequence such as CLS tokens.
            In this case, there is no position for these tokens. Positional biases will make sure those tokens do not partake in positional encoding.
        mask : Optional[torch.Tensor], optional
            By default None, but can be either:
            (1) (B, * L) or (B, * L-x). In this case, expects a boolean mask.
            This type of mask will be copied to (B, * NH, L, L) in a way such that no tokens can attend to positions indicated by False.
            This type of mask will extrapolate CLS tokens to not attend on positions indicated with False.
            (2) (B, * L, L) or (B, * L-x, L-x). In this case, can either be floating point or boolean mask.
            This type of mask will extrapolate CLS token to attend on all tokens.
            For this type of mask, the same mask is applied over all heads
            (3) (B, * NH, L, L) or (B, * NH, L-x, L-x)
            Ditto as previous case, but for this type of mask, different biases/masks can be applied per head.
        causal : bool, optional
            Perform causal attention, by default False

        Returns
        -------
        torch.Tensor
            (B, *, L, H)
        """
        q, k, v = torch.split(self.lin(x), x.size(-1), dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "... (n h) -> ... n h", n=self.nh), (q, k, v)
        )  # B, *, L, NH, H

        if mask is not None:
            if q.ndim - mask.ndim == 2:  # B, *, L -> B, *, L, L
                mask = repeat(mask, "... l -> ... (l2) l", l2=mask.shape[-1])
            if q.ndim - mask.ndim == 1:  # B, *, L, L  -> B, *, NH, L, L
                mask = repeat(mask, "... l1 l2 -> ... nh l1 l2", nh=q.shape[-2])

            assert mask.shape[:-3] == q.shape[:-3]
            if mask.dtype == torch.bool:
                mask = (~mask).to(q).masked_fill(~mask, -float("inf"))

            diff_k = k.shape[-3] - mask.shape[-1]
            diff_q = q.shape[-3] - mask.shape[-2]
            diff_k = (0 if diff_k < 0 else diff_k)
            diff_q = (0 if diff_q < 0 else diff_q)
            mask = F.pad(
                mask,
                (diff_k, 0, diff_q, 0),
                value=0,
            )

        mod_kwargs |= {"pos_q_k": pos}

        q, k, v = self.plugin.mod_qkv(q, k, v, **mod_kwargs)

        mask = self.plugin.mod_mask(mask, q, k, v, **mod_kwargs)

        return self.to_out(
            rearrange(
                self.attn(q, k, v, mask=mask, causal=causal),
                "... n h -> ... (n h)",
                n=self.nh,
            )
        )


class CrossAttnLayer(nn.Module):
    """Cross-attention layer performing
    (1) projection to q, k and v.
    (2) attention.
    (3) collapsing heads back to same shape as x.
    (4) Final linear project to z.

    Parameters
    ----------
    dim : int
        input and output hidden dimension of x
    attn : Union[Attention, RandomAttention, WindowAttention]
        attention operator. Default, random and windowed attention are implemented.
    nh : int, optional
        number of heads, dim should be divisible by this number, by default 4
    plugin : Optional[positional.Base], optional
        positional bias plugin, Can only be None. Here for API compatibility reasons.

    """

    def __init__(
        self,
        dim: int,
        attn: Union[Attention, RandomAttention, WindowAttention],
        nh: int = 4,
        plugin: Optional[positional.Base] = None,
    ):
        super().__init__()
        assert dim % nh == 0, "dim should be divisible by number of heads"

        self.q_lin = nn.Linear(dim, dim)
        self.kv_lin = nn.Linear(dim, 2 * dim)
        self.attn = attn
        self.nh = nh
        self.to_out = nn.Linear(dim, dim)

        assert plugin is None
        self.plugin = plugin if plugin is not None else positional.Base()
        

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        **mod_kwargs,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L, H)
        context : torch.Tensor
            (B, *, L2, H)
        pos : Optional[torch.Tensor], optional
            (B, *, L) or (B, *, L-x), by default None
            If pos has a smaller sequence length than x, it is assumed x has extra tokens added in the beginning of its sequence such as CLS tokens.
            In this case, there is no position for these tokens. Positional biases will make sure those tokens do not partake in positional encoding.
        mask : Optional[torch.Tensor], optional
            By default None, but can be either:
            (1) (B, * L2) or (B, * L2-x). In this case, expects a boolean mask.
            This type of mask will be copied to (B, * NH, L, L2) in a way such that no query tokens can attend to key positions indicated by False.
            This type of mask will extrapolate CLS tokens to not attend on positions indicated with False.
            (2) (B, * L, L2) or (B, * L-x, L2-x). In this case, can either be floating point or boolean mask.
            This type of mask will extrapolate CLS token to attend on all tokens.
            For this type of mask, the same mask is applied over all heads
            (3) (B, * NH, L, L2) or (B, * NH, L-x, L2-x)
            Ditto as previous case, but for this type of mask, different biases/masks can be applied per head.
        causal : bool, optional
            should be False, only here for API compatibility reasons.

        Returns
        -------
        torch.Tensor
            (B, *, L, H)
        """
        assert causal is False
        assert pos is None

        q = self.q_lin(x)
        k, v = torch.split(self.kv_lin(context), context.size(-1), dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "... (n h) -> ... n h", n=self.nh), (q, k, v)
        )  # B, *, L, NH, H

        if mask is not None:
            if q.ndim - mask.ndim == 2:  # B, *, L2 -> B, *, L, L2
                mask = repeat(mask, "... l2 -> ... (l) l2", l=q.shape[-3])
            if q.ndim - mask.ndim == 1:  # B, *, L, L2  -> B, *, NH, L, L2
                mask = repeat(mask, "... l1 l2 -> ... nh l1 l2", nh=q.shape[-2])

            assert mask.shape[:-3] == q.shape[:-3]
            if mask.dtype == torch.bool:
                mask = (~mask).to(q).masked_fill(~mask, -float("inf"))

            diff_k = k.shape[-3] - mask.shape[-1]
            diff_q = q.shape[-3] - mask.shape[-2]
            diff_k = (0 if diff_k < 0 else diff_k)
            diff_q = (0 if diff_q < 0 else diff_q)
            mask = F.pad(
                mask,
                (diff_k, 0, diff_q, 0),
                value=0,
            )

        return self.to_out(
            rearrange(
                self.attn(q, k, v, mask=mask, causal=False),
                "... n h -> ... (n h)",
                n=self.nh,
            )
        )


class TransformerLayer(nn.Module):
    """Transformer Layer.
    Performs pre-norm, attention, residual around both, then layernorm + feedforward with residual connection wrapped around it.

    Parameters
    ----------
    attn : Union[Attention, RandomAttention, WindowAttention]
        attention operator. Default, random and windowed attention are implemented.
    dim : int
        input and output hidden dimension of x
    nh : int
        number of heads, dim should be divisible by this number, by default 4
    plugin : Optional[positional.Base], optional
        positional bias plugin, for options see the list of implemented biases, by default None
    dropout : float, optional
        dropout in feedforward, by default 0.2
    glu_ff : bool, optional
        whether to use gated linear feedforward network, by default True
    activation : Literal["relu", "gelu", "swish"], optional
        activation, by default "swish"
    """

    def __init__(
        self,
        attn: Union[Attention, RandomAttention, WindowAttention],
        dim: int,
        nh: int,
        plugin: Optional[positional.Base] = None,
        dropout: float = 0.2,
        glu_ff: bool = True,
        activation: Literal["relu", "gelu", "swish"] = "swish",
    ):
        super().__init__()
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "swish":
            act = nn.SiLU()

        self.norm = nn.LayerNorm(dim)

        self.plugin = plugin if plugin is not None else positional.Base()
        self.attn = AttnLayer(dim, attn, nh=nh, plugin=self.plugin)

        project_in = (
            nn.Sequential(nn.Linear(dim, 4 * dim), act)
            if not glu_ff
            else GLU(dim, 4 * dim, act)
        )
        self.ff = Residual(
            nn.Sequential(
                nn.LayerNorm(dim),
                project_in,
                nn.Dropout(dropout),
                nn.Linear(4 * dim, dim),
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        **mod_kwargs,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L, H)
        pos : Optional[torch.Tensor], optional
            (B, *, L) or (B, *, L-x), by default None
            If pos has a smaller sequence length than x, it is assumed x has extra tokens added in the beginning of its sequence such as CLS tokens.
            In this case, there is no position for these tokens. Positional biases will make sure those tokens do not partake in positional encoding.
        mask : Optional[torch.Tensor], optional
            By default None, but can be either:
            (1) (B, * L) or (B, * L-x). In this case, expects a boolean mask.
            This type of mask will be copied to (B, * NH, L, L) in a way such that no tokens can attend to positions indicated by False.
            This type of mask will extrapolate CLS tokens to not attend on positions indicated with False.
            (2) (B, * L, L) or (B, * L-x, L-x). In this case, can either be floating point or boolean mask.
            This type of mask will extrapolate CLS token to attend on all tokens.
            For this type of mask, the same mask is applied over all heads
            (3) (B, * NH, L, L) or (B, * NH, L-x, L-x)
            Ditto as previous case, but for this type of mask, different biases/masks can be applied per head.
        causal : bool, optional
            Perform causal attention, by default False

        Returns
        -------
        torch.Tensor
            (B, *, L, H)
        """
        use_mask_to_mod_x = False
        if (mask is not None) and (mask.ndim == x.ndim - 1):
            assert mask.dtype == torch.bool
            diff = x.shape[-2] - mask.shape[-1]
            diff = (0 if diff < 0 else diff)
            mask = F.pad(mask, (diff, 0), value=True)
            use_mask_to_mod_x = True

        x = self.plugin.mod_x(
            x, pos=pos, mask=(mask if use_mask_to_mod_x else None), **mod_kwargs
        )

        x = self.attn(self.norm(x), pos=pos, mask=mask, causal=causal, **mod_kwargs) + x
        return self.ff(x)


class CrossTransformerLayer(nn.Module):
    """Cross-attention Transformer Layer.
    Performs pre-norm, attention, residual around both, then layernorm + feedforward with residual connection wrapped around it.

    Parameters
    ----------
    attn : Union[Attention, RandomAttention, WindowAttention]
        attention operator. Default, random and windowed attention are implemented.
    dim : int
        input and output hidden dimension of x
    nh : int
        number of heads, dim should be divisible by this number, by default 4
    plugin : Optional[positional.Base], optional
        positional bias plugin, for options see the list of implemented biases, by default None
    dropout : float, optional
        dropout in feedforward, by default 0.2
    glu_ff : bool, optional
        whether to use gated linear feedforward network, by default True
    activation : Literal["relu", "gelu", "swish"], optional
        activation, by default "swish"
    """

    def __init__(
        self,
        attn: Union[Attention, RandomAttention, WindowAttention],
        dim: int,
        nh: int,
        plugin: Optional[positional.Base] = None,
        dropout: float = 0.2,
        glu_ff: bool = True,
        activation: Literal["relu", "gelu", "swish"] = "swish",
    ):
        super().__init__()
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "swish":
            act = nn.SiLU()

        self.norm_1 = nn.LayerNorm(dim)
        self.plugin = plugin if plugin is not None else positional.Base()
        self.self_attn = AttnLayer(dim, attn, nh=nh, plugin=self.plugin)

        self.norm_2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttnLayer(dim, attn, nh=nh, plugin=None)

        project_in = (
            nn.Sequential(nn.Linear(dim, 4 * dim), act)
            if not glu_ff
            else GLU(dim, 4 * dim, act)
        )
        self.ff = Residual(
            nn.Sequential(
                nn.LayerNorm(dim),
                project_in,
                nn.Dropout(dropout),
                nn.Linear(4 * dim, dim),
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask_context: Optional[torch.Tensor] = None,
        mask_x: Optional[torch.Tensor] = None,
        causal: bool = False,
        **mod_kwargs,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L, H)
        context : torch.Tensor
            (B, *, L, H)
        pos : Optional[torch.Tensor], optional
            (B, *, L) or (B, *, L-x), by default None
            If pos has a smaller sequence length than x, it is assumed x has extra tokens added in the beginning of its sequence such as CLS tokens.
            In this case, there is no position for these tokens. Positional biases will make sure those tokens do not partake in positional encoding.
        mask : Optional[torch.Tensor], optional
            By default None, but can be either:
            (1) (B, * L) or (B, * L-x). In this case, expects a boolean mask.
            This type of mask will be copied to (B, * NH, L, L) in a way such that no tokens can attend to positions indicated by False.
            This type of mask will extrapolate CLS tokens to not attend on positions indicated with False.
            (2) (B, * L, L) or (B, * L-x, L-x). In this case, can either be floating point or boolean mask.
            This type of mask will extrapolate CLS token to attend on all tokens.
            For this type of mask, the same mask is applied over all heads
            (3) (B, * NH, L, L) or (B, * NH, L-x, L-x)
            Ditto as previous case, but for this type of mask, different biases/masks can be applied per head.
        causal : bool, optional
            Perform causal attention, by default False

        Returns
        -------
        torch.Tensor
            (B, *, L, H)
        """
        use_mask_to_mod_x = False
        if (mask_x is not None) and (mask_x.ndim == x.ndim - 1):
            assert mask_x.dtype == torch.bool
            diff = x.shape[-2] - mask_x.shape[-1]
            diff = (0 if diff < 0 else diff)
            mask_x = F.pad(mask_x, (diff, 0), value=True)
            use_mask_to_mod_x = True

        x = self.plugin.mod_x(
            x, pos=pos, mask=(mask_x if use_mask_to_mod_x else None), **mod_kwargs
        )

        x = self.self_attn(self.norm_1(x), pos=pos, mask=mask_x, causal=causal, **mod_kwargs) + x
        x = self.cross_attn(self.norm_2(x), context, pos=None, mask=mask_context, causal=False, **mod_kwargs) + x

        return self.ff(x)


class Transformer(nn.Module):
    """Transformer network chaining multiple transformer layers

    Parameters
    ----------
    depth : int
        number of transformer blocks to use
    dim : int
        input and output hidden dimension of x
    nh : int
        number of heads, dim should be divisible by this number
    attentiontype : Literal["vanilla", "random", "window"], optional
        attention operator, by default "vanilla"
    attention_args : dict, optional
        args passed to the attention operator init, by default {}
    plugintype : Literal["none", "sinusoidal", "learned", "learnedcont", "rotary", "ALiBi", "DPB", "XL"], optional
        positional bias plugin, by default "none"
    plugin_args : dict, optional
        arguments passed to positional bias init, by default {}
    only_apply_plugin_at_first : bool, optional
        only apply positional bias at the first layer, by default False
    dropout : float, optional
        dropout in feedforward layers. Take note that attention matrix dropout is controlled via attention_args, by default 0.2
    glu_ff : bool, optional
        whether to use gated linear feedforward network, by default True
    activation : Literal["relu", "gelu", "swish"], optional
        activation, by default "swish"
    """

    def __init__(
        self,
        depth: int,
        dim: int,
        nh: int,
        attentiontype: Literal["vanilla", "random", "window"] = "vanilla",
        attention_args: dict = {},
        plugintype: Literal[
            "none",
            "sinusoidal",
            "learned",
            "learnedcont",
            "rotary",
            "ALiBi",
            "DPB",
            "XL",
        ] = "none",
        plugin_args: dict = {},
        only_apply_plugin_at_first: bool = False,
        dropout: float = 0.2,
        glu_ff: bool = True,
        activation: Literal["relu", "gelu", "swish"] = "swish",
    ):
        super().__init__()

        assert attentiontype in ["vanilla", "random", "window"]

        if attentiontype == "vanilla":
            attn_op = Attention(**attention_args)
        elif attentiontype == "random":
            attn_op = RandomAttention(**attention_args)
        elif attentiontype == "window":
            attn_op = WindowAttention(**attention_args)

        if plugintype == "none":
            plugin = positional.Base()
        elif plugintype == "sinusoidal":
            plugin = positional.Sinusoidal(**plugin_args)
        elif plugintype == "learned":
            plugin = positional.LearnedVocab(**plugin_args)
        elif plugintype == "learnedcont":
            plugin = positional.LearnedContinuous(**plugin_args)
        elif plugintype == "rotary":
            plugin = positional.Rotary(**plugin_args)
        elif plugintype == "ALiBi":
            plugin = positional.ALiBi(**plugin_args)
        elif plugintype == "DPB":
            plugin = positional.DPB(**plugin_args)
        elif plugintype == "XL":
            plugin = positional.XL(**plugin_args)

        layers = []
        layers.append(
            TransformerLayer(
                attn_op,
                dim,
                nh,
                plugin=plugin,
                dropout=dropout,
                glu_ff=glu_ff,
                activation=activation,
            )
        )

        for _ in range(depth - 1):
            layers.append(
                TransformerLayer(
                    attn_op,
                    dim,
                    nh,
                    plugin=(None if only_apply_plugin_at_first else plugin),
                    dropout=dropout,
                    glu_ff=glu_ff,
                    activation=activation,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        **mod_kwargs,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L, H)
        pos : Optional[torch.Tensor], optional
            (B, *, L) or (B, *, L-x), by default None
            If pos has a smaller sequence length than x, it is assumed x has extra tokens added in the beginning of its sequence such as CLS tokens.
            In this case, there is no position for these tokens. Positional biases will make sure those tokens do not partake in positional encoding.
        mask : Optional[torch.Tensor], optional
            By default None, but can be either:
            (1) (B, * L) or (B, * L-x). In this case, expects a boolean mask.
            This type of mask will be copied to (B, * NH, L, L) in a way such that no tokens can attend to positions indicated by False.
            This type of mask will extrapolate CLS tokens to not attend on positions indicated with False.
            (2) (B, * L, L) or (B, * L-x, L-x). In this case, can either be floating point or boolean mask.
            This type of mask will extrapolate CLS token to attend on all tokens.
            For this type of mask, the same mask is applied over all heads
            (3) (B, * NH, L, L) or (B, * NH, L-x, L-x)
            Ditto as previous case, but for this type of mask, different biases/masks can be applied per head.
        causal : bool, optional
            Perform causal attention, by default False

        Returns
        -------
        torch.Tensor
            (B, *, L, H)
        """
        for layer in self.layers:
            x = layer(x, pos=pos, mask=mask, causal=causal, **mod_kwargs)
        return x


class TransformerEncoder(Transformer):
    """Transformer Encoder network chaining multiple transformer layers
    Equal to Transformer, only difference is that Causal = False is automatically decided in forward pass

    Parameters
    ----------
    depth : int
        number of transformer blocks to use
    dim : int
        input and output hidden dimension of x
    nh : int
        number of heads, dim should be divisible by this number
    attentiontype : Literal["vanilla", "random", "window"], optional
        attention operator, by default "vanilla"
    attention_args : dict, optional
        args passed to the attention operator init, by default {}
    plugintype : Literal["none", "sinusoidal", "learned", "learnedcont", "rotary", "ALiBi", "DPB", "XL"], optional
        positional bias plugin, by default "none"
    plugin_args : dict, optional
        arguments passed to positional bias init, by default {}
    only_apply_plugin_at_first : bool, optional
        only apply positional bias at the first layer, by default False
    dropout : float, optional
        dropout in feedforward layers. Take note that attention matrix dropout is controlled via attention_args, by default 0.2
    glu_ff : bool, optional
        whether to use gated linear feedforward network, by default True
    activation : Literal["relu", "gelu", "swish"], optional
        activation, by default "swish"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **mod_kwargs,
    ):
        for layer in self.layers:
            x = layer(x, pos=pos, mask=mask, causal=False, **mod_kwargs)
        return x


class TransformerDecoder(Transformer):
    """Transformer Decoder network chaining multiple transformer layers
    Equal to Transformer, only difference is that Causal = True is automatically decided in forward pass

    Parameters
    ----------
    depth : int
        number of transformer blocks to use
    dim : int
        input and output hidden dimension of x
    nh : int
        number of heads, dim should be divisible by this number
    attentiontype : Literal["vanilla", "random", "window"], optional
        attention operator, by default "vanilla"
    attention_args : dict, optional
        args passed to the attention operator init, by default {}
    plugintype : Literal["none", "sinusoidal", "learned", "learnedcont", "rotary", "ALiBi", "DPB", "XL"], optional
        positional bias plugin, by default "none"
    plugin_args : dict, optional
        arguments passed to positional bias init, by default {}
    only_apply_plugin_at_first : bool, optional
        only apply positional bias at the first layer, by default False
    dropout : float, optional
        dropout in feedforward layers. Take note that attention matrix dropout is controlled via attention_args, by default 0.2
    glu_ff : bool, optional
        whether to use gated linear feedforward network, by default True
    activation : Literal["relu", "gelu", "swish"], optional
        activation, by default "swish"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **mod_kwargs,
    ):
        for layer in self.layers:
            x = layer(x, pos=pos, mask=mask, causal=True, **mod_kwargs)
        return x


class TransformerSeq2Seq(nn.Module):
    """Transformer network chaining multiple transformer layers

    Parameters
    ----------
    depth : int
        number of transformer blocks to use
    dim : int
        input and output hidden dimension of x
    nh : int
        number of heads, dim should be divisible by this number
    attentiontype : Literal["vanilla", "random", "window"], optional
        attention operator, by default "vanilla"
    attention_args : dict, optional
        args passed to the attention operator init, by default {}
    plugintype : Literal["none", "sinusoidal", "learned", "learnedcont", "rotary", "ALiBi", "DPB", "XL"], optional
        positional bias plugin, by default "none"
    plugin_args : dict, optional
        arguments passed to positional bias init, by default {}
    only_apply_plugin_at_first : bool, optional
        only apply positional bias at the first layer, by default False
    dropout : float, optional
        dropout in feedforward layers. Take note that attention matrix dropout is controlled via attention_args, by default 0.2
    glu_ff : bool, optional
        whether to use gated linear feedforward network, by default True
    activation : Literal["relu", "gelu", "swish"], optional
        activation, by default "swish"
    """

    def __init__(
        self,
        enc_depth: int,
        dec_depth: int,
        dim: int,
        nh: int,
        enc_attentiontype: Literal["vanilla", "random", "window"] = "vanilla",
        enc_attention_args: dict = {},
        enc_plugintype: Literal[
            "none",
            "sinusoidal",
            "learned",
            "learnedcont",
            "rotary",
            "ALiBi",
            "DPB",
            "XL",
        ] = "none",
        enc_plugin_args: dict = {},
        enc_only_apply_plugin_at_first: bool = False,
        dec_attentiontype: Literal["vanilla", "random", "window"] = "vanilla",
        dec_attention_args: dict = {},
        dec_plugintype: Literal[
            "none",
            "sinusoidal",
            "learned",
            "learnedcont",
            "rotary",
            "ALiBi",
            "DPB",
            "XL",
        ] = "none",
        dec_plugin_args: dict = {},
        dec_only_apply_plugin_at_first: bool = False,
        dropout: float = 0.2,
        glu_ff: bool = True,
        activation: Literal["relu", "gelu", "swish"] = "swish",
    ):
        super().__init__()

        self.transformer_encoder = Transformer(
            depth=enc_depth,
            dim=dim,
            nh = nh,
            attentiontype=enc_attentiontype,
            attention_args=enc_attention_args,
            plugintype=enc_plugintype,
            plugin_args=enc_plugin_args,
            only_apply_plugin_at_first=enc_only_apply_plugin_at_first,
            dropout=dropout,
            glu_ff=glu_ff,
            activation=activation,
        )

        if dec_attentiontype == "vanilla":
            attn_op = Attention(**dec_attention_args)
        elif dec_attentiontype == "random":
            attn_op = RandomAttention(**dec_attention_args)
        elif dec_attentiontype == "window":
            attn_op = WindowAttention(**dec_attention_args)

        if dec_plugintype == "none":
            plugin = positional.Base()
        elif dec_plugintype == "sinusoidal":
            plugin = positional.Sinusoidal(**dec_plugin_args)
        elif dec_plugintype == "learned":
            plugin = positional.LearnedVocab(**dec_plugin_args)
        elif dec_plugintype == "learnedcont":
            plugin = positional.LearnedContinuous(**dec_plugin_args)
        elif dec_plugintype == "rotary":
            plugin = positional.Rotary(**dec_plugin_args)
        elif dec_plugintype == "ALiBi":
            plugin = positional.ALiBi(**dec_plugin_args)
        elif dec_plugintype == "DPB":
            plugin = positional.DPB(**dec_plugin_args)
        elif dec_plugintype == "XL":
            plugin = positional.XL(**dec_plugin_args)

        layers = []
        layers.append(
            TransformerLayer(
                attn_op,
                dim,
                nh,
                plugin=plugin,
                dropout=dropout,
                glu_ff=glu_ff,
                activation=activation,
            )
        )

        for _ in range(dec_depth - 1):
            layers.append(
                CrossTransformerLayer(
                    attn_op,
                    dim,
                    nh,
                    plugin=(None if dec_only_apply_plugin_at_first else plugin),
                    dropout=dropout,
                    glu_ff=glu_ff,
                    activation=activation,
                )
            )

        self.transformer_decoder = nn.ModuleList(layers)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_dec : torch.Tensor,
        pos_enc: Optional[torch.Tensor] = None,
        pos_dec : Optional[torch.Tensor] = None,
        mask_enc: Optional[torch.Tensor] = None,
        mask_dec: Optional[torch.Tensor] = None,
        causal_enc: bool = False,
        causal_dec: bool = True,
        **mod_kwargs,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x_{enc,dec} : torch.Tensor
            (B, *, L, H)
        pos_{enc,dec} : Optional[torch.Tensor], optional
            (B, *, L) or (B, *, L-x), by default None
            If pos has a smaller sequence length than x, it is assumed x has extra tokens added in the beginning of its sequence such as CLS tokens.
            In this case, there is no position for these tokens. Positional biases will make sure those tokens do not partake in positional encoding.
        mask_{enc,dec} : Optional[torch.Tensor], optional
            By default None, but can be either:
            (1) (B, * L) or (B, * L-x). In this case, expects a boolean mask.
            This type of mask will be copied to (B, * NH, L, L) in a way such that no tokens can attend to positions indicated by False.
            This type of mask will extrapolate CLS tokens to not attend on positions indicated with False.
            (2) (B, * L, L) or (B, * L-x, L-x). In this case, can either be floating point or boolean mask.
            This type of mask will extrapolate CLS token to attend on all tokens.
            For this type of mask, the same mask is applied over all heads
            (3) (B, * NH, L, L) or (B, * NH, L-x, L-x)
            Ditto as previous case, but for this type of mask, different biases/masks can be applied per head.
        causal_{enc,dec} : bool, optional
            Perform causal attention, by default False

        Returns
        -------
        torch.Tensor
            (B, *, L, H)
        """
        encoder_output = self.transformer_encoder(
            x_enc,
            pos=pos_enc,
            mask=mask_enc,
            causal=causal_enc,
            **mod_kwargs,
        )
        
        for layer in self.transformer_decoder:
            x_dec = layer(
                x_dec,
                context=encoder_output,
                pos=pos_dec,
                mask_x=mask_dec,
                mask_context=mask_enc,
                causal=causal_dec,
                **mod_kwargs)


        return x_dec


class Aggregator(nn.Module):
    """Aggregator module. Can be used to get a single vector from a sequence-valued input.

    Parameters
    ----------
    method : Literal["mean", "max", "cls"], optional
        aggregation method, by default "max"
    """

    def __init__(self, method: Literal["mean", "max", "cls"] = "max"):
        super().__init__()
        assert method in ["mean", "max", "cls"]
        self.method = method

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, L, H)
        mask : Optional[torch.Tensor], optional
            (B, *, L), by default None

        Returns
        -------
        torch.Tensor
            (B, *, H)
        """
        if mask is not None:
            assert mask.dtype == torch.bool
            x = x * mask.unsqueeze(-1)
        if self.method == "mean":
            return x.sum(-2) / (
                mask.sum(-1, keepdim=True) if mask is not None else x.shape[-2]
            )
        elif self.method == "max":
            return x.max(-2).values
        elif self.method == "cls":
            return x[..., 0, :]
