import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F
import math
from bio_attention import positional


def compl_mod(m, n):
    return int(n * math.ceil(m / n) - m)


class Attention(torch.nn.Module):
    def __init__(
        self,
        dropout=0.0,
        enable_math=True,
        enable_flash=True,
        enable_mem_efficient=True,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.context_manager = {
            "enable_math": enable_math,
            "enable_flash": enable_flash,
            "enable_mem_efficient": enable_mem_efficient,
        }
        self.use_context_manager = not all(
            [enable_math, enable_flash, enable_mem_efficient]
        )

    def forward(self, q, k, v, mask=None, causal=False):
        """Default attention layer

        Parameters
        ----------
        q : B, *, L1, NH, H
            queries
        k : B, *, L2, NH, H
            keys
        v : B, *, L2, NH, H
            values
        mask : B, *, NH, L1, L2, optional
            _description_, by default None
        causal : bool, optional
            whether to do causal attention or not, by default False

        Returns
        -------
        _type_
            _description_
        """
        q_shape = q.shape
        q, k, v = map(
            lambda t: rearrange(t, "b ... x n h -> (b ...) x n h").permute(0, 2, 1, 3),
            (q, k, v),
        ) # (B...), NH, L, H
        if mask is not None:
            mask = rearrange(mask, "b ... n q k -> (b ...) n q k") # (B...), NH, L1, L2

        if causal:
            causal_mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool)
            causal_mask = causal_mask.triu(diagonal=0).expand(q.shape[0], q.shape[1], -1, -1)
            mask = (mask if mask is not None else (causal_mask).to(q)).masked_fill(causal_mask, -float('inf'))

        if self.use_context_manager:
            with torch.backends.cuda.sdp_kernel(**self.context_manager):
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
        else:
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


class RandomAttention(nn.Module):
    def __init__(
        self,
        n_random_keys=64,
        dropout=0.2,
        materialize_full=False,
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

    def forward_indexed(self, q, k, v, mask=None, causal=False):
        """random attention layer

        NOTE: causal attention will erase input masks

        Parameters
        ----------
        q : B, *, L1, NH, H
            queries
        k : B, *, L2, NH, H
            keys
        v : B, *, L2, NH, H
            values
        mask : B, *, NH, L1, L2, optional
            _description_, by default None
        causal : bool, optional
            whether to do causal attention or not, by default False

        Returns
        -------
        _type_
            _description_
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

    def forward_naive(self, q, k, v, mask=None, causal=False):
        """NOTE: Is incompatible with causal attention.
        NOTE: Is incompatible with input masks attention.

        Args:
            q (_type_): _description_
            k (_type_): _description_
            v (_type_): _description_
            mask (_type_, optional): _description_. Defaults to None.
            causal (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
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
    def __init__(
        self,
        window=5,
        dropout=0.1,
        materialize_full=False,
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

    def forward_sliced(self, q, k, v, mask=None, causal=False):
        """windowed attention layer

        Parameters
        ----------
        q : B, L1, NH, H
            queries
        k : B, L2, NH, H
            keys
        v : B, L2, NH, H
            values
        mask : B, NH, L1, L2, optional
            _description_, by default None
        causal : bool, optional
            whether to do causal attention or not, by default False

        Returns
        -------
        _type_
            _description_

        NOTE: incompatible with user-defined masks
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

    def forward_naive(self, q, k, v, mask=None, causal=False):
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
    def __init__(self, dim, attn, nh=4, plugin=None):
        super().__init__()
        assert dim % nh == 0, "dim should be divisible by number of heads"

        self.lin = nn.Linear(dim, 3 * dim)
        self.attn = attn
        self.nh = nh

        self.plugin = plugin if plugin is not None else positional.Base()

    def forward(self, x, pos=None, mask=None, causal=False, **mod_kwargs):
        """
        MASK = is used by mod_x - type positional embeddings if it is of bool type and shape B, *, L
        MASK can be (B,*,L), (B,*,L,L), or (B, * NH, L, L).
        x = (B, *, L, H)
        """
        if (mask is not None) and (mask.dtype == torch.bool):
            mask = F.pad(mask,(x.shape[-2] - mask.shape[-1], 0), value = True)

        use_mask_to_mod_x = False
        if mask is not None:
            use_mask_to_mod_x = (mask.dtype == torch.bool) and (mask.shape == x.shape[:-1])

        x = self.plugin.mod_x(x, pos=pos, mask = (mask if use_mask_to_mod_x else None), **mod_kwargs)
        q, k, v = torch.split(self.lin(x), x.size(-1), dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "... (n h) -> ... n h", n=self.nh), (q, k, v)
        ) # B, *, L, NH, H
        if mask is not None:
            if q.ndim - mask.ndim == 2: #B, *, L -> B, *, L, L
                mask = repeat(mask, '... l -> ... (l2) l', l2=mask.shape[-1])
                
            if q.ndim - mask.ndim == 1: #B, *, L, L  -> B, *, NH, L, L
                mask = repeat(mask, '... l1 l2 -> ... nh l1 l2', nh=q.shape[-2])
            
            assert mask.shape[:-3] == q.shape[:-3]
            if mask.dtype == torch.bool:
                mask = (~mask).to(q).masked_fill(~mask, -float('inf'))

            mask = F.pad(mask, (k.shape[-3] - mask.shape[-1], 0, (q.shape[-3] - mask.shape[-2]), 0), value = 0)

        mod_kwargs |= {"pos_q_k": pos}

        q, k, v = self.plugin.mod_qkv(q, k, v, **mod_kwargs)

        mask = self.plugin.mod_mask(mask, q, k, v, **mod_kwargs)

        return rearrange(
            self.attn(q, k, v, mask=mask, causal=causal),
            "... n h -> ... (n h)",
            n=self.nh,
        )

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


class TransformerLayer(nn.Module):
    def __init__(
        self, attn, dim, nh, plugin=None, dropout=0.2, glu_ff=True, activation="swish"
    ):
        super().__init__()
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "swish":
            act = nn.SiLU()

        self.norm = nn.LayerNorm(dim)
        self.attn = AttnLayer(dim, attn, nh=nh, plugin=plugin)

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

    def forward(self, x, pos=None, mask=None, causal=False, **mod_kwargs):
        x = self.attn(self.norm(x), pos=pos, mask=mask, causal=causal, **mod_kwargs) + x
        return self.ff(x)


class Transformer(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        nh,
        attentiontype="vanilla",
        attention_args={},
        plugintype="none",
        plugin_args={},
        only_apply_plugin_at_first=False,
        dropout=0.2,
        glu_ff=True,
        activation="swish",
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

    def forward(self, x, pos=None, mask=None, causal=False, **mod_kwargs):
        for layer in self.layers:
            x = layer(x, pos=pos, mask=mask, causal=causal, **mod_kwargs)
        return x

class TransformerEncoder(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, pos=None, mask=None, **mod_kwargs):
        for layer in self.layers:
            x = layer(x, pos=pos, mask=mask, causal=False, **mod_kwargs)
        return x
    
class TransformerDecoder(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, pos=None, mask=None, **mod_kwargs):
        for layer in self.layers:
            x = layer(x, pos=pos, mask=mask, causal=True, **mod_kwargs)
        return x


class Aggregator(nn.Module):
    def __init__(self, method = "max"):
        super().__init__()
        assert method in ["mean", "max", "cls"]
        self.method = method

    def forward(self, x, mask = None):
        """
        X = B, *, L, H
        mask = B, *, L
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        if self.method == "mean":
            return x.sum(-2) / (mask.sum(-1, keepdim=True) if mask is not None else x.shape[-2])
        elif self.method == "max":
            return x.max(-2).values
        elif self.method == "cls":
            return x[..., 0, :]