import torch
import torch.nn as nn
import einops as eo

from uplifting_rallies.helper import court_points, MAX_FPS
from uplifting_rallies.helper import (
    KEYPOINT_VISIBLE,
    BALL_VISIBLE,
    BALL_INVISIBLE,
)


def normalize_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    """Normalize attention masks to additive format: visible=0.0, hidden=-inf."""
    mask_min = mask.min().item()
    mask_max = mask.max().item()

    if mask_max == 1 and (mask_min == 0 or mask_min == 1):
        out_dtype = mask.dtype if torch.is_floating_point(mask) else torch.float32
        additive_mask = torch.full(
            mask.shape, float("-inf"), device=mask.device, dtype=out_dtype
        )
        additive_mask.masked_fill_(mask != 0, 0.0)
        return additive_mask

    if mask_max == 0 and mask_min < -1e8:
        return mask

    raise ValueError("wrong format for masks. Should be 0, 1 or -1e9, 0.")


def pack_visible_sequence(
    x: torch.Tensor,
    times: torch.Tensor,
    mask: torch.Tensor,
    visibilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack visible tokens to the front and mark padded tail with -inf in mask."""
    B, T, D = x.shape
    keep_mask = visibilities == BALL_VISIBLE
    sorted_keep_mask, sort_indices = torch.sort(
        keep_mask.int(), dim=1, descending=True, stable=True
    )
    sorted_keep_mask = sorted_keep_mask.bool()

    sort_indices_d = sort_indices.unsqueeze(-1).expand(B, T, D)
    x = torch.gather(x, dim=1, index=sort_indices_d)
    times = torch.gather(times, dim=1, index=sort_indices)
    mask = torch.gather(mask, dim=1, index=sort_indices)
    mask = mask.masked_fill(~sorted_keep_mask, float("-inf"))
    return x, times, mask


class LinearReLUMLP(nn.Module):
    """Reusable linear stack with ReLU between all layers except the last."""

    def __init__(self, dims: list[int]):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output dimensions")
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])]
        )
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        last_idx = len(self.layers) - 1
        for idx, layer in enumerate(self.layers):
            gain = 1 if idx == last_idx else 1.414
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self.relu(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, time_rotation):
        """
        Initialize the RotaryPositionalEmbedding class.

        Parameters:
        dim (int): Dimension of the input embeddings.
        time_rotation (str): Type of time rotation to apply ('old' or 'new').
        """
        super(RotaryPositionalEmbedding, self).__init__()
        self.dim = dim
        # Precompute sinusoidal frequencies
        self.inv_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)),
            requires_grad=False,
        )

        self.base_timestep = (
            1 / MAX_FPS
        )  # It describes the minimum timestep for one rotation.
        self.time_rotation = time_rotation

    def forward(self, x, times=None):
        """
        Apply rotary positional embeddings to the input tensor.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim).
        times (torch.Tensor | None): Tensor of shape (batch_size, seq_len) indicating
            the time of each token. Required when time_rotation is "new".

        Returns:
        torch.Tensor: Tensor with rotary positional embeddings applied.
        """
        B, num_heads, T, D = x.shape
        assert D == self.dim, "Input dimension does not match embedding dimension."

        # Calculate rotation angles using precomputed inverse frequencies
        if self.time_rotation == "new":
            pos = torch.round(times / self.base_timestep)  # Shape (B, T)
        elif self.time_rotation == "old":
            pos = torch.arange(T, device=x.device, dtype=x.dtype)  # Shape (T,)
            pos = eo.repeat(pos, "t -> b t", b=B)  # Shape (B, T)
        else:
            raise ValueError("Invalid time_rotation value. Use 'old' or 'new'.")
        freqs = torch.einsum("bi,j->bij", pos, self.inv_freq)  # (B, T, D/2)
        freqs = freqs.unsqueeze(1)  # Shape (B, 1, T, D/2)
        cos = torch.cos(freqs)  # Shape (B, 1, T, D/2)
        sin = torch.sin(freqs)  # Shape (B, 1, T, D/2)

        # Split x into uneven and even parts
        x_uneven = x[..., 0::2]
        x_even = x[..., 1::2]

        # Apply rotary embedding transformation
        x_rotated_uneven = x_uneven * cos - x_even * sin
        x_rotated_even = x_uneven * sin + x_even * cos

        # Interleave the rotated components back together
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_rotated_uneven
        x_rotated[..., 1::2] = x_rotated_even

        return x_rotated


class TableEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TableEmbedding, self).__init__()
        self.dim = embed_dim
        self.in_dim = 2  # u, v in image space
        self.expected_num_tokens = len(
            court_points
        )  # expected predefined tennis court positions
        self.proj = LinearReLUMLP([self.in_dim, self.dim, self.dim])

    def forward(self, x):
        """Embed table keypoints from (B, N, 2) to (B, N, embed_dim)."""
        _, N, D = x.shape
        assert D == self.in_dim, "Input dimension does not match embedding dimension."
        if N != self.expected_num_tokens:
            raise ValueError(
                f"Expected {self.expected_num_tokens} table tokens, got {N}."
            )
        return self.proj(x)


class BallEmbedding(nn.Module):
    def __init__(self, embed_dim, in_dim=2):
        super(BallEmbedding, self).__init__()
        self.dim = embed_dim
        self.in_dim = in_dim  # usually u, v in image space
        self.proj = LinearReLUMLP([self.in_dim, self.dim, self.dim])

    def forward(self, x):
        """Embed ball tokens from (B, T, in_dim) to (B, T, embed_dim)."""
        _, _, D = x.shape
        assert D == self.in_dim, "Input dimension does not match embedding dimension."
        return self.proj(x)


class AnkleVEmbedding(nn.Module):
    def __init__(self, embed_dim, in_dim=1):
        super(AnkleVEmbedding, self).__init__()
        self.dim = embed_dim
        self.in_dim = in_dim
        self.proj = LinearReLUMLP([self.in_dim, self.dim, self.dim])

    def forward(self, x):
        """Embed ankle height from (B, 1) to (B, embed_dim)."""
        _, D = x.shape
        assert D == self.in_dim, "Input dimension does not match embedding dimension."
        return self.proj(x)


class SpecialEmbedding(nn.Module):
    """
    Embedding for originalmethod mode if interpolation is used.
    Ball and table are embedded seperately, invisible ball positions are filled with a learnable token, then both is fused.
    """

    def __init__(self, embed_dim):
        super(SpecialEmbedding, self).__init__()
        self.dim = embed_dim
        self.expected_num_tokens = len(court_points)
        self.ball_embed = nn.Linear(2, embed_dim // 2)
        self.table_embed = nn.Linear(
            2 * self.expected_num_tokens, embed_dim // 2
        )  # predefined court points with u, v coordinates
        self.fusion_layer1 = nn.Linear(2 * (embed_dim // 2), embed_dim)
        self.relu = nn.ReLU()
        self.fusion_layer2 = nn.Linear(embed_dim, embed_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.ball_embed.weight, gain=1.414)
        nn.init.constant_(self.ball_embed.bias, 0)
        nn.init.xavier_uniform_(self.table_embed.weight, gain=1.414)
        nn.init.constant_(self.table_embed.bias, 0)
        nn.init.xavier_uniform_(self.fusion_layer1.weight, gain=1.414)
        nn.init.constant_(self.fusion_layer1.bias, 0)
        nn.init.xavier_uniform_(self.fusion_layer2.weight, gain=1)
        nn.init.constant_(self.fusion_layer2.bias, 0)

    def forward(self, ball_pos, table_pos, visibilities, missing_token):
        """First embedd ball and table positions separately, then exchange invisible ball positions with a learnable token,
            finally fuse both embeddings.
        Args:
            ball_pos (torch.Tensor): Tensor of shape (batch_size, seq_len, 2)
            table_pos (torch.Tensor): Tensor of shape (batch_size, num_positions, 2) with last coordinate (visibility) already removed
            visibilities (torch.Tensor): Tensor of shape (batch_size, seq_len) indicating which ball positions are visible (True) and which not (False).
            missing_token (torch.Tensor): Learnable token to fill in for invisible ball positions. Shape (embed_dim//2,)
        """
        _, T, _ = ball_pos.shape
        __, N, __ = table_pos.shape
        if N != self.expected_num_tokens:
            raise ValueError(
                f"Expected {self.expected_num_tokens} table tokens, got {N}."
            )
        # embed seperately
        ball_embedded = self.ball_embed(ball_pos)  # (B, T, Dim/2)
        table_pos = eo.rearrange(table_pos, "b n d -> b (n d)")
        table_embedded = self.table_embed(table_pos.unsqueeze(1)).squeeze(
            1
        )  # (B, Dim/2)
        table_embedded = eo.repeat(table_embedded, "b d -> b t d", t=T)
        # interpolate missing ball positions with learnable token
        missing_mask = visibilities == BALL_INVISIBLE  # (B, T)
        ball_embedded[missing_mask] = missing_token  # fill in missing token
        # fuse both embeddings
        x = torch.cat((ball_embedded, table_embedded), dim=2)
        x = self.fusion_layer1(x)
        x = self.relu(x)
        x = self.fusion_layer2(x)
        return x


class AttentionWithRotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        time_rotation="new",
    ):
        super().__init__()
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rotary_emb = RotaryPositionalEmbedding(
            dim // num_heads, time_rotation=time_rotation
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight, gain=1)
        nn.init.xavier_uniform_(self.proj.weight, gain=1)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, mask, times, num_cls_token=0, visibilities=None):
        """Forward pass through the transformer with applying rotary positional embeddings. Use this for the ball positions
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)
            mask (torch.Tensor): Tensor of shape (batch_size, seq_len) indicating at which times there is padding and which not
            times (torch.Tensor): Tensor of shape (batch_size, seq_len) indicating the time of each position x.
            num_cls_token (int): Number of cls tokens in the input tensor. Don't apply rotary positional embeddings to these tokens.
            visibilities (torch.Tensor): Optional tensor of shape (batch_size, seq_len)
                indicating which tokens are visible (True) and which are upsampled (False).
                Used for Deferred Upsampling Token Attention (DUTA). If not used, set to None.
        Returns:
            torch.Tensor: Tensor of shape (batch_size, seq_len, dim).
        """
        B, N, C = x.shape

        # Generate q, k, v projections
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: (B, num_heads, N, head_dim)

        # Apply rotary positional embeddings to q and k
        if num_cls_token > 0:
            c_q, q = q[:, :, :num_cls_token], q[:, :, num_cls_token:]
            c_k, k = k[:, :, :num_cls_token], k[:, :, num_cls_token:]
        q = self.rotary_emb(q, times)
        k = self.rotary_emb(k, times)
        if num_cls_token > 0:
            q = torch.cat((c_q, q), dim=2)
            k = torch.cat((c_k, k), dim=2)

        # attention masking. mask considers the padded tokens
        additive_mask = mask[:, None, None, :]

        if visibilities is not None:
            # visibilities shape: (B, N), True = Real Token, False = Upsample Token
            # Find all upsample tokens
            vis_k_is_upsample = ~visibilities  # Shape (B, N)
            # Broadcast this to (B, 1, 1, N). This blocks all queries (dim 2) from attending to the upsample keys (dim 3). -inf, 0 instead of True, False for compatibility with additive_mask
            duta_mask = torch.where(
                vis_k_is_upsample[:, None, None, :],  # Shape (B, 1, 1, N)
                -torch.inf,
                0.0,
            ).to(device=additive_mask.device, dtype=additive_mask.dtype)
            # Add the DUTA mask to the padding mask
            additive_mask = additive_mask + duta_mask

        x = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,  # Use the prepared additive mask
            dropout_p=(
                self.attn_drop_prob if self.training else 0.0
            ),  # Apply dropout only during training
            is_causal=False,  # Set to True if you need causal attention
        )  # Output shape: (B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MyHead(nn.Module):
    """Head for the transformer to predict the rotation of the ball."""

    def __init__(self, dim):
        super(MyHead, self).__init__()
        self.proj = LinearReLUMLP([dim, dim // 2, dim // 4, 3])

    def forward(self, x):
        """Forward pass through the head
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, dim) or (batch_size, seq_len, dim)
        """
        return self.proj(x)


class SimpleStaticLayer(nn.Module):
    """Residual transformer block (LayerNorm -> self-attention -> MLP)."""

    def __init__(self, dim, num_heads, qkv_bias, attn_drop_rate, time_rotation):
        super(SimpleStaticLayer, self).__init__()
        self.attn = AttentionWithRotaryPositionalEmbedding(
            dim, num_heads, qkv_bias, attn_drop_rate, time_rotation=time_rotation
        )
        self.mlp1 = Mlp(
            in_features=dim, hidden_features=dim, act_layer=nn.ReLU, drop=0.0
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask, times, num_cls_token=None, visibilities=None):
        """Forward pass through the layer
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)
            mask (torch.Tensor): Tensor of shape (batch_size, seq_len) indicating at which times there is padding and which not
            times (torch.Tensor): Tensor of shape (batch_size, seq_len) indicating the time of each position x.
            num_cls_token (int): Number of cls tokens in the input tensor. Only used for rotary attention
            visibilities (torch.Tensor): Tensor of shape (batch_size, seq_len) indicating which positions are visible (True) and which not (False).
                                        Used for Deferred Upsampling Token Attention (DUTA). If not used, it is set to None.
        """
        B, T, D = x.shape

        x_res = x
        x = self.norm1(x)
        num_cls_token = 0 if num_cls_token is None else num_cls_token
        x = self.attn(x, mask, times, num_cls_token, visibilities)
        x = x + x_res
        x_res = x
        x = self.norm2(x)
        x = self.mlp1(x)
        x = x + x_res

        return x


class FirstStage(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        qkv_bias,
        attn_drop_rate,
        mode,
        time_rotation,
        interpolate_missing=False,
    ):
        super(FirstStage, self).__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        # free = no table tokens, dynamic = table tokens for dynamic camera per time step
        assert mode in [
            "dynamic",
            "originalmethod",
            "dynamicAnkle",
        ], 'mode should be either "dynamic", "dynamicAnkle" or "originalmethod"'
        self.mode = mode
        self.ankle_embed = AnkleVEmbedding(dim, in_dim=1)

        if mode == "originalmethod":
            if interpolate_missing:
                self.ball_embed = SpecialEmbedding(dim)
            else:
                self.ball_embed = BallEmbedding(
                    dim, len(court_points) * 2 + 2
                )  # num table tokens * 2 + 2 for the ball token
        else:
            self.ball_embed = BallEmbedding(dim, 2)
        if mode in ["dynamic", "dynamicAnkle"]:
            self.table_embed = TableEmbedding(dim)
            self.pos_layers = nn.ModuleList(
                [
                    SimpleStaticLayer(
                        dim, num_heads, qkv_bias, attn_drop_rate, time_rotation
                    )
                    for _ in range(4)
                ]
            )

        self.layers = nn.ModuleList(
            [
                SimpleStaticLayer(
                    dim, num_heads, qkv_bias, attn_drop_rate, time_rotation
                )
                for _ in range(self.depth)
            ]
        )

        self.position_head = MyHead(dim)

        # if true, learnable tokens are filled in for missing ball positions, if false, missing positions are just removed
        self.interpolate_missing = interpolate_missing
        if mode == "originalmethod":
            self.missing_token = nn.Parameter(
                torch.zeros((dim // 2,)), requires_grad=True
            )
        elif mode in ["dynamic", "dynamicAnkle"]:
            self.missing_token = nn.Parameter(torch.zeros((dim,)), requires_grad=True)
        else:
            raise ValueError("Unknown mode for missing token initialization.")
        nn.init.xavier_uniform_(self.missing_token.unsqueeze(0), gain=1).squeeze(0)

    def forward(self, ball_pos, table_pos, mask, times, ankle_v=None):
        """Forward pass through the transformer. Masks are already expected to be in the correct format.
        Args:
            ball_pos (torch.Tensor): Tensor of shape (batch_size, seq_len, 3) with last coordinate being the visibility
            table_pos (torch.Tensor): Tensor of shape (batch_size, num_positions, 3) with last coordinate being the visibility
            mask (torch.Tensor): Additive attention mask of shape (batch_size, seq_len),
                with 0.0 for valid tokens and -inf for masked tokens.
            times (torch.Tensor): Tensor of shape (batch_size, seq_len) indicating the time of each position x.
            ankle_v (torch.Tensor | None): Tensor of shape (batch_size, 1), used
                in dynamicAnkle mode.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                positions (B, T, 3), token features (B, T, D), updated mask (B, T), updated times (B, T).
        """
        B, T, _ = ball_pos.shape
        D = self.dim

        ball_pos, visibilities = ball_pos[:, :, :2], ball_pos[:, :, 2]

        if self.mode in ["dynamic", "dynamicAnkle"]:
            x = self.ball_embed(ball_pos)
            if self.interpolate_missing:
                missing_mask = visibilities == BALL_INVISIBLE
                x[missing_mask] = self.missing_token
        elif self.mode == "originalmethod":
            table_pos = table_pos[:, :, :2]
            if (
                self.interpolate_missing
            ):  # We use the special embedding class that does most of the work internally
                x = self.ball_embed(
                    ball_pos, table_pos, visibilities, self.missing_token
                )
            else:
                table_pos = eo.rearrange(table_pos, "b n d -> b (n d)")
                table_pos = eo.repeat(table_pos, "b n -> b t n", t=T)
                ball_pos = torch.cat((ball_pos, table_pos), dim=2)
                x = self.ball_embed(ball_pos)

        # entirely remove missing positions (in x, mask and times)
        if not self.interpolate_missing:
            x, times, mask = pack_visible_sequence(x, times, mask, visibilities)

        if self.mode in ["dynamic", "dynamicAnkle"]:
            # mask all invisible table tokens in attention
            table_visibilities = table_pos[:, :, 2]  # (B, N)
            table_mask = torch.where(
                table_visibilities == KEYPOINT_VISIBLE, 0.0, float("-inf")
            )  # (B, N)
            table_mask = torch.cat(
                (torch.zeros((B, 1), device=table_mask.device), table_mask), dim=1
            )  # (B, N+1) class token (ball position) is visible
            table_mask = eo.repeat(table_mask, "b n -> (b t) n", t=T)  # (B*T, N+1)
            # Invent some fake times -> Encodes the position in the sequence
            table_times = torch.arange(
                table_pos.shape[1], device=table_pos.device, dtype=table_pos.dtype
            ) / (
                MAX_FPS / 5
            )  # (N,)
            table_times = eo.repeat(table_times, "n -> (b t) n", b=B, t=T)  # (B*T, N)

            # embed the table positions into the same dimension as the ball positions
            table_pos = self.table_embed(
                table_pos[..., :2]
            )  # only use u, v coordinates for the table embedding, not the visibility

            # Concatenate table position and ball position
            _, N, _ = table_pos.shape
            table_pos = table_pos.unsqueeze(1).expand(B, T, N, D)
            x = x.unsqueeze(2)
            x = torch.cat((x, table_pos), dim=2)
            x = eo.rearrange(x, "b t n d -> (b t) n d")

            # Do some Attention Layers
            for layer in self.pos_layers:
                x = layer(
                    x, table_mask, table_times, num_cls_token=1
                )  # ball position as class token, fake time to encode the table keypoint, masking if invisible
            x = eo.rearrange(x, "(b t) n d -> b t n d", b=B)
            x = x[:, :, 0, :]  # Only take the ball position tokens -> (B, T, D)

            # Attention layer for ankle position
            if self.mode == "dynamicAnkle":
                ankle_pos_embed = self.ankle_embed(ankle_v[:, :1])
                ankle_kp_embed = torch.cat(
                    (ankle_pos_embed.unsqueeze(1), table_pos[:, 0, :, :]), dim=1
                )
                table_mask = eo.rearrange(table_mask, "(b t) n -> b t n", b=B)[:, 0, :]
                table_times = eo.rearrange(table_times, "(b t) n -> b t n", b=B)[
                    :, 0, :
                ]
                for layer in self.pos_layers:
                    a = layer(ankle_kp_embed, table_mask, table_times, num_cls_token=1)
                a = a[:, 0, :]

        if self.mode == "dynamicAnkle":
            x = torch.cat((a.unsqueeze(1), x), dim=1)
            # Adjust mask for cls token
            mask_tmp = mask.new_zeros((B, T + 1))
            mask_tmp[:, 1:] = mask

            for layer in self.layers:
                x = layer(x, mask_tmp, times, num_cls_token=1)
            x = x[:, 1:, :]
        else:
            for i, layer in enumerate(self.layers):
                # Use Deferred Upsampling Token Attention (DUTA) in the first layer -> Normal Tokens should not attend to Upsample Tokens
                v = (
                    visibilities == BALL_VISIBLE
                    if (i == 0 and self.interpolate_missing)
                    else None
                )
                x = layer(x, mask, times, num_cls_token=0, visibilities=v)

        positions = self.position_head(x)
        return (
            positions,
            x,
            mask,
            times,
        )  # return mask and times for later use in the second stage, since they could be modified in the first stage.


class MultiStageModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        qkv_bias,
        attn_drop_rate,
        mode,
        time_rotation,
        use_skipconnection=False,
        interpolate_missing=True,
    ):
        super(MultiStageModel, self).__init__()
        self.dim = dim
        self.depth_secondstage = 4
        self.depth_firststage = depth - 4
        self.num_heads = num_heads
        self.mode = mode
        self.time_rotation = time_rotation

        self.embed = BallEmbedding(self.dim, 3)
        self.firststage = FirstStage(
            self.dim,
            self.depth_firststage,
            num_heads,
            qkv_bias,
            attn_drop_rate,
            mode,
            time_rotation,
            interpolate_missing,
        )
        self.secondstage = nn.ModuleList(
            [
                SimpleStaticLayer(
                    self.dim, num_heads, qkv_bias, attn_drop_rate, time_rotation
                )
                for _ in range(self.depth_secondstage)
            ]
        )

        self.rotation_head = MyHead(self.dim)

        # parameter that decides if the gradient for the rotation computation is backpropagated into the first stage
        self.full_backprop = False
        # if true, the second stage gets the high dimensional tokens as input insead of the 3D positions
        self.use_skipconnection = use_skipconnection
        # if true, learnable tokens are filled in for missing ball positions, if false, missing positions are just removed
        self.interpolate_missing = interpolate_missing

    def forward(self, ball_pos, table_pos, mask, times, ankle_v=None):
        """Forward pass through the transformer
        Args:
            ball_pos (torch.Tensor): Tensor of shape (batch_size, seq_len, 3),
                where channels are (u, v, visibility).
            table_pos (torch.Tensor): Tensor of shape (batch_size, num_positions, 3) with last coordinate being the visibility
            mask (torch.tensor): Tensor of shape (batch_size, seq_len) indicating at which times there is padding and which not
            times (torch.Tensor): Tensor of shape (batch_size, seq_len) indicating the time of each position x.
            ankle_v (torch.Tensor | None): Tensor of shape (batch_size, 1) with the
                vertical position of the serving player's ankle.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                predicted rotation axes (B, T, 3) and predicted positions (B, T, 3).
        """
        # Transform the mask such that it can be added before the softmax operation.
        mask = normalize_attention_mask(mask)

        # first stage
        pos, pos_token, mask, times = self.firststage(
            ball_pos, table_pos, mask, times, ankle_v
        )

        x = pos_token if self.use_skipconnection else pos

        # stop backpropagation -> rotation computation should not influence position computations
        if not self.full_backprop:
            x = x.detach()

        if not self.use_skipconnection:
            x = self.embed(x)

        for layer in self.secondstage:
            x = layer(x, mask, times, num_cls_token=0)

        rot = self.rotation_head(x)
        return rot, pos


def get_model(
    name="connectstage",
    size="large",
    mode="dynamic",
    time_rotation="new",
    interpolate_missing=False,
):
    assert time_rotation in [
        "old",
        "new",
    ], 'time_rotation should be either "old" or "new"'
    drop_stuff = 0.0
    if name in ["connectstage"]:
        use_skipconnection = True
        if size == "small":
            model = MultiStageModel(
                32,
                8,
                4,
                True,
                drop_stuff,
                mode=mode,
                time_rotation=time_rotation,
                use_skipconnection=use_skipconnection,
                interpolate_missing=interpolate_missing,
            )
        elif size == "base":
            model = MultiStageModel(
                64,
                12,
                4,
                True,
                drop_stuff,
                mode=mode,
                time_rotation=time_rotation,
                use_skipconnection=use_skipconnection,
                interpolate_missing=interpolate_missing,
            )
        elif size == "large":
            model = MultiStageModel(
                128,
                16,
                4,
                True,
                drop_stuff,
                mode=mode,
                time_rotation=time_rotation,
                use_skipconnection=use_skipconnection,
                interpolate_missing=interpolate_missing,
            )
        elif size == "huge":
            model = MultiStageModel(
                192,
                16,
                8,
                True,
                drop_stuff,
                mode=mode,
                time_rotation=time_rotation,
                use_skipconnection=use_skipconnection,
                interpolate_missing=interpolate_missing,
            )
        else:
            raise ValueError(f"Unknown model size {size}")
    else:
        raise ValueError(f"Unknown model name {name}, use connectstage")
    model.time_rotation = time_rotation
    return model


if __name__ == "__main__":
    for size in ["large"]:
        for modelname in ["connectstage"]:
            mode = "dynamicAnkle"
            model = get_model(
                modelname,
                size,
                mode,
                time_rotation="new",
                interpolate_missing=False,
            )

            print("size:", size, "model:", modelname)

            # Calculate total number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters: {total_params}")

            # calculate only trainable parameters.
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f"Trainable number of parameters: {trainable_params}")

            print("---")
