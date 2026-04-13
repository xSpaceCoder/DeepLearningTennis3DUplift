import torch
import torch.nn as nn
import einops as eo

from uplifting.helper import court_points, MAX_FPS
from uplifting.helper import KEYPOINT_VISIBLE


def ensure_additive_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert a time mask to additive attention-mask format.

    Parameters:
        mask (torch.Tensor): Shape (B, T). Either binary mask from the dataset
            (1.0 for valid timesteps, 0.0 for padding) or an already-additive mask
            (0.0 for valid, -inf for padding).

    Returns:
        torch.Tensor: Shape (B, T) in additive-mask format.
    """
    if mask.min() == 0 and mask.max() == 1:
        return torch.zeros_like(mask).masked_fill(mask == 0, float("-inf"))
    return mask


def prepend_cls_mask(mask: torch.Tensor) -> torch.Tensor:
    """Prepend one valid CLS position to an additive time mask.

    Parameters:
        mask (torch.Tensor): Shape (B, T) in additive-mask format.

    Returns:
        torch.Tensor: Shape (B, T + 1), where the first column corresponds to
        the CLS token and is always valid (0.0).
    """
    batch_size, seq_len = mask.shape
    mask_with_cls = torch.zeros((batch_size, seq_len + 1), device=mask.device)
    mask_with_cls[:, 1:] = mask
    return mask_with_cls


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
        """Apply rotary positional embeddings to attention head features.

        Parameters:
            x (torch.Tensor): Shape (B, H, T, Dh).
            times (torch.Tensor): Shape (B, T). Required for
                ``time_rotation='new'`` and ignored for ``time_rotation='old'``.

        Returns:
            torch.Tensor: Shape (B, H, T, Dh) after RoPE rotation.
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


class CourtEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(CourtEmbedding, self).__init__()
        self.dim = embed_dim
        self.in_dim = 2  # u, v in image space
        self.num_tokens = 16  # number of predefined tennis court positions
        self.fc1 = nn.Linear(self.in_dim, self.dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.dim, self.dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        """Embed court keypoint image coordinates.

        Parameters:
            x (torch.Tensor): Shape (B, N, 2), where N is the number of court
                keypoints.

        Returns:
            torch.Tensor: Shape (B, N, embed_dim).
        """
        B, N, D = x.shape
        assert D == self.in_dim, "Input dimension does not match embedding dimension."
        assert (
            N <= self.num_tokens
        ), "Number of tokens exceeds the number of predefined court positions."
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class BallEmbedding(nn.Module):
    def __init__(self, embed_dim, in_dim=2):
        super(BallEmbedding, self).__init__()
        self.dim = embed_dim
        self.in_dim = in_dim  # usually u, v in image space
        self.fc1 = nn.Linear(self.in_dim, self.dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.dim, self.dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        """Embed ball token features.

        Parameters:
            x (torch.Tensor): Shape (B, T, in_dim). In training this is usually
                ``r_img`` with shape (B, T, 2).

        Returns:
            torch.Tensor: Shape (B, T, embed_dim).
        """
        B, N, D = x.shape
        assert D == self.in_dim, "Input dimension does not match embedding dimension."
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class AnkleVEmbedding(nn.Module):
    def __init__(self, embed_dim, in_dim=1):
        super(AnkleVEmbedding, self).__init__()
        self.dim = embed_dim
        self.in_dim = in_dim
        self.fc1 = nn.Linear(self.in_dim, self.dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.dim, self.dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        """Embed the ankle-v conditioning value.

        Parameters:
            x (torch.Tensor): Shape (B, 1).

        Returns:
            torch.Tensor: Shape (B, embed_dim).
        """
        B, D = x.shape
        assert D == self.in_dim, "Input dimension does not match embedding dimension."
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
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

    def forward(self, x, mask, times, num_cls_token=0):
        """Apply multi-head self-attention with RoPE on non-CLS tokens.

        Parameters:
            x (torch.Tensor): Shape (B, T, C).
            mask (torch.Tensor): Shape (B, T) in additive-mask format
                (0.0 valid, -inf padding).
            times (torch.Tensor): Shape (B, T), timestamps used by RoPE.
            num_cls_token (int): Number of leading tokens excluded from RoPE.

        Returns:
            torch.Tensor: Shape (B, T, C).
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

        # Faster implementation
        additive_mask = mask[:, None, None, :] + mask[:, None, :, None]
        x = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,  # Use the prepared additive mask
            dropout_p=self.attn_drop_prob,  # Pass dropout probability
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
        self.fc1 = nn.Linear(dim, dim // 2)
        self.fc2 = nn.Linear(dim // 2, dim // 4)
        self.fc3 = nn.Linear(dim // 4, 3)  # 3 for the rotation axis
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        """Predict a 3D vector from transformer features.

        Parameters:
            x (torch.Tensor): Shape (B, C) or (B, T, C).

        Returns:
            torch.Tensor: Shape (B, 3) or (B, T, 3), matching leading dims of ``x``.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class SimpleStaticLayer(nn.Module):
    """First an attention between the positions at a specific time is computed, afterwards an attention between the different times."""

    def __init__(self, dim, num_heads, qkv_bias, attn_drop_rate, time_rotation):
        super(SimpleStaticLayer, self).__init__()
        self.attn = AttentionWithRotaryPositionalEmbedding(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            time_rotation=time_rotation,
        )
        self.mlp1 = Mlp(
            in_features=dim, hidden_features=dim, act_layer=nn.ReLU, drop=0.0
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask, times, num_cls_token=None):
        """Run one pre-norm Transformer block.

        Parameters:
            x (torch.Tensor): Shape (B, T, D).
            mask (torch.Tensor): Shape (B, T), additive-mask format.
            times (torch.Tensor): Shape (B, T), per-token timestamps.
            num_cls_token (int | None): Number of leading CLS-like tokens to
                exclude from RoPE.

        Returns:
            torch.Tensor: Shape (B, T, D).
        """
        B, T, D = x.shape

        x_res = x
        x = self.norm1(x)
        if num_cls_token is not None:
            x = self.attn(x, mask, times, num_cls_token)
        else:
            x = self.attn(x, mask, times, 0)
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
    ):
        super(FirstStage, self).__init__()
        self.dim = dim

        assert mode in [
            "dynamic",
            "dynamicAnkle",
            "originalmethod",
        ], 'mode should be either "dynamic", "dynamicAnkle" or "originalmethod"'
        self.mode = mode

        self.ankle_embed = AnkleVEmbedding(dim, in_dim=1)

        if mode == "originalmethod":
            in_dim = len(court_points) * 2 + 2
            self.ball_embed = BallEmbedding(dim, in_dim)
        else:
            self.ball_embed = BallEmbedding(dim, 2)

        if mode in ["dynamic", "dynamicAnkle"]:
            self.court_embed = CourtEmbedding(dim)
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
                for _ in range(depth)
            ]
        )
        self.position_head = MyHead(dim)

    def forward(self, ball_pos, court_pos, mask, times, ankle_v_img=None):
        """Predict per-timestep 3D positions from image-space inputs.

        Parameters:
            ball_pos (torch.Tensor): Shape (B, T, 2). Comes from ``r_img`` in the
                dataset/training pipeline.
            court_pos (torch.Tensor): Shape (B, N, 3), where channels are
                ``(u, v, visibility)``.
            mask (torch.Tensor): Shape (B, T). In training this starts as binary
                (1 valid / 0 padded), but this stage also accepts additive masks.
            times (torch.Tensor): Shape (B, T), sampled frame timestamps.
            ankle_v_img (torch.Tensor | None): Shape (B, 1), ankle v-coordinate
                conditioning used by ``mode='dynamicAnkle'``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - predicted_positions: Shape (B, T, 3)
                - latent_tokens: Shape (B, T, dim)
        """
        B, T, _ = ball_pos.shape
        D = self.dim

        if self.mode == "originalmethod":
            court_pos = court_pos[:, :, :2]
            court_pos = eo.rearrange(court_pos, "b n d -> b (n d)")
            court_pos = eo.repeat(court_pos, "b n -> b t n", t=T)
            ball_pos = torch.cat((ball_pos, court_pos), dim=2)

        x = self.ball_embed(ball_pos)

        if self.mode in ["dynamic", "dynamicAnkle"]:
            court_visibilities = court_pos[:, :, 2]
            court_mask = torch.where(
                court_visibilities == KEYPOINT_VISIBLE, 0.0, float("-inf")
            )
            # Change: num_cls_token depends on ankle
            num_cls = 1
            court_mask = torch.cat(
                (torch.zeros((B, num_cls), device=court_mask.device), court_mask), dim=1
            )
            court_mask = eo.repeat(court_mask, "b n -> (b t) n", t=T)

            court_times = eo.repeat(
                torch.arange(court_pos.shape[1], device=court_pos.device)
                / (MAX_FPS / 5),
                "n -> (b t) n",
                b=B,
                t=T,
            )
            court_pos_emb = (
                self.court_embed(court_pos[..., :2]).unsqueeze(1).expand(B, T, -1, D)
            )

            x = torch.cat((x.unsqueeze(2), court_pos_emb), dim=2)
            x = eo.rearrange(x, "b t n d -> (b t) n d")
            for layer in self.pos_layers:
                x = layer(x, court_mask, court_times, num_cls_token=num_cls)
            x = eo.rearrange(x, "(b t) n d -> b t n d", b=B)[:, :, 0, :]
            if self.mode == "dynamicAnkle":
                ankle_pos_embed = self.ankle_embed(ankle_v_img[:, :1])
                ankle_kp_embed = torch.cat(
                    (ankle_pos_embed.unsqueeze(1), court_pos_emb[:, 0, :, :]), dim=1
                )
                court_mask = eo.rearrange(court_mask, "(b t) n -> b t n", b=B)[:, 0, :]
                court_times = eo.rearrange(court_times, "(b t) n -> b t n", b=B)[
                    :, 0, :
                ]
                for layer in self.pos_layers:
                    a = layer(
                        ankle_kp_embed, court_mask, court_times, num_cls_token=num_cls
                    )
                a = a[:, 0, :]

        if self.mode == "dynamicAnkle":
            x = torch.cat((a.unsqueeze(1), x), dim=1)
            # Adjust mask for cls token
            mask = prepend_cls_mask(mask)

            for layer in self.layers:
                x = layer(x, mask, times, num_cls_token=1)
            x = x[:, 1:, :]
        else:
            for layer in self.layers:
                x = layer(x, mask, times)

        return self.position_head(x), x


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
    ):
        super(MultiStageModel, self).__init__()
        self.dim = dim
        self.firststage = FirstStage(
            dim,
            depth - 4,
            num_heads,
            qkv_bias,
            attn_drop_rate,
            mode,
            time_rotation,
        )
        self.secondstage = nn.ModuleList(
            [
                SimpleStaticLayer(
                    dim, num_heads, qkv_bias, attn_drop_rate, time_rotation
                )
                for _ in range(4)
            ]
        )
        self.embed = BallEmbedding(self.dim, 3)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim), requires_grad=True)
        nn.init.xavier_uniform_(self.cls_token)
        self.rotation_head = MyHead(self.dim)
        self.full_backprop = False
        self.use_skipconnection = use_skipconnection

    def forward(self, ball_pos, court_pos, mask, times, ankle_v_img=None):
        """Predict global rotation and per-frame 3D trajectory.

        Parameters:
            ball_pos (torch.Tensor): Shape (B, T, 2), ball image coordinates.
            court_pos (torch.Tensor): Shape (B, N, 3), court keypoints with
                visibility channel.
            mask (torch.Tensor): Shape (B, T), binary or additive time mask.
            times (torch.Tensor): Shape (B, T), timestamps for each ball token.
            ankle_v_img (torch.Tensor | None): Shape (B, 1), ankle v feature.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - predicted_rotation: Shape (B, 3)
                - predicted_positions: Shape (B, T, 3)
        """
        B, T, _ = ball_pos.shape
        mask = ensure_additive_mask(mask)

        pos, pos_token = self.firststage(ball_pos, court_pos, mask, times, ankle_v_img)
        x = pos_token if self.use_skipconnection else pos
        if not self.full_backprop:
            x = x.detach()
        if not self.use_skipconnection:
            x = self.embed(x)

        x = torch.cat((self.cls_token.expand(B, 1, self.dim), x), dim=1)
        mask_tmp = prepend_cls_mask(mask)

        for layer in self.secondstage:
            x = layer(x, mask_tmp, times, num_cls_token=1)

        return self.rotation_head(x[:, 0, :]), pos


def get_model(
    name="connectstage",
    size="large",
    mode="dynamic",
    time_rotation="new",
):
    """Factory for the uplifting model used by training/validation scripts.

    Parameters:
        name (str): Supported: ``'connectstage'``.
        size (str): One of ``'small'``, ``'base'``, ``'large'``, ``'huge'``.
        mode (str): Token fusion mode passed into ``FirstStage``
            (e.g. ``'dynamic'``, ``'dynamicAnkle'``, ``'originalmethod'``).
        time_rotation (str): RoPE timing mode, ``'old'`` or ``'new'``.

    Returns:
        MultiStageModel: Configured model instance with attribute
        ``model.time_rotation`` set.
    """
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
            )
        else:
            raise ValueError(f"Unknown model size {size}")
    else:
        raise ValueError(f"Unknown model name {name}")
    model.time_rotation = time_rotation
    return model


if __name__ == "__main__":
    for size in ["large"]:
        for modelname in ["connectstage"]:
            mode = "dynamic"
            model = get_model(modelname, size, mode=mode)

            print("size:", size, "model:", modelname, "mode: ", mode)

            # Calculate total number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters: {total_params}")

            # calculate only trainable parameters.
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f"Trainable number of parameters: {trainable_params}")

            print("---")
