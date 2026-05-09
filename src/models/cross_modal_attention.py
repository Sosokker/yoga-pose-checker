import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class JointEmbedding(nn.Module):
    """Projects 17 keypoints from 4D (x, y, z, visibility) to d_model dimensions."""

    def __init__(self, input_dim: int = 4, d_model: int = 128, num_joints: int = 17, dropout: float = 0.0):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model

        self.linear = nn.Linear(input_dim, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_joints, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)  # (batch, 17, d_model)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + self.pos_embedding
        return out


class JointSelfAttention(nn.Module):
    """
    Standard multi-head self-attention over the 17 joint tokens.
    Models anatomical relationships (e.g., knee-ankle-hip).
    """

    def __init__(self, d_model: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        residual = x

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (batch, heads, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(context)

        return self.layer_norm(residual + out)


class JointSelfAttentionEncoder(nn.Module):
    """Stack of JointSelfAttention layers."""

    def __init__(self, d_model: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [JointSelfAttention(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention where pose tokens (Q) attend to class prototypes (K, V).
    """

    def __init__(self, d_model: int = 128, num_classes: int = 107, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.scale = math.sqrt(d_model)

        self.class_prototypes = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_joints, d_model = x.shape

        Q = x
        K = self.class_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        V = K

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.bmm(attn_weights, V)
        out = self.layer_norm(x + context)

        return out, attn_weights


class CrossModalAttentionBridge(nn.Module):
    """Stack of CrossModalAttention layers."""

    def __init__(
        self,
        d_model: int = 128,
        num_classes: int = 107,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossModalAttention(d_model, num_classes, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_attn = []
        for layer in self.layers:
            x, attn = layer(x)
            all_attn.append(attn)
        return x, all_attn


class ClassificationHead(nn.Module):
    """Pool across joints (mean+max) with optional raw keypoint skip connection, then FC layers."""

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 107,
        dropout: float = 0.3,
        raw_input_dim: int = 4,
        pool_type: str = "meanmax",
        use_flatten_raw: bool = False,
    ):
        super().__init__()
        self.pool_type = pool_type
        self.use_flatten_raw = use_flatten_raw
        if pool_type == "meanmax":
            pooled_dim = d_model * 2
            raw_pooled_dim = raw_input_dim * 2
        else:
            pooled_dim = d_model
            raw_pooled_dim = raw_input_dim

        if use_flatten_raw:
            raw_feat_dim = 17 * raw_input_dim
        else:
            raw_feat_dim = d_model
            self.raw_proj = nn.Linear(raw_pooled_dim, d_model)

        layers = []
        prev_dim = pooled_dim + raw_feat_dim
        for h in [hidden_dim, hidden_dim // 2]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "meanmax":
            return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor, raw: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self._pool(x)
        if raw is not None:
            if self.use_flatten_raw:
                raw_feat = raw.view(raw.size(0), -1)
            else:
                raw_pooled = self._pool(raw)
                raw_feat = self.raw_proj(raw_pooled)
            pooled = torch.cat([pooled, raw_feat], dim=-1)
        return self.net(pooled)


class CrossModalPoseClassifier(nn.Module):
    """
    Full model:
    Joint Embedding → [Optional Self-Attention] → Cross-Modal Attention Bridge → Classification Head.
    """

    def __init__(
        self,
        num_classes: int = 107,
        input_dim: int = 4,
        d_model: int = 128,
        num_joints: int = 17,
        num_attention_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        embed_dropout: float = 0.1,
        use_joint_self_attention: bool = False,
        num_self_attention_layers: int = 2,
        num_self_attention_heads: int = 4,
        pool_type: str = "meanmax",
        use_flatten_raw: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_joint_self_attention = use_joint_self_attention
        self.input_dim = input_dim

        self.embedding = JointEmbedding(input_dim, d_model, num_joints, dropout=embed_dropout)

        if use_joint_self_attention:
            self.self_attention = JointSelfAttentionEncoder(
                d_model=d_model,
                num_heads=num_self_attention_heads,
                num_layers=num_self_attention_layers,
                dropout=dropout,
            )
        else:
            self.self_attention = None

        self.attention_bridge = CrossModalAttentionBridge(
            d_model, num_classes, num_attention_layers, dropout
        )
        self.head = ClassificationHead(
            d_model, hidden_dim, num_classes, dropout, raw_input_dim=input_dim, pool_type=pool_type, use_flatten_raw=use_flatten_raw
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raw = x
        x = self.embedding(x)

        if self.self_attention is not None:
            x = self.self_attention(x)

        x, all_attn = self.attention_bridge(x)
        logits = self.head(x, raw=raw)
        return logits, all_attn


class BaselineMLP(nn.Module):
    """
    Simple 3-layer MLP baseline on flattened keypoints.
    Input: 17 joints * 4 dims = 68
    """

    def __init__(self, input_dim: int = 68, hidden_dims: list[int] = [256, 128], num_classes: int = 107, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x)
