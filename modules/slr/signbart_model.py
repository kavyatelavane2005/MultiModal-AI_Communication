"""SignBart network used for isolated WLASL-100 sign recognition.

The module layout and parameter names match the public SignBart checkpoint.
Inputs are MediaPipe sequences with shape (batch, time, 75, 2).
"""
import torch
from torch import nn


def _padding_attention_mask(mask, dtype, target_length=None):
    batch_size, source_length = mask.shape
    target_length = target_length or source_length
    expanded = mask[:, None, None, :].expand(
        batch_size, 1, target_length, source_length
    ).to(dtype)
    inverted = 1.0 - expanded
    return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)


def _decoder_attention_mask(mask, input_shape, input_embeddings):
    batch_size, target_length = input_shape
    expanded = mask[:, None, None, :].expand(
        batch_size, 1, target_length, target_length
    ).to(input_embeddings.dtype)
    inverted = 1.0 - expanded
    expanded = inverted.masked_fill(
        inverted.to(torch.bool), torch.finfo(input_embeddings.dtype).min
    )
    causal = torch.tril(torch.ones(
        target_length,
        target_length,
        device=input_embeddings.device,
        dtype=input_embeddings.dtype,
    ))
    return expanded + causal[None, None, :, :]


class PositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, inputs_embeds):
        batch_size, sequence_length = inputs_embeds.shape[:2]
        positions = torch.arange(
            sequence_length, dtype=torch.long, device=self.weight.device
        ).expand(batch_size, -1).to(inputs_embeds.device)
        return super().forward(positions + self.offset)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(input_dim, num_classes)

    def forward(self, values):
        return self.out_proj(self.dropout(values))


class Projection(nn.Module):
    def __init__(self, config):
        super().__init__()
        feature_count = len(config["joint_idx"])
        self.proj_x1 = nn.Linear(feature_count, config["d_model"])
        self.proj_y1 = nn.Linear(feature_count, config["d_model"])

    def forward(self, keypoints):
        return (
            self.proj_x1(keypoints[:, :, :, 0]),
            self.proj_y1(keypoints[:, :, :, 1]),
        )


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, values):
        batch_size, sequence_length, _ = values.shape
        return values.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _attend(self, query, key, value, mask):
        weights = torch.matmul(query, key.transpose(-1, -2))
        if mask is not None:
            weights = weights + mask
        probabilities = torch.softmax(weights, dim=-1)
        probabilities = nn.functional.dropout(
            probabilities, p=self.dropout, training=self.training
        )
        output = torch.matmul(probabilities, value)
        batch_size, _, sequence_length, _ = output.shape
        output = output.transpose(1, 2).reshape(
            batch_size, sequence_length, self.d_model
        )
        return self.out_proj(output)


class SelfAttention(Attention):
    def forward(self, hidden_states, attention_mask):
        query = self._split_heads(self.q_proj(hidden_states) * self.scaling)
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        return self._attend(query, key, value, attention_mask)


class CrossAttention(Attention):
    def forward(self, hidden_states, key_value_states, attention_mask):
        query = self._split_heads(self.q_proj(hidden_states) * self.scaling)
        key = self._split_heads(self.k_proj(key_value_states))
        value = self._split_heads(self.v_proj(key_value_states))
        return self._attend(query, key, value, attention_mask)


class CausalSelfAttention(SelfAttention):
    def forward(self, hidden_states, attention_mask):
        query = self._split_heads(self.q_proj(hidden_states) * self.scaling)
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        target_length = hidden_states.shape[1]
        causal = torch.tril(torch.ones(
            target_length, target_length, device=hidden_states.device
        )).view(1, 1, target_length, target_length)
        weights = torch.matmul(query, key.transpose(-1, -2))
        weights = weights.masked_fill(causal == 0, float("-inf"))
        weights = weights + attention_mask
        probabilities = torch.softmax(weights, dim=-1)
        probabilities = nn.functional.dropout(
            probabilities, p=self.dropout, training=self.training
        )
        output = torch.matmul(probabilities, value)
        batch_size, _, sequence_length, _ = output.shape
        output = output.transpose(1, 2).reshape(
            batch_size, sequence_length, self.d_model
        )
        return self.out_proj(output)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]
        self.self_attn = SelfAttention(
            self.d_model,
            config["encoder_attention_heads"],
            config["attention_dropout"],
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = config["dropout"]
        self.activation_fn = nn.GELU()
        self.activation_dropout = config["activation_dropout"]
        self.fc1 = nn.Linear(self.d_model, config["encoder_ffn_dim"])
        self.fc2 = nn.Linear(config["encoder_ffn_dim"], self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, hidden_states, attention_mask):
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = self.final_layer_norm(residual + hidden_states)
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            limit = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-limit, max=limit)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config["dropout"]
        self.layerdrop = config["encoder_layerdrop"]
        self.embed_positions = PositionalEmbedding(
            config["max_position_embeddings"], config["d_model"]
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config["encoder_layers"])]
        )
        self.layernorm_embedding = nn.LayerNorm(config["d_model"])

    def forward(self, x_embed, attention_mask):
        x_embed = x_embed + self.embed_positions(x_embed)
        hidden_states = self.layernorm_embedding(x_embed)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        attention_mask = _padding_attention_mask(attention_mask, x_embed.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]
        self.self_attn = CausalSelfAttention(
            self.d_model,
            config["decoder_attention_heads"],
            config["attention_dropout"],
        )
        self.dropout = config["dropout"]
        self.activation_fn = nn.GELU()
        self.activation_dropout = config["activation_dropout"]
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.encoder_attn = CrossAttention(
            self.d_model,
            config["decoder_attention_heads"],
            config["attention_dropout"],
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.fc1 = nn.Linear(self.d_model, config["decoder_ffn_dim"])
        self.fc2 = nn.Linear(config["decoder_ffn_dim"], self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, hidden_states, attention_mask,
                encoder_hidden_states, encoder_attention_mask):
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states, encoder_hidden_states, encoder_attention_mask
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = self.encoder_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        return self.final_layer_norm(residual + hidden_states)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config["dropout"]
        self.layerdrop = config["decoder_layerdrop"]
        self.embed_positions = PositionalEmbedding(
            config["max_position_embeddings"], config["d_model"]
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config["decoder_layers"])]
        )
        self.layernorm_embedding = nn.LayerNorm(config["d_model"])

    def forward(self, encoder_hidden_states, encoder_attention_mask,
                y_embed, attention_mask):
        input_shape = y_embed.size()[:-1]
        decoder_mask = _decoder_attention_mask(
            attention_mask, input_shape, y_embed
        )
        encoder_mask = _padding_attention_mask(
            encoder_attention_mask,
            y_embed.dtype,
            target_length=input_shape[-1],
        )
        y_embed = y_embed + self.embed_positions(y_embed)
        hidden_states = self.layernorm_embedding(y_embed)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                decoder_mask,
                encoder_hidden_states,
                encoder_mask,
            )
        return hidden_states


class SignBart(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.joint_idx = config["joint_idx"]
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.classification_head = ClassificationHead(
            config["d_model"],
            config["num_labels"],
            config["classifier_dropout"],
        )
        self.projection = Projection(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, keypoints, attention_mask, labels=None):
        batch_size = keypoints.shape[0]
        keypoints = keypoints[:, :, self.joint_idx, :]
        x_embed, y_embed = self.projection(keypoints)
        encoder_outputs = self.encoder(x_embed, attention_mask)
        decoder_outputs = self.decoder(
            encoder_outputs,
            attention_mask,
            y_embed,
            attention_mask,
        )
        last_indices = (attention_mask == 1).float().cumsum(dim=1).argmax(dim=1)
        last_outputs = decoder_outputs[
            torch.arange(batch_size, device=decoder_outputs.device), last_indices
        ]
        logits = self.classification_head(last_outputs)
        loss = self.loss_fn(logits, labels.to(logits.device)) if labels is not None else None
        return loss, logits
