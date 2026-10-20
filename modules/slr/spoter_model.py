import copy
import torch
import torch.nn as nn
from typing import Optional


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """TransformerDecoderLayer omitting redundant self-attention operation."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(SPOTERTransformerDecoderLayer, self).__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        del self.self_attn

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SPOTERTransformerDecoder(nn.Module):
    """Custom decoder container for PyTorch 2.x compatibility.
    
    Standard nn.TransformerDecoder in PyTorch 2.x expects self.layers[0].self_attn
    to exist. Since SPOTER custom decoder layer omits self_attn, this container
    calls the layers directly and accepts PyTorch 2.x causal kwargs.
    """

    def __init__(self, layers, norm=None):
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False, **kwargs):
        output = tgt
        for mod in self.layers:
            output = mod(
                output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class SPOTER(nn.Module):
    """Official SPOTER (Sign POse-based TransformER) Architecture.
    
    Trained on WLASL100 dataset.
    Input: (seq_len, 54, 2) sequence of skeletal coordinates
    Output: (1, 100) class logits
    """

    def __init__(self, num_classes=100, hidden_dim=108):
        super().__init__()

        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim))
        self.pos = nn.Parameter(
            torch.cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1)
            .flatten(0, 1)
            .unsqueeze(0)
        )
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, 9, 6, 6)
        self.linear_class = nn.Linear(hidden_dim, num_classes)

        # Deactivate initial self-attention in decoder layers
        custom_decoder_layer = SPOTERTransformerDecoderLayer(
            self.transformer.d_model, self.transformer.nhead, 2048, 0.1, "relu"
        )
        layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)
        self.transformer.decoder = SPOTERTransformerDecoder(layers, norm=self.transformer.decoder.norm)

    def forward(self, inputs):
        """Forward pass.
        
        Args:
            inputs: Tensor of shape (seq_len, 54, 2) or (seq_len, 108)
                    normalized and shifted by -0.5
        Returns:
            Logits of shape (1, num_classes)
        """
        # (seq_len, 108) -> (seq_len, 1, 108)
        if inputs.dim() == 3:
            h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        elif inputs.dim() == 2:
            h = torch.unsqueeze(inputs, 1).float()
        else:
            h = inputs.float()

        # Positional encoding + class query attention
        h = self.transformer(self.pos + h, self.class_query.unsqueeze(0)).transpose(0, 1)
        res = self.linear_class(h)  # (1, 1, num_classes)
        return res.squeeze(0)        # (1, num_classes)
