import torch
from torch.nn import nn

from timm.models.vision_transformer import Mlp


class EmbeddingNetwork(nn.Module):
    def __init__(self, hidden_size, m) -> None:
        super().__init__()
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)


class DeepSets(nn.Module):
    def __init__(self, embedding_network: nn.Module, output_network: nn.Module):
        super().__init__()
        self.embedding_network = embedding_network
        self.output_network = output_network

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_network(input)
        return self.output_network(torch.sum(embedding, dim=1))