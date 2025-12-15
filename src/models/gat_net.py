import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class HepatotoxicityGAT(nn.Module):
    """Graph Attention Network for hepatotoxicity prediction"""
    
    def __init__(self, node_features, hidden_dim=64, num_heads=4, num_layers=3, dropout=0.2):
        super(HepatotoxicityGAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(node_features, hidden_dim, heads=num_heads, dropout=dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
        
        # Last layer (single head for classification)
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, batch, return_attention_weights=False):
        attention_weights = []
        
        # GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            if return_attention_weights:
                x, (edge_index_att, att_weights) = gat_layer(
                    x, edge_index, return_attention_weights=True
                )
                attention_weights.append((edge_index_att, att_weights))
            else:
                x = gat_layer(x, edge_index)
            
            # Apply activation and dropout (except for last layer)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        out = self.classifier(x)
        
        if return_attention_weights:
            return out, attention_weights
        return out