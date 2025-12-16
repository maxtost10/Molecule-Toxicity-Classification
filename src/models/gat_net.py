import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATConv, global_mean_pool
import pytorch_lightning as pl
from sklearn.metrics import f1_score, roc_auc_score

class HepatotoxicityGAT(pl.LightningModule):
    """
    Graph Attention Network for hepatotoxicity prediction.
    """
    
    def __init__(self, node_features, pos_weight=None, hidden_dim=64, num_heads=4, num_layers=3, dropout=0.2, lr=0.001):
        super(HepatotoxicityGAT, self).__init__()
        
        self.save_hyperparameters()
        self.dropout = dropout
        self.lr = lr
        
        # --- 1. Multi-Column Embeddings ---
        # We create a specific embedding layer for EACH column in the input x.
        # node_features: The number of columns in x (e.g., 9).
        # We use a vocab size of 120 for ALL columns to be safe (covers Atomic Num ~118).
        # For columns with fewer options (like Degree), the higher indices just go unused.
        self.feature_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=120, embedding_dim=hidden_dim) 
            for _ in range(node_features)
        ])
        
        # --- 2. GAT Layers ---
        self.gat_layers = nn.ModuleList()
        
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=3)
        )
        
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=3)
            )
            
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, edge_dim=3)
        )
        
        # --- 3. Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight) if pos_weight else None)
        self.validation_step_outputs = []

    def forward(self, x, edge_index, edge_attr, batch, return_attention_weights=False):
        # --- STEP 1: Sum of Embeddings ---
        # x shape: [Num_Nodes, Num_Features] (e.g., 9 columns)
        # We iterate over columns and add their embeddings together.
        
        x_embedded = 0
        
        # Loop through each column (Atom Type, Degree, etc.)
        for i, embedding_layer in enumerate(self.feature_embeddings):
            # Extract column 'i'
            column_values = x[:, i].long()
            
            # Embed it and add to total
            # Each layer learns: "What does Degree=3 mean?" vs "What does Carbon mean?"
            x_embedded += embedding_layer(column_values)
            
        x = x_embedded
        
        # --- STEP 2: GAT Layers ---
        attention_weights = []
        
        for i, gat_layer in enumerate(self.gat_layers):
            if return_attention_weights:
                x, (edge_index_att, att_weights) = gat_layer(
                    x, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
                attention_weights.append((edge_index_att, att_weights))
            else:
                x = gat_layer(x, edge_index, edge_attr=edge_attr)
            
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # --- STEP 3: Pooling & Classification ---
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        
        if return_attention_weights:
            return out, attention_weights
        return out

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = self.criterion(out.squeeze(), batch.y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = self.criterion(out.squeeze(), batch.y)
        probs = torch.sigmoid(out.squeeze())
        preds = (probs > 0.5).float()
        self.validation_step_outputs.append({'val_loss': loss, 'preds': preds, 'probs': probs, 'targets': batch.y})
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs: return
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
        all_probs = torch.cat([x['probs'] for x in outputs]).cpu().numpy()
        all_targets = torch.cat([x['targets'] for x in outputs]).cpu().numpy()
        val_f1 = f1_score(all_targets, all_preds)
        try: val_auc = roc_auc_score(all_targets, all_probs)
        except: val_auc = 0.5
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)
        self.log('val_auc', val_auc, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10),
            'monitor': 'val_f1', 
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]