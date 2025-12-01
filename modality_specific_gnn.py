"""
Modality-Specific GNN Encoders for FC and SC.

Key insight: FC and SC are fundamentally different modalities:
- FC: Functional connectivity - captures temporal correlations, distributed patterns
- SC: Structural connectivity - captures anatomical pathways, local structure

Just like CLIP uses different architectures for text (Transformer) and images (CNN/ViT),
we use different GNN architectures optimized for each brain connectivity modality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer.
    Learns importance weights for each edge/neighbor.
    """
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        
        # Per-head output dimension
        self.head_dim = out_features // num_heads if concat else out_features
        
        # Learnable parameters for each head
        self.W = nn.Parameter(torch.Tensor(num_heads, in_features, self.head_dim))
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Linear transformation for each head: [batch, heads, nodes, head_dim]
        h = torch.einsum('bni,hio->bhno', x, self.W)
        
        # Compute attention coefficients
        # Concatenate [h_i || h_j] for all pairs
        h_i = h.unsqueeze(3).expand(-1, -1, -1, num_nodes, -1)  # [batch, heads, nodes, nodes, head_dim]
        h_j = h.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)  # [batch, heads, nodes, nodes, head_dim]
        h_concat = torch.cat([h_i, h_j], dim=-1)  # [batch, heads, nodes, nodes, 2*head_dim]
        
        # Attention scores: e_ij = LeakyReLU(a^T [W h_i || W h_j])
        e = torch.einsum('bhnmo,hoi->bhnm', h_concat, self.a).squeeze(-1)  # [batch, heads, nodes, nodes]
        e = self.leaky_relu(e)
        
        # Mask attention scores where there's no edge
        # Create mask: 1 where edge exists, -inf where no edge
        mask = (adj.unsqueeze(1) == 0)  # [batch, 1, nodes, nodes]
        e = e.masked_fill(mask, float('-inf'))
        
        # Attention weights via softmax
        alpha = F.softmax(e, dim=-1)  # [batch, heads, nodes, nodes]
        alpha = self.dropout_layer(alpha)
        
        # Aggregate neighbor features
        h_prime = torch.einsum('bhnm,bhmo->bhno', alpha, h)  # [batch, heads, nodes, head_dim]
        
        # Concatenate or average heads
        if self.concat:
            output = h_prime.reshape(batch_size, num_nodes, -1)  # [batch, nodes, heads*head_dim]
        else:
            output = h_prime.mean(dim=1)  # [batch, nodes, head_dim]
        
        return output


class MultiScaleGNNLayer(nn.Module):
    """
    Multi-scale GNN that captures both local and global patterns.
    Uses different hop neighborhoods and combines them.
    """
    
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        
        # Local (1-hop) pathway
        self.local_fc = nn.Linear(in_features, out_features // 2)
        
        # Global (2-hop) pathway  
        self.global_fc = nn.Linear(in_features, out_features // 2)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)
    
    def forward(self, x, adj):
        """
        Args:
            x: [batch, nodes, in_features]
            adj: [batch, nodes, nodes]
        """
        # Normalize adjacency
        adj_norm = self._normalize_adj(adj)
        
        # Local: 1-hop aggregation
        local_agg = torch.bmm(adj_norm, x)  # [batch, nodes, in_features]
        local_out = self.local_fc(local_agg)
        
        # Global: 2-hop aggregation
        adj_2hop = torch.bmm(adj_norm, adj_norm)
        global_agg = torch.bmm(adj_2hop, x)
        global_out = self.global_fc(global_agg)
        
        # Combine local and global
        out = torch.cat([local_out, global_out], dim=-1)
        out = self.dropout(out)
        out = self.norm(out)
        out = F.relu(out)
        
        return out
    
    def _normalize_adj(self, adj):
        """Row-normalize adjacency matrix."""
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adj / row_sum


class AttentionPooling(nn.Module):
    """
    Attention-based graph pooling.
    Learns importance weights for each node (soft filtering).
    """
    
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, nodes, features]
        
        Returns:
            pooled: [batch, features]
            attention_weights: [batch, nodes] (for interpretability)
        """
        # Compute attention scores for each node
        scores = self.attention_fc(x)  # [batch, nodes, 1]
        weights = F.softmax(scores, dim=1)  # [batch, nodes, 1]
        
        # Weighted sum of node features
        pooled = (x * weights).sum(dim=1)  # [batch, features]
        
        return pooled, weights.squeeze(-1)


class FunctionalConnectivityEncoder(nn.Module):
    """
    Encoder specialized for Functional Connectivity (FC).
    
    FC characteristics:
    - Captures temporal correlations between regions
    - Distributed patterns across the whole brain
    - Dense connectivity (many non-zero entries)
    - Symmetric matrix
    
    Architecture design:
    - Deeper network (3-4 layers) to capture distributed patterns
    - Graph Attention to learn which connections are important
    - Multi-scale processing (local + global)
    - LayerNorm (better for continuous-valued FC)
    """
    
    def __init__(self, matrix_dim, num_nodes, embed_dim=256, 
                 hidden_dims=[128, 128, 128], num_heads=4, dropout=0.2):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.num_nodes = num_nodes
        
        # Initial node embedding from adjacency
        self.node_embed = nn.Linear(matrix_dim, hidden_dims[0])
        
        # Multi-layer GAT encoder
        self.gat_layers = nn.ModuleList()
        in_dim = hidden_dims[0]
        for hidden_dim in hidden_dims:
            self.gat_layers.append(
                GraphAttentionLayer(in_dim, hidden_dim, num_heads, dropout, concat=True)
            )
            in_dim = hidden_dim
        
        # Multi-scale layer for global patterns
        self.multiscale = MultiScaleGNNLayer(hidden_dims[-1], hidden_dims[-1], dropout)
        
        # Attention pooling (learns which nodes matter)
        self.attention_pool = AttentionPooling(hidden_dims[-1])
        
        # Final projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, fc_matrix):
        """
        Args:
            fc_matrix: [batch_size, matrix_dim, matrix_dim]
        
        Returns:
            embeddings: [batch_size, embed_dim]
        """
        batch_size = fc_matrix.shape[0]
        
        # Initial node features from adjacency rows
        x = self.node_embed(fc_matrix)  # [batch, nodes, hidden_dim]
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply GAT layers
        for gat in self.gat_layers:
            x_new = gat(x, fc_matrix)
            x = F.relu(x_new) + x if x.shape == x_new.shape else F.relu(x_new)  # Residual if dims match
            x = self.dropout(x)
        
        # Multi-scale aggregation
        x = self.multiscale(x, fc_matrix)
        
        # Attention pooling to graph-level representation
        graph_embed, _ = self.attention_pool(x)
        
        # Project to final embedding
        embedding = self.projection(graph_embed)
        
        return embedding


class StructuralConnectivityEncoder(nn.Module):
    """
    Encoder specialized for Structural Connectivity (SC).
    
    SC characteristics:
    - Captures anatomical white matter pathways
    - Sparse connectivity (many zero entries)
    - Local structural organization
    - Asymmetric matrix (directional pathways)
    
    Architecture design:
    - Shallower network (2 layers) focused on local structure
    - Standard GNN (not GAT) since SC already defines structure
    - BatchNorm (better for sparse, discrete-like SC)
    - Smaller hidden dims (less distributed patterns)
    """
    
    def __init__(self, matrix_dim, num_nodes, embed_dim=256,
                 hidden_dims=[96, 96], dropout=0.2):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.num_nodes = num_nodes
        
        # Initial node embedding
        self.node_embed = nn.Linear(matrix_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(num_nodes)
        
        # GNN layers (simpler than FC encoder)
        self.gnn_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.gnn_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.gnn_layers.append(nn.BatchNorm1d(num_nodes))
        
        # Attention pooling
        self.attention_pool = AttentionPooling(hidden_dims[-1])
        
        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _gnn_layer(self, x, adj):
        """Single GNN aggregation step."""
        # Normalize adjacency
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj / row_sum
        
        # Aggregate neighbors
        x_agg = torch.bmm(adj_norm, x)
        return x_agg
    
    def forward(self, sc_matrix):
        """
        Args:
            sc_matrix: [batch_size, matrix_dim, matrix_dim]
        
        Returns:
            embeddings: [batch_size, embed_dim]
        """
        batch_size = sc_matrix.shape[0]
        
        # Initial node features
        x = self.node_embed(sc_matrix)  # [batch, nodes, hidden_dim]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply GNN layers
        for i in range(0, len(self.gnn_layers), 2):
            # GNN aggregation
            x_agg = self._gnn_layer(x, sc_matrix)
            # Linear transformation
            x = self.gnn_layers[i](x_agg)
            # Batch norm
            x = self.gnn_layers[i+1](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Attention pooling
        graph_embed, _ = self.attention_pool(x)
        
        # Project to final embedding
        embedding = self.projection(graph_embed)
        
        return embedding


class ModalitySpecificGNNCLIP(nn.Module):
    """
    CLIP model with modality-specific encoders for FC and SC.
    
    Key design principles:
    1. Different architectures for different modalities (like text/image in CLIP)
    2. FC encoder: Deep GAT with attention, captures distributed patterns
    3. SC encoder: Shallow GNN with BatchNorm, captures local structure
    4. Both project to shared 256-dim embedding space
    5. Trained with contrastive loss to align paired FC-SC samples
    """
    
    def __init__(self, fc_dim, sc_dim, embed_dim=256, num_nodes=90,
                 fc_hidden_dims=[128, 128, 128], sc_hidden_dims=[96, 96],
                 num_heads=4, dropout=0.2,
                 use_projection: bool = False, proj_hidden_dim: int = 256):
        super().__init__()
        
        self.fc_encoder = FunctionalConnectivityEncoder(
            matrix_dim=fc_dim,
            num_nodes=num_nodes,
            embed_dim=embed_dim,
            hidden_dims=fc_hidden_dims,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.sc_encoder = StructuralConnectivityEncoder(
            matrix_dim=sc_dim,
            num_nodes=num_nodes,
            embed_dim=embed_dim,
            hidden_dims=sc_hidden_dims,
            dropout=dropout
        )
        
        # Learnable temperature parameter (like in CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        # Optional small projection heads (help stabilize contrastive training)
        self.use_projection = use_projection
        if self.use_projection:
            # Two-layer MLP projection (embed_dim -> proj_hidden_dim -> embed_dim)
            self.proj_fc = nn.Sequential(
                nn.Linear(embed_dim, proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(proj_hidden_dim),
                nn.Linear(proj_hidden_dim, embed_dim)
            )
            self.proj_sc = nn.Sequential(
                nn.Linear(embed_dim, proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(proj_hidden_dim),
                nn.Linear(proj_hidden_dim, embed_dim)
            )
        else:
            self.proj_fc = None
            self.proj_sc = None
    
    def encode_fc(self, fc_matrix):
        """Encode functional connectivity to embedding space."""
        return self.fc_encoder(fc_matrix)
    
    def encode_sc(self, sc_matrix):
        """Encode structural connectivity to embedding space."""
        return self.sc_encoder(sc_matrix)
    
    def forward(self, fc_matrix, sc_matrix):
        """
        Forward pass through both encoders.

        Returns:
            fc_embed, sc_embed, logit_scale
        """
        # Raw embeddings from encoders
        fc_embed_raw = self.encode_fc(fc_matrix)
        sc_embed_raw = self.encode_sc(sc_matrix)

        # Apply projection heads if requested
        if self.use_projection and (self.proj_fc is not None) and (self.proj_sc is not None):
            fc_embed = self.proj_fc(fc_embed_raw)
            sc_embed = self.proj_sc(sc_embed_raw)
        else:
            fc_embed = fc_embed_raw
            sc_embed = sc_embed_raw

        return fc_embed, sc_embed, self.logit_scale.exp()


def create_modality_specific_gnn_clip(fc_dim=90, sc_dim=90, embed_dim=256, 
                                       num_nodes=90, dropout=0.2,
                                       fc_hidden_dims=[128,128,128], sc_hidden_dims=[96,96],
                                       num_heads=4, use_projection: bool = False, proj_hidden_dim: int = 256):
    """
    Factory function to create modality-specific GNN-CLIP model.
    
    Default configuration:
    - FC encoder: 3-layer GAT with 128 hidden dims, 4 attention heads
    - SC encoder: 2-layer GNN with 96 hidden dims
    - Shared embedding dimension: 256
    - Dropout: 0.2
    """
    return ModalitySpecificGNNCLIP(
        fc_dim=fc_dim,
        sc_dim=sc_dim,
        embed_dim=embed_dim,
        num_nodes=num_nodes,
        fc_hidden_dims=fc_hidden_dims,
        sc_hidden_dims=sc_hidden_dims,
        num_heads=num_heads,
        dropout=dropout,
        use_projection=use_projection,
        proj_hidden_dim=proj_hidden_dim
    )
