"""
AdvancedBrainTransformer: A transformer-based model for neural data processing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embedding.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with multi-head attention and feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, mask=None):
        """
        Forward pass through transformer encoder block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Processed tensor of same shape as input
        """
        # Multi-head attention with residual connection and layer normalization
        x_norm = self.layernorm1(x)
        x_norm = x_norm.permute(1, 0, 2)  # (seq_len, batch, embed_dim) for nn.MultiheadAttention
        
        if mask is not None:
            attn_output, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        else:
            attn_output, _ = self.attn(x_norm, x_norm, x_norm)
            
        attn_output = attn_output.permute(1, 0, 2)  # Back to (batch, seq_len, embed_dim)
        x = x + self.dropout(attn_output)
        
        # Feed-forward network with residual connection and layer normalization
        x_norm = self.layernorm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        
        return x


class AdvancedBrainTransformer(nn.Module):
    """
    Transformer-based model for processing neural spike data.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input embedding layer
        self.embedding = nn.Linear(config.NUM_NEURONS, config.EMBED_DIM)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.EMBED_DIM, 
            dropout=config.DROPOUT_RATE
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                config.EMBED_DIM,
                config.NUM_HEADS,
                config.FF_DIM,
                config.DROPOUT_RATE
            ) for _ in range(config.NUM_LAYERS)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.EMBED_DIM, config.OUTPUT_DIM)
        
    def forward(self, x, padding_mask=None, task='multi'):
        """
        Forward pass through BrainTransformer.
        
        Args:
            x: Input spike data of shape (batch, seq_len, num_neurons)
            padding_mask: Optional mask for padded values (batch, seq_len)
            task: Task type (e.g., 'multi', 'classification')
            
        Returns:
            Dictionary with model outputs including embeddings and task-specific outputs
        """
        # Apply input embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process through transformer encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, padding_mask)
        
        # Project to output dimension
        output = self.output_projection(x)
        
        # Return embeddings and outputs in a dictionary
        results = {
            'embeddings': x,  # Encoder representations
            'output': output,  # Projected output
            'task_output': None  # Placeholder for task-specific output
        }
        
        # Task-specific processing
        if task == 'classification':
            # For classification, pool over sequence dimension
            results['task_output'] = torch.mean(output, dim=1)
        else:
            # For sequence-to-sequence tasks
            results['task_output'] = output
            
        return results
