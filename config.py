"""
Configuration settings for neural network models.
"""

# Neural data parameters
NUM_NEURONS = 128  # Number of neurons in recordings
SEQUENCE_LENGTH = 100  # Default sequence length for time series data

# BrainFormer model parameters
EMBED_DIM = 64  # Embedding dimension
HIDDEN_DIM = 128  # Hidden layer dimension
NUM_HEADS = 4  # Number of attention heads
NUM_LAYERS = 3  # Number of transformer layers
DROPOUT_RATE = 0.1  # Dropout rate for regularization
FF_DIM = 256  # Feed-forward dimension in transformer
OUTPUT_DIM = 32  # Output dimension for predictions

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
