import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import os


def raster_plot(spike_data, time_range=None, neuron_range=None, title="Spike Raster Plot", 
                figsize=(10, 6), markersize=1, alpha=0.5):
    """
    Create a raster plot from spike data.
    
    Args:
        spike_data: Can be:
            - binned spikes tensor (batch, time, neurons)
            - list of spike times per neuron
        time_range: Optional tuple (start, end) to restrict time axis
        neuron_range: Optional tuple (start, end) to restrict displayed neurons
        title: Plot title
        figsize: Figure size
        markersize: Size of spike markers
        alpha: Transparency of markers
    """
    plt.figure(figsize=figsize)
    
    # Handle different input formats
    if isinstance(spike_data, torch.Tensor):
        # Handle binned data (batch, time, neurons)
        if len(spike_data.shape) == 3:
            spike_data = spike_data[0]  # Take first batch element
        
        # Convert to numpy if needed
        if isinstance(spike_data, torch.Tensor):
            spike_data = spike_data.cpu().numpy()
        
        # Now spike_data is (time, neurons)
        times, neurons = spike_data.shape
        
        # Apply ranges
        if time_range is not None:
            start_t, end_t = time_range
            spike_data = spike_data[start_t:end_t]
            time_offset = start_t
        else:
            time_offset = 0
            
        if neuron_range is not None:
            start_n, end_n = neuron_range
            spike_data = spike_data[:, start_n:end_n]
            neuron_offset = start_n
        else:
            neuron_offset = 0
        
        # Plot spike times for each neuron
        for n in range(spike_data.shape[1]):
            spike_times = np.where(spike_data[:, n] > 0)[0]
            
            if len(spike_times) > 0:
                # For each spike time, we might have multiple spikes (count > 1)
                for t in spike_times:
                    count = int(spike_data[t, n])
                    plt.plot([t + time_offset] * count, [n + neuron_offset] * count, 'k.',
                             markersize=markersize, alpha=alpha)
    
    elif isinstance(spike_data, list):
        # Handle list of spike times per neuron
        if neuron_range is None:
            neuron_range = (0, len(spike_data))
        start_n, end_n = neuron_range
        
        for n, neuron_spikes in enumerate(spike_data[start_n:end_n], start=start_n):
            if time_range is not None:
                start_t, end_t = time_range
                neuron_spikes = [t for t in neuron_spikes if start_t <= t < end_t]
            
            plt.plot(neuron_spikes, [n] * len(neuron_spikes), 'k.',
                     markersize=markersize, alpha=alpha)
    
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Neuron")
    plt.tight_layout()
    
    return plt.gca()


def plot_factors(factors, num_to_plot=None, smooth_sigma=None, figsize=(12, 8)):
    """
    Plot extracted latent factors.
    
    Args:
        factors: Latent factors tensor (batch, time, factors)
        num_to_plot: Number of factors to display (default: all)
        smooth_sigma: Standard deviation for Gaussian smoothing
        figsize: Figure size
    """
    if isinstance(factors, torch.Tensor):
        factors = factors.detach().cpu().numpy()
    
    if len(factors.shape) == 3:
        factors = factors[0]  # Take first batch element
    
    n_factors = factors.shape[1]
    if num_to_plot is None or num_to_plot > n_factors:
        num_to_plot = n_factors
    
    plt.figure(figsize=figsize)
    time_steps = np.arange(factors.shape[0])
    
    for i in range(num_to_plot):
        factor = factors[:, i]
        
        if smooth_sigma is not None:
            factor = gaussian_filter1d(factor, smooth_sigma)
        
        plt.subplot(num_to_plot, 1, i+1)
        plt.plot(time_steps, factor)
        plt.ylabel(f"Factor {i+1}")
        
        if i == 0:
            plt.title("Latent Factors")
        if i == num_to_plot - 1:
            plt.xlabel("Time")
    
    plt.tight_layout()
    return plt.gcf()


def plot_reconstruction(original_data, reconstructed_data, neuron_indices=None, 
                        time_range=None, figsize=(12, 8)):
    """
    Plot original vs reconstructed neural data.
    
    Args:
        original_data: Original spike data (batch, time, neurons)
        reconstructed_data: Reconstructed rates (batch, time, neurons)
        neuron_indices: Which neurons to plot (default: first 5)
        time_range: Optional tuple (start, end) to restrict time axis
        figsize: Figure size
    """
    if isinstance(original_data, torch.Tensor):
        original_data = original_data.detach().cpu().numpy()
    if isinstance(reconstructed_data, torch.Tensor):
        reconstructed_data = reconstructed_data.detach().cpu().numpy()
    
    # Take first batch if batched
    if len(original_data.shape) == 3:
        original_data = original_data[0]
    if len(reconstructed_data.shape) == 3:
        reconstructed_data = reconstructed_data[0]
    
    # Now data is (time, neurons)
    n_time, n_neurons = original_data.shape
    
    # Select neurons to plot
    if neuron_indices is None:
        neuron_indices = range(min(5, n_neurons))
    elif isinstance(neuron_indices, int):
        neuron_indices = range(min(neuron_indices, n_neurons))
    
    n_plot = len(neuron_indices)
    
    # Apply time range
    if time_range is not None:
        start_t, end_t = time_range
        time_steps = np.arange(start_t, end_t)
        original_data = original_data[start_t:end_t]
        reconstructed_data = reconstructed_data[start_t:end_t]
    else:
        time_steps = np.arange(n_time)
    
    plt.figure(figsize=figsize)
    
    for i, neuron_idx in enumerate(neuron_indices):
        plt.subplot(n_plot, 1, i+1)
        
        # Plot original spikes
        plt.stem(time_steps, original_data[:, neuron_idx], 'b', 
                markerfmt='b.', basefmt='b-', label='Original Spikes', use_line_collection=True)
        
        # Plot reconstructed rates
        plt.plot(time_steps, reconstructed_data[:, neuron_idx], 'r-', label='Reconstructed Rate')
        
        plt.ylabel(f"Neuron {neuron_idx}")
        if i == 0:
            plt.title("Original Spikes vs Reconstructed Rates")
            plt.legend()
        if i == n_plot - 1:
            plt.xlabel("Time")
    
    plt.tight_layout()
    return plt.gcf()


def plot_2d_factors(factors, labels=None, pca_dims=2, figsize=(10, 8)):
    """
    Project factors to 2D and visualize, optionally colored by condition.
    
    Args:
        factors: Latent factors tensor (batch, time, factors)
        labels: Optional labels for coloring points (batch, time)
        pca_dims: Number of PCA dimensions for projection
        figsize: Figure size
    """
    if isinstance(factors, torch.Tensor):
        factors = factors.detach().cpu().numpy()
    
    # Reshape to (batch*time, factors)
    batch_size, seq_len, n_factors = factors.shape
    factors_flat = factors.reshape(-1, n_factors)
    
    # Apply PCA
    pca = PCA(n_components=pca_dims)
    factors_pca = pca.fit_transform(factors_flat)
    
    # Reshape back to (batch, time, pca_dims)
    factors_pca = factors_pca.reshape(batch_size, seq_len, pca_dims)
    
    # Plot
    plt.figure(figsize=figsize)
    
    for b in range(batch_size):
        # Extract trajectory for this batch
        trajectory = factors_pca[b]
        
        if labels is not None and len(labels) > b:
            label = labels[b]
            plt.scatter(trajectory[:, 0], trajectory[:, 1], c=label, 
                        cmap='viridis', alpha=0.7, s=10)
            plt.colorbar(label='Condition')
        else:
            # Plot with time-based coloring
            plt.scatter(trajectory[:, 0], trajectory[:, 1], 
                        c=np.arange(seq_len), cmap='viridis', 
                        alpha=0.7, s=10)
            plt.colorbar(label='Time')
        
        # Plot trajectory line
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        # Mark start and end
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'ko', markersize=8)
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'kx', markersize=8)
    
    plt.title(f"Latent Factor Trajectories (PCA, {pca_dims} components)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def create_data_for_lfads(spike_data, seq_len, train_fraction=0.8, 
                          batch_size=32, device='cuda', multi_sequence=False):
    """
    Prepare data for LFADS training.
    
    Args:
        spike_data: Spike data in one of these formats:
            - Array of shape (trials, time, neurons)
            - Array of shape (time, neurons) for single trial
        seq_len: Sequence length for each LFADS input
        train_fraction: Fraction of data to use for training
        batch_size: Batch size for data loaders
        device: Device to put tensors on
        multi_sequence: Whether the data consists of multiple sequences
        
    Returns:
        train_dl: Training data loader
        valid_dl: Validation data loader
    """
    # Handle different input formats
    if not multi_sequence and len(spike_data.shape) == 2:
        # Single sequence (time, neurons)
        time_steps, n_neurons = spike_data.shape
        
        # Convert to (1, time, neurons) for consistent processing
        spike_data = spike_data.reshape(1, time_steps, n_neurons)
        multi_sequence = True
    
    if multi_sequence:
        # Input is (trials, time, neurons)
        n_sequences, time_steps, n_neurons = spike_data.shape
        
        # Create list to hold all sequence chunks
        all_chunks = []
        
        # Process each sequence
        for i in range(n_sequences):
            # Get this sequence
            seq = spike_data[i]
            
            # Divide into chunks of seq_len
            for start in range(0, time_steps - seq_len + 1, seq_len // 2):  # 50% overlap
                chunk = seq[start:start+seq_len]
                if len(chunk) == seq_len:  # Only include complete chunks
                    all_chunks.append(chunk)
        
        # Convert list to array
        all_chunks = np.array(all_chunks)
        
        # Split into train/valid
        n_chunks = len(all_chunks)
        n_train = int(n_chunks * train_fraction)
        
        # Shuffle chunks
        indices = np.random.permutation(n_chunks)
        train_indices = indices[:n_train]
        valid_indices = indices[n_train:]
        
        # Create PyTorch datasets and loaders
        train_data = torch.tensor(all_chunks[train_indices], dtype=torch.float32).to(device)
        valid_data = torch.tensor(all_chunks[valid_indices], dtype=torch.float32).to(device)
        
        train_ds = torch.utils.data.TensorDataset(train_data, torch.zeros(len(train_data)))
        valid_ds = torch.utils.data.TensorDataset(valid_data, torch.zeros(len(valid_data)))
        
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)
        
        return train_dl, valid_dl
    
    else:
        raise ValueError("Input data format not recognized")