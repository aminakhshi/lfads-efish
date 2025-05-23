import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np


class PiecewiseProportionalDropout(nn.Module):
    """
    Implementation of piecewise dropout from the LFADS paper, where using different dropout
    proportions for input vs. hidden states improves performance.
    """
    def __init__(self, input_p=0.25, hidden_p=0.5):
        super().__init__()
        self.input_p = input_p
        self.hidden_p = hidden_p
    
    def forward(self, input_tensor, hidden_tensor):
        if self.training:
            input_mask = torch.bernoulli(
                torch.ones_like(input_tensor) * (1 - self.input_p)
            ) / (1 - self.input_p)
            hidden_mask = torch.bernoulli(
                torch.ones_like(hidden_tensor) * (1 - self.hidden_p)
            ) / (1 - self.hidden_p)
            
            return input_tensor * input_mask, hidden_tensor * hidden_mask
        return input_tensor, hidden_tensor


class ConstrainedGRUCell(nn.Module):
    """
    GRU Cell with optional constraint on recurrent matrix to help with
    exploding/vanishing gradients.
    """
    def __init__(self, input_size, hidden_size, max_norm=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_norm = max_norm
        
        # Update gate parameters
        self.weight_xz = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_z = nn.Parameter(torch.Tensor(hidden_size))
        
        # Reset gate parameters
        self.weight_xr = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_r = nn.Parameter(torch.Tensor(hidden_size))
        
        # Output parameters
        self.weight_xh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for weight in [self.weight_xz, self.weight_hz, self.weight_xr, 
                       self.weight_hr, self.weight_xh, self.weight_hh]:
            nn.init.kaiming_uniform_(weight, a=np.sqrt(5))
            
        fan_in_x, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_xh)
        fan_in_h, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_hh)
        
        bound = 1 / np.sqrt(fan_in_x)
        nn.init.uniform_(self.bias_z, -bound, bound)
        nn.init.uniform_(self.bias_r, -bound, bound)
        nn.init.uniform_(self.bias_h, -bound, bound)
    
    def _constrain_recurrent(self, weight):
        """Apply soft constraints to recurrent weights"""
        if self.max_norm is not None:
            # Apply column-wise normalization to recurrent weights
            norm = torch.norm(weight, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm)
            weight = weight * (desired / (1e-7 + norm))
        return weight
    
    def forward(self, x, h):
        """
        GRU Cell forward pass with constrained recurrent matrix
        Args:
            x: input tensor of shape (batch, input_size)
            h: hidden state tensor of shape (batch, hidden_size)
        Returns:
            h_new: new hidden state tensor of shape (batch, hidden_size)
        """
        # Constrain recurrent weights
        weight_hz = self._constrain_recurrent(self.weight_hz)
        weight_hr = self._constrain_recurrent(self.weight_hr)
        weight_hh = self._constrain_recurrent(self.weight_hh)
        
        # Update gate
        z = torch.sigmoid(
            F.linear(x, self.weight_xz.t()) +
            F.linear(h, weight_hz.t()) +
            self.bias_z
        )
        
        # Reset gate
        r = torch.sigmoid(
            F.linear(x, self.weight_xr.t()) +
            F.linear(h, weight_hr.t()) +
            self.bias_r
        )
        
        # Candidate hidden state
        h_tilde = torch.tanh(
            F.linear(x, self.weight_xh.t()) +
            F.linear(r * h, weight_hh.t()) +
            self.bias_h
        )
        
        # New hidden state
        h_new = (1 - z) * h + z * h_tilde
        
        return h_new


class BiDirectionalGRU(nn.Module):
    """
    Bidirectional GRU encoder used in LFADS to encode neural data.
    """
    def __init__(self, input_size, hidden_size, dropout_p=0.0, max_norm=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else None
        
        # Forward and backward GRU cells
        self.gru_forward = ConstrainedGRUCell(input_size, hidden_size, max_norm)
        self.gru_backward = ConstrainedGRUCell(input_size, hidden_size, max_norm)
    
    def forward(self, x):
        """
        Process sequence bidirectionally with GRU.
        Args:
            x: input tensor of shape (batch, seq_len, input_size)
        Returns:
            forward_states: forward GRU states (batch, seq_len, hidden_size)
            backward_states: backward GRU states (batch, seq_len, hidden_size)
            combined_final: combined final hidden state (batch, 2*hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initial hidden states
        h_forward = torch.zeros(batch_size, self.hidden_size, device=device)
        h_backward = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Store all states
        forward_states = []
        backward_states = []
        
        # Forward pass
        for t in range(seq_len):
            if self.dropout is not None:
                h_forward = self.dropout(h_forward)
            h_forward = self.gru_forward(x[:, t], h_forward)
            forward_states.append(h_forward)
        
        # Backward pass
        for t in range(seq_len-1, -1, -1):
            if self.dropout is not None:
                h_backward = self.dropout(h_backward)
            h_backward = self.gru_backward(x[:, t], h_backward)
            backward_states.insert(0, h_backward)  # Insert at front to maintain seq order
        
        # Convert to tensors
        forward_states = torch.stack(forward_states, dim=1)  # (batch, seq_len, hidden)
        backward_states = torch.stack(backward_states, dim=1)  # (batch, seq_len, hidden)
        
        # Combine final states
        combined_final = torch.cat([forward_states[:, -1], backward_states[:, 0]], dim=-1)
        
        return forward_states, backward_states, combined_final


class VariationalEncoder(nn.Module):
    """
    Variational encoder for LFADS, which maps neural data to a distribution over
    initial conditions for the generator.
    """
    def __init__(self, input_size, hidden_size, latent_size, dropout_p=0.0, max_norm=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Bidirectional encoder to process data
        self.encoder = BiDirectionalGRU(input_size, hidden_size, dropout_p, max_norm)
        
        # Transform encoder outputs to distribution parameters
        self.fc_mean = nn.Linear(2 * hidden_size, latent_size)
        self.fc_logvar = nn.Linear(2 * hidden_size, latent_size)
    
    def forward(self, x):
        """
        Encode input sequence to distribution over latent initial conditions.
        Args:
            x: input tensor of shape (batch, seq_len, input_size)
        Returns:
            mean: mean of latent distribution (batch, latent_size)
            logvar: log variance of latent distribution (batch, latent_size)
            z: sampled latent vector (batch, latent_size)
            kl: KL divergence loss term
        """
        # Encode inputs
        _, _, encoder_final = self.encoder(x)
        
        # Map to distribution parameters
        mean = self.fc_mean(encoder_final)
        logvar = self.fc_logvar(encoder_final)
        
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        # Sample from distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        # Calculate KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        
        return mean, logvar, z, kl


class InputEncoder(nn.Module):
    """
    Encoder for inferred inputs (controller) in LFADS.
    """
    def __init__(self, input_size, hidden_size, latent_size, dropout_p=0.0, max_norm=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Bidirectional encoder
        self.encoder = BiDirectionalGRU(input_size, hidden_size, dropout_p, max_norm)
        
        # Output layers for means and logvars at each timestep
        self.fc_mean = nn.Linear(hidden_size * 2, latent_size)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_size)
    
    def forward(self, x):
        """
        Encode inputs to inferred input distributions for each timestep.
        Args:
            x: input tensor of shape (batch, seq_len, input_size)
        Returns:
            means: means of input distributions (batch, seq_len, latent_size)
            logvars: log variances of input distributions (batch, seq_len, latent_size)
            zs: sampled latent vectors (batch, seq_len, latent_size)
            kl: KL divergence loss term
        """
        batch_size, seq_len, _ = x.shape
        
        # Get bidirectional encoder states
        forward_states, backward_states, _ = self.encoder(x)
        
        # Combine forward and backward states for each timestep
        combined_states = torch.cat([forward_states, backward_states], dim=2)
        
        # Calculate distribution parameters for each timestep
        means = self.fc_mean(combined_states)
        logvars = self.fc_logvar(combined_states)
        
        # Clamp logvars for stability
        logvars = torch.clamp(logvars, min=-10.0, max=10.0)
        
        # Sample from distribution at each timestep
        stds = torch.exp(0.5 * logvars)
        eps = torch.randn_like(stds)
        zs = means + eps * stds
        
        # Calculate KL divergence (sum over time and latent dimensions)
        kl = -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp(), dim=2)
        
        return means, logvars, zs, kl


class Generator(nn.Module):
    """
    Generator RNN for LFADS. Takes initial condition and inferred inputs
    and produces latent factors.
    """
    def __init__(self, latent_size, hidden_size, factor_size, input_size=0,
                 dropout_p=0.0, max_norm=None):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.factor_size = factor_size
        self.input_size = input_size
        
        # Initialize parameters to convert latent initial state to generator initial state
        self.fc_init = nn.Linear(latent_size, hidden_size)
        
        # Generator GRU cell
        self.gru = ConstrainedGRUCell(input_size, hidden_size, max_norm)
        
        # Factors output layer
        self.fc_factors = nn.Linear(hidden_size, factor_size)
        
        # Optional dropout
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else None
    
    def forward(self, z, inferred_inputs=None):
        """
        Generate latent factors from initial condition and optional inferred inputs.
        Args:
            z: initial condition tensor of shape (batch, latent_size)
            inferred_inputs: optional tensor of inferred inputs (batch, seq_len, input_size)
        Returns:
            factors: generated factors (batch, seq_len, factor_size)
        """
        batch_size = z.shape[0]
        device = z.device
        seq_len = inferred_inputs.shape[1] if inferred_inputs is not None else 1
        
        # Initialize generator state from latent initial condition
        generator_state = torch.tanh(self.fc_init(z))
        
        # Create empty inputs if none provided
        if inferred_inputs is None:
            inferred_inputs = torch.zeros(batch_size, seq_len, self.input_size, device=device)
        
        # Generate factors for each timestep
        states = []
        factors = []
        
        for t in range(seq_len):
            # Apply dropout to state if needed
            if self.dropout is not None and self.training:
                generator_state = self.dropout(generator_state)
            
            # Get input for this timestep (zeros if no inferred inputs)
            if inferred_inputs is not None:
                input_t = inferred_inputs[:, t]
            else:
                input_t = torch.zeros(batch_size, self.input_size, device=device)
            
            # Update generator state
            generator_state = self.gru(input_t, generator_state)
            states.append(generator_state)
            
            # Generate factors from state
            factor_t = self.fc_factors(generator_state)
            factors.append(factor_t)
        
        # Stack factors for all timesteps
        factors = torch.stack(factors, dim=1)  # (batch, seq_len, factor_size)
        states = torch.stack(states, dim=1)    # (batch, seq_len, hidden_size)
        
        return factors, states


class ObservationModel(nn.Module):
    """
    Maps latent factors to observed spike rates.
    """
    def __init__(self, factor_size, output_size, bias_init=None):
        super().__init__()
        self.factor_size = factor_size
        self.output_size = output_size
        
        # Linear mapping followed by exponential to get non-negative rates
        self.fc_rates = nn.Linear(factor_size, output_size)
        
        # Initialize bias if provided (e.g., from data)
        if bias_init is not None:
            with torch.no_grad():
                self.fc_rates.bias.copy_(bias_init)
    
    def forward(self, factors):
        """
        Map factors to rates for Poisson distribution.
        Args:
            factors: latent factors (batch, seq_len, factor_size)
        Returns:
            rates: predicted rates (batch, seq_len, output_size)
        """
        logrates = self.fc_rates(factors)  # Linear transformation
        rates = torch.exp(logrates)  # Exponential to get positive rates
        return rates


class LFADS(nn.Module):
    """
    Latent Factor Analysis via Dynamical Systems.
    A variational sequential auto-encoder for analyzing neural population activity.
    """
    def __init__(self, input_size, hidden_size=100, latent_size=32, factor_size=8,
                 inferred_input_size=0, generator_size=100, dropout_p=0.0,
                 max_norm=200.0, poisson_loss=True, bias_init=None, 
                 kl_weight_init_con=0.1, kl_weight_init_enc=0.1,
                 kl_weight_schedule_length=2000):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.factor_size = factor_size
        self.inferred_input_size = inferred_input_size
        self.generator_size = generator_size
        self.poisson_loss = poisson_loss
        
        # KL divergence weight scheduling
        self.kl_weight_init_enc = kl_weight_init_enc
        self.kl_weight_init_con = kl_weight_init_con
        self.kl_weight_schedule_length = kl_weight_schedule_length
        self.kl_weight_enc = kl_weight_init_enc
        self.kl_weight_con = kl_weight_init_con
        self.iteration = 0
        
        # Create encoder for initial conditions
        self.encoder = VariationalEncoder(
            input_size, hidden_size, latent_size, dropout_p, max_norm)
        
        # Create optional input encoder for inferred inputs
        self.input_encoder = None
        if inferred_input_size > 0:
            self.input_encoder = InputEncoder(
                input_size, hidden_size, inferred_input_size, dropout_p, max_norm)
        
        # Create generator
        self.generator = Generator(
            latent_size, generator_size, factor_size, 
            inferred_input_size, dropout_p, max_norm)
        
        # Create observation model
        self.observation = ObservationModel(factor_size, input_size, bias_init)
    
    def forward(self, x, return_factors=False):
        """
        Forward pass through LFADS.
        Args:
            x: input tensor of shape (batch, seq_len, input_size)
            return_factors: whether to return latent factors
        Returns:
            rates: predicted rates (batch, seq_len, input_size)
            kl_div: KL divergence loss term
            factors: optional latent factors
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode sequences to initial condition distributions
        g0_mean, g0_logvar, g0, kl_g0 = self.encoder(x)
        
        # Optionally encode to inferred input distributions
        u_mean, u_logvar, u, kl_u = None, None, None, torch.zeros(batch_size, device=x.device)
        if self.input_encoder is not None:
            u_mean, u_logvar, u, kl_u = self.input_encoder(x)
        
        # Generate latent factors from initial conditions and inferred inputs
        factors, generator_states = self.generator(g0, u)
        
        # Map factors to rates
        rates = self.observation(factors)
        
        # Calculate KL divergence with current weights
        kl_div = self.kl_weight_enc * kl_g0.mean() + self.kl_weight_con * kl_u.sum(dim=1).mean()
        
        if return_factors:
            return rates, kl_div, factors
        return rates, kl_div
    
    def compute_recon_loss(self, x, rates):
        """
        Compute reconstruction loss.
        Args:
            x: input data (batch, seq_len, input_size)
            rates: predicted rates (batch, seq_len, input_size)
        Returns:
            loss: reconstruction loss (Poisson or MSE)
        """
        if self.poisson_loss:
            # Poisson negative log-likelihood
            log_rates = torch.log(rates + 1e-8)
            recon_loss = torch.mean(rates - x * log_rates)
        else:
            # Mean squared error
            recon_loss = F.mse_loss(rates, x)
            
        return recon_loss
    
    def update_kl_weight(self):
        """Update KL weights according to schedule"""
        self.iteration += 1
        if self.iteration < self.kl_weight_schedule_length:
            progress = self.iteration / self.kl_weight_schedule_length
            # Linear ramp-up
            self.kl_weight_enc = self.kl_weight_init_enc + (1.0 - self.kl_weight_init_enc) * progress
            self.kl_weight_con = self.kl_weight_init_con + (1.0 - self.kl_weight_init_con) * progress
    
    def loss_function(self, x, rates, kl_div):
        """
        Compute total loss: reconstruction + KL divergence.
        Args:
            x: input data (batch, seq_len, input_size)
            rates: predicted rates (batch, seq_len, input_size)
            kl_div: KL divergence loss term
        Returns:
            total_loss: combined loss
            recon_loss: reconstruction loss
            kl_div: KL divergence term
        """
        recon_loss = self.compute_recon_loss(x, rates)
        total_loss = recon_loss + kl_div
        
        return total_loss, recon_loss, kl_div
    
    def predict(self, x):
        """
        Generate predictions from input data.
        Args:
            x: input data (batch, seq_len, input_size)
        Returns:
            rates: predicted rates
            factors: latent factors
        """
        self.eval()
        with torch.no_grad():
            rates, _, factors = self.forward(x, return_factors=True)
        return rates, factors