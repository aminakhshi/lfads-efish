import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

from lfads.model import LFADS


class LFADSTrainer:
    """
    Trainer class for LFADS model.
    """
    def __init__(self, model, train_dl, val_dl=None, lr=1e-3, scheduler_patience=5,
                 scheduler_factor=0.5, clip_gradient=5.0, checkpoint_dir='checkpoints'):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=scheduler_patience, 
            factor=scheduler_factor, verbose=True
        )
        self.clip_gradient = clip_gradient
        self.device = next(model.parameters()).device
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Track training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Run one epoch of training"""
        self.model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        num_batches = 0
        
        with tqdm(self.train_dl, desc="Training") as progress_bar:
            for x_batch, _ in progress_bar:  # Second element is ignored (dummy target)
                x_batch = x_batch.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                rates, kl_div = self.model(x_batch)
                total_loss, recon_loss, kl_loss = self.model.loss_function(x_batch, rates, kl_div)
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                if self.clip_gradient:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)
                
                # Update weights
                self.optimizer.step()
                
                # Update KL weight according to schedule
                self.model.update_kl_weight()
                
                # Track losses
                epoch_loss += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                num_batches += 1
                
                progress_bar.set_postfix({
                    'loss': total_loss.item(),
                    'recon': recon_loss.item(),
                    'kl': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                })
        
        return epoch_loss / num_batches, epoch_recon / num_batches, epoch_kl / num_batches
    
    def validate(self):
        """Run validation"""
        if self.val_dl is None:
            return 0.0, 0.0, 0.0
        
        self.model.eval()
        val_loss = 0
        val_recon = 0
        val_kl = 0
        num_batches = 0
        
        with torch.no_grad():
            for x_batch, _ in tqdm(self.val_dl, desc="Validating"):
                x_batch = x_batch.to(self.device)
                
                # Forward pass
                rates, kl_div = self.model(x_batch)
                total_loss, recon_loss, kl_loss = self.model.loss_function(x_batch, rates, kl_div)
                
                # Track losses
                val_loss += total_loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                num_batches += 1
        
        return val_loss / num_batches, val_recon / num_batches, val_kl / num_batches
    
    def train(self, num_epochs, save_every=5, validate_every=1):
        """
        Train the model for multiple epochs.
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoints every N epochs
            validate_every: Validate every N epochs
        """
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_recon, train_kl = self.train_epoch()
            self.train_losses.append((train_loss, train_recon, train_kl))
            
            # Validate
            val_loss, val_recon, val_kl = 0, 0, 0
            if self.val_dl is not None and (epoch + 1) % validate_every == 0:
                val_loss, val_recon, val_kl = self.validate()
                self.val_losses.append((val_loss, val_recon, val_kl))
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    print(f"New best model saved (val_loss: {val_loss:.6f})")
            
            # Print metrics
            print(f"Train Loss: {train_loss:.6f} (Recon: {train_recon:.6f}, KL: {train_kl:.6f})")
            if self.val_dl is not None and (epoch + 1) % validate_every == 0:
                print(f"Val Loss: {val_loss:.6f} (Recon: {val_recon:.6f}, KL: {val_kl:.6f})")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        print("Training completed!")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'iteration': self.model.iteration,
            'kl_weight_enc': self.model.kl_weight_enc,
            'kl_weight_con': self.model.kl_weight_con,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.model.iteration = checkpoint['iteration']
        self.model.kl_weight_enc = checkpoint['kl_weight_enc']
        self.model.kl_weight_con = checkpoint['kl_weight_con']
        
        print(f"Checkpoint loaded from {path}")
    
    def plot_losses(self, figsize=(12, 5)):
        """Plot training and validation losses"""
        plt.figure(figsize=figsize)
        
        # Training losses
        epochs = range(1, len(self.train_losses) + 1)
        train_total = [x[0] for x in self.train_losses]
        train_recon = [x[1] for x in self.train_losses]
        train_kl = [x[2] for x in self.train_losses]
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_total, 'b-', label='Total Loss')
        plt.plot(epochs, train_recon, 'g-', label='Reconstruction')
        plt.plot(epochs, train_kl, 'r-', label='KL Divergence')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Validation losses if available
        if self.val_losses:
            val_epochs = range(1, len(self.val_losses) + 1)
            val_total = [x[0] for x in self.val_losses]
            val_recon = [x[1] for x in self.val_losses]
            val_kl = [x[2] for x in self.val_losses]
            
            plt.subplot(1, 2, 2)
            plt.plot(val_epochs, val_total, 'b-', label='Total Loss')
            plt.plot(val_epochs, val_recon, 'g-', label='Reconstruction')
            plt.plot(val_epochs, val_kl, 'r-', label='KL Divergence')
            plt.title('Validation Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        plt.tight_layout()
        plt.show()