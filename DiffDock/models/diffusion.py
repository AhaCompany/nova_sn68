import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion model for protein-ligand binding
    """
    def __init__(self, 
                 denoise_fn, 
                 num_timesteps=1000,
                 beta_schedule='linear',
                 beta_start=1e-4,
                 beta_end=0.02):
        super().__init__()
        
        self.denoise_fn = denoise_fn
        self.num_timesteps = num_timesteps
        
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'cosine':
            steps = torch.arange(num_timesteps + 1) / num_timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')
        
        self.register_buffer('betas', betas)
        
        # Diffusion process parameters
        alphas = 1. - betas
        self.register_buffer('alphas', alphas)
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and noise"""
        return (
            extract(self.alphas_cumprod, t, x_t.shape) * x_t -
            extract(torch.sqrt(1. - self.alphas_cumprod), t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance
    
    def p_mean_variance(self, x, t):
        """
        Compute the model's predicted mean and variance for x_{t-1} given x_t
        """
        # Predict noise
        predicted_noise = self.denoise_fn(x, t)
        
        # Predict x_0
        x_recon = self.predict_start_from_noise(x, t, predicted_noise)
        
        # Clip x_0 for better stability
        x_recon = torch.clamp(x_recon, -1, 1)
        
        # Forward process posterior parameters
        model_mean, posterior_variance = self.q_posterior(x_recon, x, t)
        
        return model_mean, posterior_variance, x_recon
    
    @torch.no_grad()
    def p_sample(self, x, t):
        """
        Sample x_{t-1} from the model using generative process
        """
        model_mean, model_variance, x_recon = self.p_mean_variance(x, t)
        
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        # Compute x_{t-1}
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        Generate samples from the model
        """
        device = next(self.parameters()).device
        
        # Start from pure Gaussian noise
        x = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, torch.full((shape[0],), t, device=device, dtype=torch.long))
        
        return x
    
    def forward(self, x, noise=None):
        """
        Forward diffusion process
        """
        b = x.shape[0]
        device = x.device
        
        # Sample timesteps uniformly
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x)
        
        # Forward noising process
        x_noisy = (
            torch.sqrt(extract(self.alphas_cumprod, t, x.shape)) * x +
            torch.sqrt(1. - extract(self.alphas_cumprod, t, x.shape)) * noise
        )
        
        # Predict the noise
        predicted_noise = self.denoise_fn(x_noisy, t)
        
        # Return MSE loss
        return F.mse_loss(predicted_noise, noise)

# Helper functions
def extract(a, t, shape):
    """
    Extract values from a at timesteps t and reshape to match shape
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(shape) - 1)))