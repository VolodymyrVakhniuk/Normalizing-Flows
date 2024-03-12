import torch
import torch.nn as nn
import torch.nn.functional as F

class Flow_1D(nn.Module):
    def __init__(self, num_mixture_components, alpha=4, beta=4):
        super().__init__()

        # Modeling the invertible differentiable mapping via the mixture of logistics
        self.means = nn.Parameter(torch.randn(num_mixture_components))
        self.logscales = nn.Parameter(torch.randn(num_mixture_components))
        self.mixing_coefs = nn.Parameter(torch.randn(num_mixture_components))

        self.num_mixture_components = num_mixture_components
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        # Computing the components of mixture of logistics CDF
        logistic_cdf_components = 1 / (1 + torch.exp(-(x - self.means) / torch.exp(self.logscales)))

        # Make sure that mixing coefficients sum to 1!
        normalized_mixing_coeffs = F.softmax(self.mixing_coefs, dim=0).unsqueeze(1)

        # Weighting the mixture components by mixing coefficients
        mixed_logistic_cdf = torch.matmul(logistic_cdf_components, normalized_mixing_coeffs)

        return mixed_logistic_cdf
    

    def get_loss(self, x):
        x.requires_grad_(True)

        # Forward pass
        z = self(x)

        # Computing the derivative wrt x
        dz_dx = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]

        # MLE loss using beta(alpha, beta) distribution as distribution over z
        loss_beta = -((self.alpha - 1) * torch.log(z + 1e-9) + (self.beta - 1) * torch.log(1 - z + 1e-9)).mean()
        loss_derivative = -torch.log(torch.abs(dz_dx) + 1e-9).mean()

        loss = loss_beta + loss_derivative
        return loss
    