import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import beta as Beta
from scipy.optimize import bisect


class Flow1D(nn.Module):
    def __init__(self, num_mixture_components=5, alpha=4, beta=4):
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
        loss_beta_distrib = -((self.alpha - 1) * torch.log(z + 1e-9) + (self.beta - 1) * torch.log(1 - z + 1e-9)).mean()
        loss_derivative = -torch.log(torch.abs(dz_dx) + 1e-9).mean()

        loss = loss_beta_distrib + loss_derivative
        return loss


    @torch.no_grad()
    def sample(self, num_samples, device=torch.device("cpu")):
        self.eval()

        x_samples = []
        # Sample zs from Beta distribution
        z_samples = Beta.rvs(self.alpha, self.beta, size=num_samples)
        
        for i, z_sample in enumerate(z_samples):
            # f: R -> R
            def f(x): 
                 # Computing the components of mixture of logistics CDF
                logistic_cdf_components = 1 / (1 + torch.exp(-(x - self.means) / torch.exp(self.logscales)))

                # Make sure that mixing coefficients sum to 1!
                normalized_mixing_coeffs = F.softmax(self.mixing_coefs, dim=0).unsqueeze(1)

                # Weighting the mixture components by mixing coefficients
                mixed_logistic_cdf = torch.matmul(logistic_cdf_components, normalized_mixing_coeffs)

                # Finding the inverse of CDF at z is equivalent to finding the root of CDF - z
                return mixed_logistic_cdf.item() - z_sample
            
            x_sample = bisect(f, a=-10, b=10, xtol=1e-5, maxiter=50)
            x_samples.append(x_sample)

        x_samples = torch.tensor(x_samples, device=device).view(num_samples, -1)

        self.train()
        return x_samples



if __name__ == "__main__":
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    # Creating the model
    model = Flow1D().to(device)

    # Doing sampling sanity check
    print(model.sample(num_samples=12))


    