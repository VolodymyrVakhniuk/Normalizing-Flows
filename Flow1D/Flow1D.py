import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import beta as Beta
from scipy.special import betaln
from scipy.optimize import bisect

# Implementation of 1D normalizing flow model that maps unknown 1D input distribution to Beta(alpha, beta) distribution
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
    

    def log_likelihood(self, x):
        x.requires_grad_(True)

        # Forward pass
        z = self(x)

        # Computing the derivative wrt x
        dz_dx = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=torch.ones_like(z), retain_graph=False, create_graph=False)[0]

        # log_likelihood is similar to get_loss, but with all constants inplace 
        log_reciprocal_beta = -betaln(self.alpha, self.beta)  
        log_likelihood_z = log_reciprocal_beta + (self.alpha - 1) * torch.log(z + 1e-9) + (self.beta - 1) * torch.log(1 - z + 1e-9)
        jacobian = torch.log(torch.abs(dz_dx) + 1e-9)

        log_likelihood_x = log_likelihood_z + jacobian
        return log_likelihood_x
    

    @torch.no_grad()
    def sample_from_latent(self, z):
        self.eval()

        # Bisect method cant work with tensors on GPU
        means = self.means.to("cpu")
        logscales = self.logscales.to("cpu")
        mixing_coefs = self.mixing_coefs.to("cpu")

        x_samples = []
        for i, z_sample in enumerate(z):
            # f: R -> R
            def f(x): 
                 # Computing the components of mixture of logistics CDF
                logistic_cdf_components = 1 / (1 + torch.exp(-(x - means) / torch.exp(logscales)))

                # Make sure that mixing coefficients sum to 1!
                normalized_mixing_coeffs = F.softmax(mixing_coefs, dim=0).unsqueeze(1)

                # Weighting the mixture components by mixing coefficients
                mixed_logistic_cdf = torch.matmul(logistic_cdf_components, normalized_mixing_coeffs)

                # Finding the inverse of CDF at z is equivalent to finding the root of CDF - z
                return mixed_logistic_cdf.item() - z_sample
            
            x_sample = bisect(f, a=-10, b=10, xtol=1e-5, maxiter=50)
            x_samples.append(x_sample)
        
        x_samples = torch.tensor(x_samples, device=torch.device("cpu")).view(z.size(0), -1)

        self.train()
        return x_samples


    @torch.no_grad()
    def sample(self, num_samples):
        self.eval()

        # Sample zs from Beta distribution
        z_samples = Beta.rvs(self.alpha, self.beta, size=num_samples)

        # Converting to tensor and adding a batch dimension
        z_samples = torch.tensor(z_samples, device=torch.device("cpu")).unsqueeze(1)
        x_samples = self.sample_from_latent(z_samples)

        self.train()
        return x_samples



if __name__ == "__main__":
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    # Creating the model
    model = Flow1D().to(device)

    # Doing sampling sanity check
    # print(model.sample(num_samples=12))




    