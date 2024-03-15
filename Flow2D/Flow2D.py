import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import bisect

# Implemeting 2d autoregressive flow.
# Assume that input random variables x1, x2 are given, we need to learn joing p(x1,x2)
# IDEA: p(x1,x2) = p(x1)p(x2|x1)
# 1) Construct a flow from x1 to z1 (flow is a mixture of logistics)
# 2) Constrcut a conditional flow from x2 to z2 given x1 (flow is a mixture of logistics), 
# use neural network that outputs the parameters of conditional flow given x1

class Flow2D(nn.Module):
    def __init__(self, num_mixture_components=5):
        super().__init__()

        self.num_mixture_components = 5

        # Representing flow from x1 to z1 using the mixture of logistics
        self.means_f1 = nn.Parameter(torch.randn(num_mixture_components))
        self.logscales_f1 = nn.Parameter(torch.randn(num_mixture_components))
        self.mixing_coefs_f1 = nn.Parameter(torch.randn(num_mixture_components))

        # Representing conditional flow from x2 to z2 using the mixture of logistics
        # with parameters given by the following NN
        self.conditional_flow_params = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=3*num_mixture_components)
        )


    def eval_logistic_cdf(self, x, means, logscales, mixing_coefs):
        # Computing the components of mixture of CDFs
        cdf_components = 1 / (1 + torch.exp(-(x - means) / torch.exp(logscales)))

        # Weighting the mixture components by mixing coefficients
        mixed_cdf = (cdf_components * mixing_coefs).sum(dim=1).unsqueeze(-1)

        return mixed_cdf
        

    def forward(self, x):
        # Weird cloning, reason: pytorch is unfriendly with slicing
        x1 = x[:, 0].unsqueeze(-1).clone()
        x2 = x[:, 1].unsqueeze(-1).clone()

        # 1) Flow x1 to z1 by evaluating the first mixed logistic CDF
        # Make sure that mixing coefficients sum to 1!
        mixing_coeffs_f1 = F.softmax(self.mixing_coefs_f1, dim=0)
        z1 = self.eval_logistic_cdf(x1, self.means_f1, self.logscales_f1, mixing_coeffs_f1)

        # 2) Flow x2 to z2 by:
        # -> a) Obtaining the parameters for conditional flow given x1
        cond_flow_params = self.conditional_flow_params(x1)

        means_f2 =          cond_flow_params[:, 0 * self.num_mixture_components : 1 * self.num_mixture_components]
        logscales_f2 =      cond_flow_params[:, 1 * self.num_mixture_components : 2 * self.num_mixture_components]
        mixing_coefs_f2 =   cond_flow_params[:, 2 * self.num_mixture_components : 3 * self.num_mixture_components]
        
        # -> b) Evaluationg the second mixed logistic CDF
        # Make sure that mixing coefficients sum to 1!
        mixing_coefs_f2 = F.softmax(mixing_coefs_f2, dim=1)
        z2 = self.eval_logistic_cdf(x2, means_f2, logscales_f2, mixing_coefs_f2)
    
        return (x1, x2), (z1, z2)
    

    def get_loss(self, x):
        x.requires_grad_(True)

        # Forward pass
        (x1, x2), (z1, z2) = self(x)

        # Jacobian
        df1_dx1 = torch.autograd.grad(outputs=z1, inputs=x1, grad_outputs=torch.ones_like(z1), retain_graph=True, create_graph=True)[0]
        df2_dx2 = torch.autograd.grad(outputs=z2, inputs=x2, grad_outputs=torch.ones_like(z2), retain_graph=True, create_graph=True)[0]

        # Assume that distribution over z is uniform [0,1] =>
        # p(z) = const => does not matter from optimization perspective!
        loss_uniform = 0.0
        loss_jacobian = -(torch.log(torch.abs(df1_dx1) + 1e-9) + torch.log(torch.abs(df2_dx2) + 1e-9)).mean()

        loss = loss_uniform + loss_jacobian
        return loss
    

    @torch.no_grad()
    def sample(self, num_samples, device=torch.device("cpu")):
        self.eval()
        self.to(device)

        x1_samples = []
        x2_samples = []

        z_samples = torch.rand(num_samples, 2, device=device)

        # Make sure that mixing coefficients sum to 1!
        mixing_coefs_f1 = F.softmax(self.mixing_coefs_f1, dim=0)
        
        # 1) Invert the flow from x1 to z1 to get x1
        for z1_sample in z_samples[:, 0]:
            x1_sample = self._invert_cdf(
                z1_sample, 
                means=self.means_f1,
                logscales=self.logscales_f1,
                mixing_coefs=mixing_coefs_f1
            )
            x1_samples.append(x1_sample)

        # 2) Given x1, compute the parameters of the conditional flow from x2 to z2
        # -> Need to make sure x1_samples and neural net self.conditional_flow_params are on same device!
        nn_device = model.conditional_flow_params[0].weight.device
        x1_samples = torch.tensor(x1_samples, device=nn_device).unsqueeze(-1)

        # -> Computing the parameters for conditional flow
        cond_flow_params = self.conditional_flow_params(x1_samples)

        # Moving tensors back to specified device
        x1_samples = x1_samples.to(device)
        cond_flow_params = cond_flow_params.to(device)

        # Extractign the parameters for conditional flow
        means_f2 =          cond_flow_params[:, 0 * self.num_mixture_components : 1 * self.num_mixture_components]
        logscales_f2 =      cond_flow_params[:, 1 * self.num_mixture_components : 2 * self.num_mixture_components]
        mixing_coefs_f2 =   cond_flow_params[:, 2 * self.num_mixture_components : 3 * self.num_mixture_components]

        # Make sure that mixing coefficients sum to 1!
        mixing_coefs_f2 = F.softmax(mixing_coefs_f2, dim=1)

        # 3) Invert the conditional flow from x2 to z2 to get x2
        for i, z2_sample in enumerate(z_samples[:, 1]):
            x2_sample = self._invert_cdf(
                z2_sample, 
                means=means_f2[i],
                logscales=logscales_f2[i],
                mixing_coefs=mixing_coefs_f2[i]
            )
            x2_samples.append(x2_sample)
        
        # Putting everything together
        x2_samples = torch.tensor(x2_samples, device=device).unsqueeze(-1)
        x_samples = torch.cat([x1_samples, x2_samples], dim=1)

        self.train()
        return x_samples


    def _invert_cdf(self, z_sample, means, logscales, mixing_coefs):
        # f: R -> R
        def f(x): 
            # Computing the components of mixture of logistics CDF
            cdf_components = 1 / (1 + torch.exp(-(x - means) / torch.exp(logscales)))

            # Weighting the mixture components by mixing coefficients
            mixed_cdf = (cdf_components * mixing_coefs).sum()

            # Finding the inverse of CDF at z is equivalent to finding the root of CDF - z
            return mixed_cdf.item() - z_sample
            
        x_sample = bisect(f, a=-10, b=10, xtol=1e-5, maxiter=50)
        return x_sample



        

# TODO: 
# 1) check the dimensions of z1, z2, etc: z1: 64x1
# 2) make sure broadcasting in eval_logistic_cdf works for conditional flow case (means_f2, ... are now batches!!!)
# 3) move this file to appropriate folder
# 4) implement sample from latent (flow inversion)
# 5) train on swiss roll dataset and visualize the learned density (plot todo) using 4)
    


if __name__ == "__main__":
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    # Creating fake data
    swiss_roll_data = torch.randn(64, 2, device=device)

    # Creating the model
    model = Flow2D().to(device)

    # Sanity checks
    # print(model(swiss_roll_data).size())
    # print(model.get_loss(swiss_roll_data))
    # print(model.sample(num_samples=10))



