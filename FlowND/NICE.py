import torch
import torch.nn as nn
import torch.nn.functional as F


# Prior for NICE, see section 3.4 of NICE paper for details
class StandardLogistic(torch.distributions.Distribution):
    def __init__(self):
        super().__init__(validate_args=False)

    def log_prob(self, x):
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):
        z = torch.distributions.Uniform(0., 1.).sample(size)
        return torch.log(z) - torch.log(1. - z)


# Implementing an additive coupling layer, see section 3.2 of NICE paper for details
class AdditiveCouplingLayer(nn.Module):
    # input_dim, hidden_dim, num_layers belong to neural net m specification
    # partition = even: even indices in the image (flattened) are passed unchanged: I1 = even indices
    # partition = odd: odd indices in the image (flattened) are passed unchanged: I1 = odd indices
    def __init__(self, input_dim, hidden_dim, num_layers, partition):
        super().__init__()

        # Creating the Neural Network m
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim) )
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.m = nn.Sequential(*layers)

        self.partition = partition

    
    # y_I1 = x_I1
    # y_I2 = x_I2 - m(x_I1)
    def forward(self, x):
        # Extracting corresponing half of the input elements
        x_I1, x_I2 = self._split(x)

        # Applying the additive flow transformation
        y_I1 = x_I1
        y_I2 = x_I2 + self.m(x_I1)

        # Putting the output elements into corresponding indices
        y = self._combine(y_I1, y_I2)
        return y
    

    # x_I1 = y_I1
    # x_I2 = y_I2 - m(y_I1)
    @torch.no_grad()
    def inverse(self, y):
        self.eval()

        # Extracting corresponing half of the output elements
        y_I1, y_I2 = self._split(y)
    
        # Applying the additive flow transformation
        x_I1 = y_I1
        x_I2 = y_I2 - self.m(y_I1)

        # Putting the output elements into corresponding indices
        x = self._combine(x_I1, x_I2)

        self.train()
        return x
    
    
    # Extracts even and odd indices from a tensor in a corresponding order
    def _split(self, tensor):
        if self.partition == 'even':
            return tensor[:, 0::2], tensor[:, 1::2]
        elif self.partition == "odd":
            return tensor[:, 1::2], tensor[:, 0::2]


    # Extracts even and odd indices in a corresponding order to form a tensor
    def _combine(self, tensor_I1, tensor_I2):
        tensor = torch.zeros_like(tensor_I1.new_empty(tensor_I1.size(0), tensor_I1.size(1) * 2))
        if self.partition == 'even':
            tensor[:, 0::2], tensor[:, 1::2] = tensor_I1, tensor_I2
        elif self.partition == "odd":
            tensor[:, 1::2], tensor[:, 0::2] = tensor_I1, tensor_I2
        return tensor
    


class NICE(nn.Module):
    def __init__(self, num_coupling_layers, input_dim, hidden_dim, num_m_layers, prior):
        super().__init__()
        self.prior = prior

        # Creating a network of coupling layers
        partition = "odd"
        self.flow = nn.ModuleList()
        for i in range(num_coupling_layers):
            self.flow.append(AdditiveCouplingLayer(
                input_dim=int(input_dim / 2),
                hidden_dim=hidden_dim,
                num_layers=num_m_layers,
                partition=partition)
            )
            partition = "even" if partition == "odd" else "odd"

        self.input_dim = input_dim

        # Initialized to 0 because it will be exponentiated
        self.log_scale = nn.Parameter(torch.zeros(input_dim, requires_grad=True))


    def forward(self, x):
        z = x
        for coupling_layer in self.flow:
            z = coupling_layer(z)
    
        # Performing diagonal scaling to increase expressiveness 
        z = torch.exp(self.log_scale) * z
        return z
    

    def get_loss(self, x):
        # See NICE paper section 3.3 for details
        z = self(x)
        loss = self.prior.log_prob(z).sum(dim=1) + torch.sum(torch.abs(self.log_scale))
        return loss
    

    @torch.no_grad()
    def sample_from_latent(self, z):
        self.eval()

        x = z
        for coupling_layer in reversed(self.flow):
            x = coupling_layer.inverse(x)
        
        self.train()
        return x
    
    
    @torch.no_grad()
    def sample(self, num_samples):
        self.eval()

        z = self.prior.sample((num_samples, self.input_dim)).to(self.log_scale.device)
        x = self.sample_from_latent(z)

        self.train()
        return x
    






