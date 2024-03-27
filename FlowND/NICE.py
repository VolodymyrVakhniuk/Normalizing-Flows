import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

        # print(x_I1.size())

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


    # Assumes x is flat
    def forward(self, x):
        z = x
        for coupling_layer in self.flow:
            z = coupling_layer(z)
    
        # Performing diagonal scaling to increase expressiveness 
        z = torch.exp(self.log_scale) * z
        return z
    

    def get_loss(self, x):
        # See NICE paper section 3.3 for details
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        z = self(x)
        loss = -(self.prior.log_prob(z).sum(dim=1) + torch.sum(self.log_scale)).mean()

        return loss
    

    @torch.no_grad()
    def sample_from_latent(self, z):
        self.eval()

        # Inverse of forward
        x = z / torch.exp(self.log_scale)
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
        return x.view(-1, 1, 28, 28)



if __name__ == "__main__":
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    # Creating fake data
    image_batch = torch.randn(10, 1, 28, 28, device=device)

    # Creating the model
    prior = StandardLogistic()
    model = NICE(num_coupling_layers=5, input_dim=28*28, hidden_dim=256, num_m_layers=3, prior=StandardLogistic()).to(device)


    z = model(image_batch.view(image_batch.size(0), -1))
    image_batch_inv = model.sample_from_latent(z)

    with np.printoptions(formatter={'float': '{:0.5f}'.format}):
        print(image_batch[0, :, 0].cpu().detach().numpy())
        print(image_batch_inv.view(-1, 1, 28, 28)[0, :, 0].cpu().detach().numpy())
    # Sanity checks
    # print(model.get_loss(image_batch))
    # print(model.sample(num_samples=10).size())
    # print(model.sample(num_samples=10))




# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # DO NOT CHANGE THIS CELL
# class StandardLogistic(torch.distributions.Distribution):
#     """Standard logistic distribution.
#     """
#     def __init__(self):
#         super(StandardLogistic, self).__init__(validate_args=False)

#     def log_prob(self, x):
#         """Computes data log-likelihood.
#         Args:
#             x: input tensor.
#         Returns:
#             log-likelihood.
#         """
#         return -(F.softplus(x) + F.softplus(-x))

#     def sample(self, size):
#         """Samples from the distribution.
#         Args:
#             size: number of samples to generate.
#         Returns:
#             samples.
#         """
#         z = torch.distributions.Uniform(0., 1.).sample(size)
#         return torch.log(z) - torch.log(1. - z)

# class AdditiveCouplingLayer(nn.Module): 
#     def __init__(self, input_dim, hidden_dim, num_layers, partition):
#         super().__init__()
#         assert partition in ['odd', 'even']
#         self.partition = partition
#         _get_even = lambda xs: xs[:, 0::2]
#         _get_odd = lambda xs: xs[:, 1::2]
#         if (partition == 'even'):
#             self._first = _get_even
#             self._second = _get_odd
#         else:
#             self._first = _get_odd
#             self._second = _get_even
        
#         _modules = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
#         for _ in range(num_layers):
#             _modules.append(nn.Linear(hidden_dim, hidden_dim) )
#             _modules.append(nn.ReLU())
#         _modules.append(nn.Linear(hidden_dim, input_dim) )
#         self.net = nn.Sequential(*_modules)

    
#     def forward(self, x):
#         """Map an input through the partition and nonlinearity.
#         y1 = x1
#         y2 = x2 + m(x1)
#         """
#         # YOUR CODE HERE
#         x_first = self._first(x)
#         x_second = self._second(x)
#         # print(x_first.size())
#         m_x_first = self.net(x_first)

#         y_first = x_first
#         y_second = x_second + m_x_first

#         out = torch.zeros_like(x)
#         if self.partition == "even":
#           out[:, 0::2] = y_first
#           out[:, 1::2] = y_second
#         elif self.partition == "odd":
#           out[:, 1::2] = y_first
#           out[:, 0::2] = y_second

#         return out

#     def inverse(self, y):
#         """Inverse mapping through the layer. Gradients should be turned off for this pass.
#         x1 = y1
#         x2 = y2 - m(y1)
#         """
#         # YOUR CODE HERE
#         y_first = self._first(y)
#         y_second = self._second(y)
#         m_y_first = self.net(y_first)

#         x_first = y_first
#         x_second = y_second - m_y_first

#         out = torch.zeros_like(y)
#         if self.partition == "even":
#           out[:, 0::2] = x_first
#           out[:, 1::2] = x_second
#         elif self.partition == "odd":
#           out[:, 1::2] = x_first
#           out[:, 0::2] = x_second

#         return out



# class NICE(nn.Module):
#     def __init__(self, num_coupling_layers, input_dim, hidden_dim, num_m_layers, prior):
#         super(NICE, self).__init__()
#         assert (input_dim % 2 == 0)
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         print("DSNKNAIDSN")
#         # define network
#         # YOUR CODE HERE

#         # Note: self.hidden_dim and num_m_layers refers to the neural network embedded in the coupling layer;
#         # The input and output of the coupling layer have the same dimensionality! namely - input_dim;
#         self.coupling_1 = AdditiveCouplingLayer(int(self.input_dim / 2), self.hidden_dim, num_m_layers, partition="odd")
#         self.coupling_2 = AdditiveCouplingLayer(int(self.input_dim / 2), self.hidden_dim, num_m_layers, partition="even")
#         self.coupling_3 = AdditiveCouplingLayer(int(self.input_dim / 2), self.hidden_dim, num_m_layers, partition="odd")
#         self.coupling_4 = AdditiveCouplingLayer(int(self.input_dim / 2), self.hidden_dim, num_m_layers, partition="even")

#         # Initialized to 0 because it will be exponentiated
#         self.s = nn.Parameter(torch.zeros(input_dim, requires_grad=True))

#         # self.scale = nn.Parameter(
#         #     torch.zeros((1, dim)), requires_grad=True)

#         self.prior = prior
#         self.register_buffer('_dummy', torch.empty([0, ]))


#     def forward(self, x):
#         """Forward pass through all invertible coupling layers.
#         Args:
#             x: Input data. float tensor of shape (batch_size, input_dim).
#         Returns:
#             z: Latent variable. float tensor of shape (batch_size, input_dim).
#         """
#         # YOUR CODE HERE
#         z_1 = self.coupling_1(x)
#         z_2 = self.coupling_2(z_1)
#         z_3 = self.coupling_3(z_2)
#         z_4 = self.coupling_4(z_3)
    
#         # Performing diagonal scaling;
#         # In order to save memory, we don't create diagonal matrix S directly;
#         # Instead, we "model" the multiplication by diagonal matrix by elementwise product with a vector s.
#         # Note: Evokes torch broadcasting to multiply every example in a batch by s!
#         z = torch.exp(self.s) * z_4
#         return z


#     def inverse(self, z):
#         """Invert a set of draws from logistic prior
#         Args:
#             z: Latent variable. float tensor of shape (batch_size, input_dim).
#         Returns:
#             x: Generated data. float tensor of shape (batch_size, input_dim).
#         """
#         with torch.no_grad():
#             # YOUR CODE HERE
#             z_4 = z / torch.exp(self.s)
            
#             z_3 = self.coupling_4.inverse(z_4)
#             z_2 = self.coupling_3.inverse(z_3)
#             z_1 = self.coupling_2.inverse(z_2)
#             x =   self.coupling_1.inverse(z_1)

#         return x
    

#     def log_prob(self, x):
#         """Computes data log-likelihood. (See Section 3.3 in the NICE paper.)
#         Args:
#             x: input minibatch.
#         Returns:
#             log_p: log-likelihood of input.
#         """
#         # YOUR CODE HERE
#         z = self.forward(x)
#         log_p = self.prior.log_prob(z).sum(dim=1) + torch.sum(self.s)
      
#         return log_p
    

#     def get_loss(self, x):
#         return -self.log_prob(x.view(x.size(0), -1)).mean()

    
#     def sample(self, num_samples):
#         """Generates samples.
#         Args:
#             num_samples: number of samples to generate.
#         Returns:
#             x: samples from the data space X.
#         """
#         z = self.prior.sample((num_samples, self.input_dim)).to(self._dummy.device)
#         # YOUR CODE HERE
#         x = self.inverse(z)
#         return x


