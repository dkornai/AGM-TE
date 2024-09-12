import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from agm_te.dataset import AgmTrainingData




# The RNN model
class ApproxGenModel(nn.Module):
    def __init__(self, dyn_model_type, obs_model_type, input_dim, target_dim, hidden_size, num_layers, timestep=None):
        """
        Initialize the dynamics model as an RNN with the given parameters.
        
        Parameters:
        ----------
        dyn_model_type :  str 
            the type of dynamics model to use (can be MLP, RNN, LSTM, GRU)
        obs_model_type : str
            the type of model for the observed data \\
            'gaussian', in which case the outputs are the mean and variance of a multivariate diagonal gaussian distribution \\
            'regression' in which case the output is directly the observed value \\
            'poisson' in which case the output is the rate parameters of a multivariate Poisson distribution
        input_dim : int
            total dimensionality of the input data (features)
        target_dim: int
            dimensionality of the target variable
        hidden_size : int
            dimensionality of of the hidden state
        num_layers : int
            the number of layers in the RNN
       
        Returns:
        -------
        None
        """
        super(ApproxGenModel, self).__init__()
    
        ### RNN Section is responsible for learning the dynamics of the system in the latent space ###
        assert isinstance(input_dim, int), "input_dim must be an integer"
        assert input_dim > 0, "input_dim must be a positive integer"
        self.input_dim = input_dim
        assert isinstance(hidden_size, int), "hidden_size must be an integer"
        assert hidden_size > 0, "hidden_size must be a positive integer"
        assert isinstance(target_dim, int), "target_dim must be an integer"
        assert target_dim <= input_dim, "target_dim must be less than or equal to input_dim"
        self.target_dim = target_dim
        assert dyn_model_type in ['MLP', 'MLPDiff', 'MLPTanh', 'RNN', 'LSTM', 'GRU'], "dyn_model_type must be either 'MLP', 'MLPDiff', 'MLPTanh', 'RNN', 'LSTM' or 'GRU'"
        assert isinstance(num_layers, int), "num_layers must be an integer"
        assert num_layers > 0, "num_layers must be a positive integer"
        
        if   dyn_model_type == 'MLP':
            self.dynamicsmodel = MLPReluDynamics(     input_dim, hidden_size, num_layers)
        elif dyn_model_type == 'MLPDiff':
            self.dynamicsmodel = MLPReluDiffDynamics( input_dim, hidden_size, num_layers, target_dim)
        elif dyn_model_type == 'MLPTanh':
            self.dynamicsmodel = MLPTanhDynamics(     input_dim, hidden_size, num_layers)
        elif dyn_model_type == 'RNN':
            self.dynamicsmodel = RNNDynamics(         input_dim, hidden_size, num_layers)
        elif dyn_model_type == 'LSTM':
            self.dynamicsmodel = LSTMDynamics(        input_dim, hidden_size, num_layers)
        elif dyn_model_type == 'GRU':
            self.dynamicsmodel = GRUDynamics(         input_dim, hidden_size, num_layers)

        ### Readout Section is responsible for mapping the latent space to the target space ###
        assert obs_model_type in ['gaussian', 'regression', 'poisson'], "obs_model_type must be either 'gaussian','regression', or 'poisson'"
        if obs_model_type == 'poisson':
            assert timestep is not None, "timestep must be provided for Poisson models"
            assert 0.0 < timestep, "timestep should be a float greater than 0"
        if timestep is None:
            assert obs_model_type != 'poisson', "timestep parameter is not required for gaussian or regression models"
        if dyn_model_type == 'MLPDiff':
            assert obs_model_type == 'regression', "MLPDiff model can only be used with regression output"
        self.obs_model_type = obs_model_type
        # set up the readout layer according to the model type
        if obs_model_type == 'gaussian':
            self.readout = nn.Linear(  hidden_size, target_dim*2)   # mean and variance for each target dimension
        
        elif obs_model_type == 'regression':
            if dyn_model_type == 'MLPDiff':
                self.readout = nn.Identity()    # the dynamics model already predicts the difference between the current and previous states
            else:
                self.readout = nn.Linear(  hidden_size, target_dim)     # direct prediction for each target dimension
        
        elif obs_model_type == 'poisson':
            self.readout = RateReadout(hidden_size, target_dim, timestep)     # rate parameter for each target dimension

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, input):
        """
        Forward pass through the model.
        """
        latent = self.dynamicsmodel(input)
        out = self.readout(latent)
        return out

class MLPTanhDynamics(nn.Module):
    """
    Simple multi-layer perceptron dynamics model.
    """
    def __init__(self, input_dim, hidden_size, num_layers):
        super(MLPTanhDynamics, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = F.tanh(layer(x))
        return self.output_layer(x)
    
class MLPReluDynamics(nn.Module):
    """
    Simple multi-layer perceptron dynamics model.
    """
    def __init__(self, input_dim, hidden_size, num_layers):
        super(MLPReluDynamics, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

class MLPReluDiffDynamics(nn.Module):
    """
    Simple multi-layer perceptron dynamics model, that predicts the difference between the current and previous states.
    """
    def __init__(self, input_dim, hidden_size, num_layers, target_dim):
        self.target_dim = target_dim
        super(MLPReluDiffDynamics, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, target_dim)

    def forward(self, x):
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = F.relu(layer(out))
        return self.output_layer(out) + x[...,:self.target_dim] # first target_dim elements are the previous state

class RNNDynamics(nn.Module):
    """
    Simple RNN dynamics model.
    """
    def __init__(self, input_dim, hidden_size, num_layers):
        super(RNNDynamics, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_size, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out
    
class LSTMDynamics(nn.Module):
    """
    Simple LSTM dynamics model.
    """
    def __init__(self, input_dim, hidden_size, num_layers):
        super(LSTMDynamics, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out
    
class GRUDynamics(nn.Module):
    """
    Simple GRU dynamics model.
    """
    def __init__(self, input_dim, hidden_size, num_layers):
        super(GRUDynamics, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        out, _ = self.gru(x)
        return out

class RateReadout(nn.Module):
    """
    Rate readout layer for Poisson models. The latent state z is read out as:\\
    r = exp(Wz+b) * r0
    """
    def __init__(self, input_features, output_features, timestep):
        super(RateReadout, self).__init__()
        self.linear = nn.Linear(input_features, output_features, bias=True)
        # Initialize the rate vector
        self.rate_vector = nn.Parameter(torch.ones(output_features))
        self.timestep = torch.tensor(timestep)

    def forward(self, z):
        return torch.exp(self.linear(z)) * self.rate_vector * self.timestep


"""
CUSTOM LOSS FUNCTIONS
"""

def gaussian_neg_log_likelihood(predicted:torch.Tensor, target:torch.Tensor):
    """
    Custom loss function that calculates the mean negative log likelihood of a multivariate diagonal gaussian model

    Parameters:
    ----------
    predicted : Tensor 
        tensor of shape (t, 2d) containing predicted means and variances.
    target : Tensor
        tensor of shape (t, d) representing the observed sequence 
    
    Returns:
    -------
    Tensor
        Mean negative log likelihood
    """
    # Separate means and variances from the predictions
    mus = predicted[:,:, 0::2]                   # Even indexed outputs: means
    sigmas_diag_log = predicted[:,:, 1::2]       # Odd indexed outputs: log variances
    sigmas_diag = torch.exp(sigmas_diag_log)     # Odd indexed outputs: variances, ensuring positivity
    
    determinant_term    = torch.sum(sigmas_diag_log, dim=-1)
    quadratic_term      = torch.sum(torch.pow(target - mus, 2) / sigmas_diag, dim=-1)

    constant_term = 0.5 * target.shape[-1] * torch.log(torch.tensor(2 * torch.pi))

    return constant_term + 0.5*torch.mean(determinant_term + quadratic_term)  # Return mean negative log likelihood

def poisson_neg_log_likelihood(predicted:torch.Tensor, target:torch.Tensor):
    """
    Custom loss function that calculates the mean negative log likelihood of a set of poissons

    Parameters:
    ----------
    predicted : Tensor 
        tensor of shape (t, d) containing predicted rates for the inhomogeneous Poisson process.
    target : Tensor
        tensor of shape (t, d) representing the observed sequence of event observations/timestep

    Returns:
    -------
    Tensor
        Mean negative log likelihood
    """
    per_neuron_per_timestep_loss = target*torch.log(predicted) - predicted + torch.lgamma(target + 1) # lgamma(n+1) = log(n!)
    return torch.mean(torch.sum(-per_neuron_per_timestep_loss, dim=-1))


"""
DATA LOADER FUNCTION
"""


def data_loader_direct(device, features, targets, batch_size):
    """
    When the complete dataset can be loaded into the GPU VRAM, this function is MUCH (10X + ) faster than DataLoader.
    """

    # Ensure tensors are contiguous
    features_contiguous = torch.tensor(features, dtype=torch.float32, device=device).contiguous()
    targets_contiguous  = torch.tensor(targets, dtype=torch.float32, device=device).contiguous()

    n_traj = features.shape[0]
    batches = []
    
    for start_idx in range(0, n_traj, batch_size):
        end_idx = min(start_idx + batch_size, n_traj)
        feature_batch   = features_contiguous[start_idx:end_idx].contiguous()
        target_batch    = targets_contiguous[start_idx:end_idx].contiguous()
        batches.append((feature_batch, target_batch))

    return batches

"""
TRAINING FUNCTION
"""

def _train_agm(
        model:              ApproxGenModel,
        data:               AgmTrainingData,
        batch_size =        1,
        epochs =            1000,
        learning_rate =     0.001,
        lr_decay_step =     100,
        lr_decay_gamma =    1.0,
        optimize =          'adam',
        l2_penalty =        0.0,
        ):
    
    # define the loss function according to the model type
    if   model.obs_model_type == 'gaussian':
        criterion = gaussian_neg_log_likelihood
    elif model.obs_model_type == 'regression':
        criterion = nn.MSELoss()
    elif model.obs_model_type == 'poisson':
        criterion = poisson_neg_log_likelihood

    # define the optimizers
    if optimize == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    elif optimize == 'sgd':
        optimizer = optim.SGD( model.parameters(), lr=learning_rate, weight_decay=l2_penalty)

    # Initialize the scheduler
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    # load the data into the device (typically GPU) memory
    training_data = data_loader_direct(model.get_device(), data.input, data.target, batch_size)

    losses = np.zeros((epochs, len(training_data))) # keep track of the loss over epochs and batches
    # iterate over the epochs
    for epoch in range(epochs):
        i = 0
        for features, targets in training_data:
            # forward pass
            predicted = model(features)
            # get the loss
            loss = criterion(predicted, targets)
            losses[epoch, i] = loss.item()
            # get the gradients
            optimizer.zero_grad()
            loss.backward()
            # update the weights
            optimizer.step()
            i += 1

        # decay the learning rate
        scheduler.step()

        # keep track of the running loss
        if epoch > 10:
            running_loss = np.mean(losses[epoch-10:epoch, :])

        # print updates
        if epoch > 10 and epoch % 10 == 0: # print updates every 10 epochs
            print(f'Epoch [{epoch}/{epochs}], Loss: {np.round(running_loss, 8)}'   , end='\r')

    # avarage across the batches to get a history of the loss over epochs
    loss_history = np.round(np.mean(losses, axis=1), 8).reshape(-1)

    return model, loss_history
     

