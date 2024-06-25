import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt


class RateReadout(nn.Module):
    """
    Rate readout layer for Poisson models. The latent state z is read out as:\\
    r = exp(Wz+b) * r0\\
    """
    def __init__(self, input_features, output_features):
        super(RateReadout, self).__init__()
        self.linear = nn.Linear(input_features, output_features, bias=True)
        # Initialize the rate vector
        self.rate_vector = nn.Parameter(torch.ones(output_features))

    def forward(self, z):
        return torch.exp(self.linear(z)) * self.rate_vector

# The LSTM-based RNN model
class RNNDynamicsModel(nn.Module):
    def __init__(self, rnn_type, model_type, feature_data_dim, hidden_size, num_layers, target_data_dim, n_traj, device=None):
        """
        Initialize the dynamics model as an RNN with the given parameters.
        
        Parameters:
        ----------
        rnn_type : str 
            the type of RNN to use (RNN, LSTM, GRU)
        model_type : str
            the type of model to parametrise using the RNN. Can be \\
            'gaussian', in which case the output is the mean and variance of the distribution \\
            'regression' in which case the output is the mean only \\
            'poisson' in which case the output is the rate parameter of a Poisson distribution
        feature_data_dim : int
            total dimensionality of the input data (features)
        hidden_size : int
            dimensionality of of the hidden state
        num_layers : int
            the number of layers in the RNN
        target_data_dim: int
            dimensionality of the target variable
        device : torch.device
            the device to run the model on (default is None, in which case the model is run on the GPU if available, otherwise the CPU)

        Returns:
        -------
        None
        """
        
        super(RNNDynamicsModel, self).__init__()
        
        # set up the device on whuch to run the model
        assert device is None or isinstance(device, torch.device), "device must be a torch.device object or None"
        
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # set up the RNN section (could be RNN, LSTM, or GRU, and could have multiple layers)
        assert rnn_type in ['RNN', 'LSTM', 'GRU'], "rnn_type must be either 'RNN', 'LSTM' or 'GRU'"
        assert isinstance(feature_data_dim, int), "feature_data_dim must be an integer"
        assert isinstance(hidden_size, int), "hidden_size must be an integer"
        assert isinstance(num_layers, int), "num_layers must be an integer"
        assert hidden_size > 0, "hidden_size must be a positive integer"
        assert num_layers > 0, "num_layers must be a positive integer"
        self.feature_data_dim = feature_data_dim
        
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(feature_data_dim, hidden_size, batch_first=True, num_layers=num_layers)
        
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(feature_data_dim, hidden_size, batch_first=True, num_layers=num_layers)
       
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(feature_data_dim, hidden_size, batch_first=True, num_layers=num_layers)

        # set up the initial hidden state for each trajectory
        self.h_0 = nn.Parameter(torch.zeros(n_traj, num_layers, hidden_size))

        # set up the model type
        assert model_type in ['gaussian', 'regression', 'poisson'], "model_type must be either 'gaussian','regression', or 'poisson'"
        self.model_type = model_type
        # set up the readout layer according to the model type
        assert isinstance(target_data_dim, int), "output_size must be an integer"
        assert feature_data_dim >= target_data_dim, "dimensionality of target variable must be at least as large as the input variable"
        self.target_data_dim = target_data_dim
        
        if model_type == 'gaussian':
            output_size = target_data_dim*2
            self.readout = nn.Linear(hidden_size, output_size)
        
        elif model_type == 'regression':
            output_size = target_data_dim
            self.readout = nn.Linear(hidden_size, output_size)
        
        elif model_type == 'poisson':
            output_size = target_data_dim
            self.readout = RateReadout(hidden_size, output_size)

        # finally, move the model to the device
        self.to(self.device)

    def forward(self, x, traj_idx):
        """
        Forward pass through the model.
        """
        out, _ = self.rnn(x, h_0 = self.h_0[traj_idx])
        out = self.readout(out)
        return out
    

def data_loader(device, feature_tensor, target_tensor, batch_size):
    """
    Moves the data to the GPU and creates a DataLoader object.
    """
    # Ensure tensors are contiguous
    features_contiguous = feature_tensor.contiguous()
    targets_contiguous  = target_tensor.contiguous()
    
    # Move tensors to GPU
    features_gpu    = features_contiguous.to(device)
    targets_gpu     = targets_contiguous.to(device)
    
    # Create DataLoader object
    train_dataset   = TensorDataset(features_gpu, targets_gpu)
    train_loader    = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    return train_loader

def data_loader_direct(
        device, 
        feature_array: np.ndarray, 
        target_array: np.ndarray, 
        batch_size: int
        ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    
    """
    Create a list representing the batches of data to be fed to the model.
    If the complete dataset fits into GPU VRAM, this function speeds up training considerably, compared to using DataLoader.
    """

    assert isinstance(feature_array, np.ndarray), "feature_array must be a numpy array"
    assert isinstance(target_array, np.ndarray), "target_array must be a numpy array"
    assert feature_array.shape[0] == target_array.shape[0], "feature_array and target_array must have the same number of timesteps"

    # ensure tensors are contiguous in memory
    features_contiguous = torch.tensor(feature_array, dtype=torch.float32).contiguous()
    targets_contiguous  = torch.tensor(target_array, dtype=torch.float32).contiguous()

    num_timesteps = feature_array.shape[0]
    sequences = []
    
    for start_idx in range(0, num_timesteps, batch_size):
        end_idx = min(start_idx + batch_size, num_timesteps)
        feature_seq   = features_contiguous[start_idx:end_idx].contiguous().to(device)
        target_seq    = targets_contiguous[start_idx:end_idx].contiguous().to(device)
        sequences.append((feature_seq, target_seq))
    
    return sequences

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
    mus = predicted[:, 0::2]                        # Even indexed outputs: means
    sigmas_diag_log = predicted[:, 1::2]            # Odd indexed outputs: log variances
    sigmas_diag = torch.exp(predicted[:, 1::2])     # Odd indexed outputs: variances, ensuring positivity
    
    determinant_term    = torch.sum(sigmas_diag_log, dim=1)
    quadratic_term      = torch.sum(torch.pow(target - mus, 2) / sigmas_diag, dim=1)

    constant_term = 0.5 * target.shape[1] * torch.log(torch.tensor(2 * torch.pi))

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
    return torch.mean(torch.sum(-per_neuron_per_timestep_loss, dim=1))

"""
TRAINING FUNCTION
"""

def train_RNNDynamicsModel(
        model:          RNNDynamicsModel, 
        feature_array:  np.ndarray, 
        target_array:   np.ndarray, 
        batch_size = 100, 
        epochs = 1000, 
        learning_rate = 0.01, 
        plot = False
        ) -> tuple[RNNDynamicsModel, float]:
    """
    Train the RNN model on the given data.
    """
    
    assert isinstance(model, RNNDynamicsModel), "model must be an instance of RNNDynamicsModel"

    device = model.device

    # create the data loader after checking compatibility
    assert model.feature_data_dim == feature_array.shape[1], f"model has different input dimensionality ({model.feature_data_dim}) than the feature_array ({feature_array.shape[1]})"
    assert model.target_data_dim == target_array.shape[1], f"model has different output dimensionality ({model.target_data_dim}) than the target_array ({target_array.shape[1]})"
    
    train_loader = data_loader_direct(device, feature_array, target_array, batch_size)

    # define the loss function according to the model type
    if model.model_type == 'gaussian':
        criterion = gaussian_neg_log_likelihood
    
    elif model.model_type == 'regression':
        criterion = nn.MSELoss()

    elif model.model_type == 'poisson':
        criterion = poisson_neg_log_likelihood

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#optim.SGD(model.parameters(), lr=learning_rate)

    # main training loop
    training_start = time.time()
    last_print = training_start

    losses = np.zeros((epochs, int(np.ceil(feature_array.shape[0]/batch_size)))) # keep track of the loss
    
    for epoch in range(epochs):
        i = 0
        for features, targets in train_loader:
            # forward pass
            predicted = model(features)
            # get the loss
            loss = criterion(predicted, targets)
            # store the loss
            losses[epoch, i] = loss.item()
            # get the gradients
            optimizer.zero_grad()
            loss.backward()
            # update the weights
            optimizer.step()

            i += 1
        
        
        if (time.time() - last_print) > 0.5: # print regular updates for long training times
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {np.mean(losses[epoch, :])}         ', end='\r')
            last_print = time.time()

        if (epoch+1) % 10 == 0: # print every 10 epochs
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {np.mean(losses[epoch, :])}         ', end='\r')
            last_print = time.time()
    
    print()

    if plot:
        # plot loss over training
        plt.figure(figsize=(18, 6))
        plt.title('Train loss vs epoch')
        plt.plot(losses, alpha = 0.5)
        plt.plot(np.mean(losses, axis=1), 'k', linewidth=3, label = 'mean loss')
        plt.legend()
        plt.show()

    # calculate average loss in the last 10 epochs
    avg_loss = np.mean(losses[-10:])

    return model, avg_loss

"""
VISUALIZATION FUNCTIONS
"""

def get_means_variances(output):
    """
    Process the raw outputs of the model to separate the means and variances.

    :param output: A tensor of shape (t, 2k) containing interleaved predicted means and log variances.
    :return: Two tensors, one for the means and one for the variances.
    """
    # Separate means and log variances
    means = output[:, 0::2]  # Even indexed outputs: means
    log_variances = output[:, 1::2]  # Odd indexed outputs: log variances

    # Convert log variances to variances
    variances = torch.exp(log_variances)

    return means, variances

def plot_RNNDynamicsModel_pred(model, feature_array, target_array, plot_start = 0, plot_end = None, by_dim = False):
    """
    Plot the predictions of the RNN model on the given data.
    """
    assert isinstance(model, RNNDynamicsModel), "model must be an instance of RNNDynamicsModel"
    assert isinstance(feature_array, np.ndarray), "feature_array must be a numpy array"
    assert isinstance(target_array, np.ndarray), "target_array must be a numpy array"
    assert feature_array.shape[0] == target_array.shape[0], "feature_array and target_array must have the same number of timesteps"
    assert model.feature_data_dim == feature_array.shape[1], f"model has different input dimensionality ({model.feature_data_dim}) than the feature_array ({feature_array.shape[1]})"
    assert model.target_data_dim == target_array.shape[1], f"model has different output dimensionality ({model.target_data_dim}) than the target_array ({target_array.shape[1]})"

    device = model.device

    # move the data to the GPU
    feature_tensor = torch.tensor(feature_array, dtype=torch.float32).to(device)
    

    # run the model
    predicted = model(feature_tensor)

    # extract the results and move them to the CPU
    model_type = model.model_type
    if model_type == 'gaussian':
        predicted = predicted.view(-1, model.target_data_dim*2)
    elif model_type == 'regression':
        predicted = predicted.view(-1, model.target_data_dim)
    elif model_type == 'poisson':
        predicted = predicted.view(-1, model.target_data_dim)
    predicted = predicted.detach().cpu()

    # plot the results
    if plot_end is None:
        plot_end = feature_array.shape[0]

    if model_type == 'gaussian':
        means, variances = get_means_variances(predicted)
        stds = np.sqrt(variances)
        if by_dim == False:
        
            plt.figure(figsize=(18, 3))
            plt.title('Model vs target')
            plt.plot(means[plot_start:plot_end])
            plt.plot(target_array[plot_start:plot_end], linestyle='--')
            plt.show()
        elif by_dim == True:
            for i in range(model.target_data_dim):
                plt.figure(figsize=(18, 3))
                plt.title(f'Model vs target for dimension {i}')
                plt.plot(means[plot_start:plot_end, i])
                plt.fill_between(np.arange(plot_start, plot_end), means[plot_start:plot_end, i]-2*stds[plot_start:plot_end, i], means[plot_start:plot_end, i]+2*stds[plot_start:plot_end, i], alpha=0.5, label="95% CI")
                plt.plot(target_array[plot_start:plot_end, i], linestyle='--')
                plt.show()
    
    if model_type == 'regression':
        means = predicted
        if by_dim == False:
        
            plt.figure(figsize=(18, 3))
            plt.title('Model vs target')
            plt.plot(means[plot_start:plot_end])
            plt.plot(target_array[plot_start:plot_end], linestyle='--')
            plt.show()
        elif by_dim == True:
            for i in range(model.target_data_dim):
                plt.figure(figsize=(18, 3))
                plt.title(f'Model vs target for dimension {i}')
                plt.plot(means[plot_start:plot_end, i])
                plt.plot(target_array[plot_start:plot_end, i], linestyle='--')
                plt.show()

    if model_type == 'poisson':
        means = predicted
        if by_dim == False:
        
            plt.figure(figsize=(18, 3))
            plt.title('Model vs target')
            plt.plot(means[plot_start:plot_end])
            plt.plot(target_array[plot_start:plot_end], linestyle='--')
            plt.show()
        elif by_dim == True:
            for i in range(model.target_data_dim):
                plt.figure(figsize=(18, 3))
                plt.title(f'Model vs target for dimension {i}')
                plt.plot(means[plot_start:plot_end, i])
                plt.plot(target_array[plot_start:plot_end, i], linestyle='--')
                plt.show()