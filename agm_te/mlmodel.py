import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from agm_te.dataset import DirectDataLoader

class InputEnbed(nn.Module):
    """
    Embed the input data into a space of the same dimensionality as the hidden state in the RNN.
    """
    def __init__(self, input_dim, output_dim):
        super(InputEnbed, self).__init__()
        #self.linear = nn.Linear(input_dim, input_dim, bias=True)
        self.linear2 = nn.Linear(input_dim, output_dim, bias=True)
    
    def forward(self, x):
        #return self.linear2(torch.pow(self.linear(x), 3))
        return self.linear2(x)

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

class MinimalReadout(nn.Module):
    """
    Minimal readout layer for regression models. The latent state z is read out as:\\
    r = Wz+b, where W is diagonal
    """
    def __init__(self, input_features, output_features):
        assert input_features == output_features, "input_features must be equal to output_features for the minimal readout layer"
        super(MinimalReadout, self).__init__()
        self.readout_vector = nn.Parameter(torch.ones(input_features))
        self.bias = nn.Parameter(torch.zeros(input_features))

    def forward(self, latent):
        return latent * self.readout_vector + self.bias

# The RNN model
class RNNDynamicsModel(nn.Module):
    def __init__(self, rnn_type, model_type, var_to_dim, var_from_dim, var_cond_dim, hidden_size, num_layers, device=None):
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
        

        """Input Section is responsible for embedding the input data into the latent space"""
        assert isinstance(hidden_size, int), "hidden_size must be an integer"
        assert hidden_size > 0, "hidden_size must be a positive integer"

        # embed the past of "var_to" into the latent space (this will be present in both models)
        assert isinstance(var_to_dim, int), "var_to_dim must be an integer"
        assert var_to_dim > 0, "var_to_dim must be a positive integer"
        self.var_to_dim = var_to_dim
        self.embed_var_to = InputEnbed(var_to_dim, hidden_size)
        
        # embed the past of "var_from" into the latent space (this will be present in the second model)
        assert isinstance(var_from_dim, int), "var_from_dim must be an integer"
        assert var_from_dim >= 0, "var_from_dim must be a non-negative integer"
        self.var_from_dim = var_from_dim
        if var_from_dim > 0:
            self.embed_var_from = InputEnbed(var_from_dim, hidden_size)
        else:
            self.embed_var_from = None

        # embed the past of "var_cond" into the latent space (this will be present in both models)
        assert isinstance(var_cond_dim, int), "var_cond_dim must be an integer"
        assert var_cond_dim >= 0, "var_cond_dim must be a non-negative integer"
        self.var_cond_dim = var_cond_dim
        if var_cond_dim > 0:
            self.embed_var_cond = InputEnbed(var_cond_dim, hidden_size)
        else:
            self.embed_var_cond = None


        """RNN Section is responsible for learning the dynamics of the system in the latent space"""
        assert rnn_type in ['RNN', 'LSTM', 'GRU'], "rnn_type must be either 'RNN', 'LSTM' or 'GRU'"
        assert isinstance(num_layers, int), "num_layers must be an integer"
        assert num_layers > 0, "num_layers must be a positive integer"
        
        if rnn_type == 'RNN':
            self.dynamicsmodel = nn.RNN(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
        
        elif rnn_type == 'LSTM':
            self.dynamicsmodel = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
       
        elif rnn_type == 'GRU':
            self.dynamicsmodel = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)


        """Readout Section is responsible for mapping the latent space to the target space"""
        assert model_type in ['gaussian', 'regression', 'poisson'], "model_type must be either 'gaussian','regression', or 'poisson'"
        self.model_type = model_type
        # set up the readout layer according to the model type
        if model_type == 'gaussian':
            output_size = var_to_dim*2
            if output_size != hidden_size:
                self.readout = nn.Linear(hidden_size, output_size)
            else:
                self.readout = MinimalReadout(hidden_size, output_size)
        
        elif model_type == 'regression':
            output_size = var_to_dim
            if output_size != hidden_size:
                self.readout = nn.Linear(hidden_size, output_size)
            else:
                self.readout = MinimalReadout(hidden_size, output_size)
        
        elif model_type == 'poisson':
            output_size = var_to_dim
            self.readout = RateReadout(hidden_size, output_size)


        # finally, move the model to the device
        self.to(self.device)

    def forward(self, var_to_past, var_from_past, var_cond_past):
        """
        Forward pass through the model.
        """
        latent = self.embed_var_to(var_to_past)
        if var_from_past is not None:
            latent += self.embed_var_from(var_from_past)
        if var_cond_past is not None:
            latent += self.embed_var_cond(var_cond_past)
        
        evolved_latent, _ = self.dynamicsmodel(latent)
        
        readout = self.readout(evolved_latent)
        
        return readout



def init_dynamicsmodels_from_loaders(
        data_1:         DirectDataLoader, 
        data_2:         DirectDataLoader,
        hidden_size, 
        rnn_type =      "RNN", 
        model_type =    "gaussian", 
        num_layers =    1, 
        ) ->            tuple[RNNDynamicsModel, RNNDynamicsModel]:
    """
    Create a dynamics model from a DirectDataLoader object.

    Parameters:
    ----------
    data_1 : DirectDataLoader
        the data loader object that contains the data to train the first model
    data_2 : DirectDataLoader
        the data loader object that contains the data to train the second model
    rnn_type : str 
        the type of RNN to use (RNN, LSTM, GRU)
    model_type : str
        the type of model to parametrise using the RNN. Can be \\
        'gaussian', in which case the output is the mean and variance of the distribution \\
        'regression' in which case the output is the mean only \\
        'poisson' in which case the output is the rate parameter of a Poisson distribution
    hidden_size : int
        dimensionality of of the hidden state
    num_layers : int
        the number of layers in the RNN
    device : torch.device
        the device to run the model on (default is None, in which case the model is run on the GPU if available, otherwise the CPU)

    Returns:
    -------
    RNNDynamicsModel
        the dynamics model
    """

    assert isinstance(data_1, DirectDataLoader), "data must be an instance of DirectDataLoader"
    assert isinstance(data_2, DirectDataLoader), "data must be an instance of DirectDataLoader"
    assert data_1.device == data_2.device, "data_1 and data_2 must be on the same device"

    assert data_1.var_to_dim == data_2.var_to_dim, "data_1 and data_2 must have the same target variable dimension"
    assert data_1.var_cond_dim == data_2.var_cond_dim, "data_1 and data_2 must have the same conditional variable dimension"
    assert data_1.var_from_dim == 0, "data_1 must have no var_from dimension"
    assert data_2.var_from_dim > 0, "data_2 must have a var_from dimension"

    model_1 = RNNDynamicsModel(
        rnn_type = rnn_type,
        model_type = model_type,
        var_to_dim = data_1.var_to_dim,
        var_from_dim = 0,
        var_cond_dim = data_1.var_cond_dim,
        hidden_size = hidden_size,
        num_layers = num_layers,
        device = data_1.device
    )

    model_2 = RNNDynamicsModel(
        rnn_type = rnn_type,
        model_type = model_type,
        var_to_dim = data_2.var_to_dim,
        var_from_dim = data_2.var_from_dim,
        var_cond_dim = data_2.var_cond_dim,
        hidden_size = hidden_size,
        num_layers = num_layers,
        device = data_2.device
    )

    return model_1, model_2

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
    mus = predicted[:,:, 0::2]                        # Even indexed outputs: means
    sigmas_diag_log = predicted[:,:, 1::2]            # Odd indexed outputs: log variances
    sigmas_diag = torch.exp(predicted[:,:, 1::2])     # Odd indexed outputs: variances, ensuring positivity
    
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
TRAINING FUNCTION
"""

def train_RNNDynamicsModel(
        model_1:        RNNDynamicsModel,
        data_1:         DirectDataLoader,
        model_2:        RNNDynamicsModel,
        data_2:         DirectDataLoader,
        epochs =        1000, 
        learning_rate = 0.001,
        l2_penalty =    0.0,
        noise_std =     0.0,
        plot_loss =     False,
        loss_history =  False,
        optimize =      'adam'
        ):
    """
    Train the RNN model on the given data.
    """
    
    assert isinstance(model_1, RNNDynamicsModel), "model_1 must be an instance of RNNDynamicsModel"
    assert isinstance(model_2, RNNDynamicsModel), "model_2 must be an instance of RNNDynamicsModel"
    assert model_1.model_type == model_2.model_type, "model_1 and model_2 must have the same model type"
    assert model_1.device == model_2.device, "model_1 and model_2 must be on the same device"
    assert model_1.var_to_dim == model_2.var_to_dim, "model_1 and model_2 must have the same target variable dimension"
    assert model_1.var_cond_dim == model_2.var_cond_dim, "model_1 and model_2 must have the same conditional variable dimension"
    assert model_1.var_from_dim == 0, "model_1 must have no var_from dimension"
    assert model_2.var_from_dim > 0, "model_2 must have a var_from dimension"

    assert isinstance(data_1, DirectDataLoader), "data_1 must be an instance of DirectDataLoader"
    assert isinstance(data_2, DirectDataLoader), "data_2 must be an instance of DirectDataLoader"

    assert isinstance(epochs, int), "epochs must be an integer"
    assert isinstance(learning_rate, float), "learning_rate must be a float"
    assert isinstance(plot_loss, bool), "plot must be a boolean"
    assert isinstance(loss_history, bool), "loss_history must be a boolean"
    assert optimize == 'adam' or optimize == 'sgd', "optimize must be either 'adam' or 'sgd'"

   

    # define the loss function according to the model type
    if model_1.model_type == 'gaussian':
        criterion = gaussian_neg_log_likelihood
    
    elif model_1.model_type == 'regression':
        criterion = nn.MSELoss()

    elif model_1.model_type == 'poisson':
        criterion = poisson_neg_log_likelihood

    # define the optimizers
    if optimize == 'adam':
        optimizer_1 = optim.Adam(model_1.parameters(), lr=learning_rate, weight_decay=l2_penalty)
        optimizer_2 = optim.Adam(model_2.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    elif optimize == 'sgd':
        optimizer_1 = optim.SGD(model_1.parameters(), lr=learning_rate, weight_decay=l2_penalty)
        optimizer_2 = optim.SGD(model_2.parameters(), lr=learning_rate, weight_decay=l2_penalty)

    # main training loop
    training_start = time.time()
    last_print = training_start

    losses_1 = np.zeros((epochs+1, len(data_1))) # keep track of the loss over epochs and batches of the first model
    losses_2 = np.zeros((epochs+1, len(data_2))) # keep track of the loss over epochs and batches of the second model
    
    alternate = 50

    # iterate over the epochs
    for epoch in range(0, epochs, alternate):
        # copy over the parameters for from the first model to the second model
        if epoch > alternate:
            # copy readout 
            model_1.readout.load_state_dict(model_2.readout.state_dict())
            # copy var to embedding
            model_1.embed_var_to.load_state_dict(model_2.embed_var_to.state_dict())
            # copy conditioning variable embedding
            if model_1.embed_var_cond is not None:
                model_1.embed_var_cond.load_state_dict(model_2.embed_var_cond.state_dict())
            optimizer_1.zero_grad() # zero the gradients of the first model
        
        # iterate over the data sequences for the first model
        for j in range(alternate):
            i = 0
            for features_var_to, features_var_from, features_var_cond, target_var_to in data_1:
                # forward pass
                predicted = model_1(var_to_past=features_var_to, var_from_past=features_var_from, var_cond_past=features_var_cond)
                # get the loss
                loss = criterion(predicted, target_var_to + noise_std*torch.randn_like(target_var_to))
                # store the loss
                losses_1[epoch+j, i] = loss.item()
                # get the gradients
                optimizer_1.zero_grad()
                loss.backward()
                # update the weights
                optimizer_1.step()
                i += 1

        running_loss_1 = np.round(np.mean(losses_1[:epoch+alternate, :]),6)

        # copy over the parameters for from the first model to the second model
        if epoch > alternate:
            # copy readout 
            model_2.readout.load_state_dict(model_1.readout.state_dict())
            # copy var to embedding
            model_2.embed_var_to.load_state_dict(model_1.embed_var_to.state_dict())
            # copy conditioning variable embedding
            if model_2.embed_var_cond is not None:
                model_2.embed_var_cond.load_state_dict(model_1.embed_var_cond.state_dict())
            optimizer_2.zero_grad() # zero the gradients of the second model

        # iterate over the data sequences for the second model
        for j in range(alternate):
            i = 0
            for features_var_to, features_var_from, features_var_cond, target_var_to in data_2:
                # forward pass
                predicted = model_2(var_to_past=features_var_to, var_from_past=features_var_from, var_cond_past=features_var_cond)
                # get the loss
                loss = criterion(predicted, target_var_to + noise_std*torch.randn_like(target_var_to))
                # store the loss
                losses_2[epoch+j, i] = loss.item()
                # get the gradients
                optimizer_2.zero_grad()
                loss.backward()
                # update the weights
                optimizer_2.step()
                i += 1

        running_loss_2 = np.round(np.mean(losses_2[:epoch+alternate, :]),6)

        running_TE = np.round(running_loss_1 - running_loss_2, 4)
        
        if ((time.time() - last_print) > 0.5) or (epoch % 100 == 0): # print regular updates for long training times or every 100 epochs
            print(f'Epoch [{epoch}/{epochs}], Model 1 Loss: {running_loss_1}, Model 2 Loss: {running_loss_2}, Estimated TE: {running_TE}', end='\r')
            last_print = time.time()
    print(f'Epoch [{epochs}/{epochs}], Model 1 Loss: {running_loss_1}, Model 2 Loss: {running_loss_2}, Estimated TE: {running_TE}', end='\r')

    print()

    if plot_loss:
        # plot loss over training
        plt.figure(figsize=(18, 3))
        plt.title('Train loss vs epoch')
        plt.plot(losses_1, alpha = 0.5, label = 'model 1 loss')
        plt.plot(losses_2, alpha = 0.5, label = 'model 2 loss')
        plt.legend()
        plt.show()
    
    if loss_history: # return the loss at each epoch
        loss_return = np.round(np.array(losses_1) - np.array(losses_2),4)
    elif not loss_history:
        loss_return = np.round(running_TE, 4)

    return model_1, model_2, loss_return

def get_means_variances_torch(output:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

def predict_with_RNNDynamicsModel(
        model: RNNDynamicsModel,
        var_from: np.ndarray,
        var_to_start: np.ndarray,
        var_cond: np.ndarray = None,
        random_rep: int = 0,
        random_noise: float = 0.00
        ):
    """
    Use a learnt dynamics model to perform simulations of counterfactual trajectories.
    """
    assert isinstance(model, RNNDynamicsModel), "model must be an instance of RNNDynamicsModel"
    assert random_rep == 0 or random_rep > 10, "random_rep must be 0 (deterministic operation) or greater than 10 for sufficient samples for stochastic operation"
    assert model.model_type == 'gaussian', "simulations only implemented for gaussian models"

    var_from = torch.tensor(var_from).to(model.device)
    var_to_past = torch.zeros((var_from.shape[0], var_to_start.shape[1])).to(model.device)
    var_to_past[:var_to_start.shape[0],:] = torch.tensor(var_to_start).to(model.device)
    var_cond = torch.tensor(var_cond).to(model.device) if var_cond is not None else None

    # reset the hidden state of the model
    
    with torch.no_grad():
        # deterministic operation, always take the mean of the distribution as the next value. Always gives the same result
        if random_rep == 0:
            model.dynamicsmodel.hidden_state = None
            for i in range(var_to_start.shape[0], var_from.shape[0]):
                var_from_past = var_from[:i]
                if var_cond is not None:
                    var_cond_past = var_cond[:i]
                else:
                    var_cond_past = None
                predicted = model(var_to_past[:i], var_from_past, var_cond_past)
                means, variances = get_means_variances_torch(predicted)
                var_to_past[i] = means[-1]

            pred = var_to_past.detach().cpu().numpy()
            return pred
        # stochastic operation, sample from the distribution at each timestep. Gives different results each time
        else:
            res_holder = np.zeros((random_rep, var_to_past.shape[0], var_to_past.shape[1]))
            for r in range(random_rep):
                model.dynamicsmodel.hidden_state = None
                for i in range(var_to_start.shape[0], var_from.shape[0]):
                    var_from_past = var_from[:i]
                    if var_cond is not None:
                        var_cond_past = var_cond[:i]
                    else:
                        var_cond_past = None
                    predicted = model(var_to_past[:i], var_from_past, var_cond_past)
                    means, variances = get_means_variances_torch(predicted)
                    sample = torch.normal(means[-1, :], torch.sqrt(variances[-1, :])) + torch.randn_like(means[-1, :])*random_noise
                    var_to_past[i] = sample

                temp_res = var_to_past.detach().cpu().numpy()
                res_holder[r, :, :] = temp_res

            pred_mean = np.mean(res_holder, axis=0)
            pred_std = np.std(res_holder, axis=0)
            return pred_mean, pred_std



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
    variances = np.exp(log_variances)

    return means, variances

def plot_RNNDynamicsModel_pred(model:RNNDynamicsModel, dataloader:DirectDataLoader, batch_index, traj_idx, plot_start = 0, plot_end = None, by_dim = False):
    """
    Plot the predictions of the RNN model on the given data.
    """
    assert isinstance(model, RNNDynamicsModel), "model must be an instance of RNNDynamicsModel"
    assert isinstance(dataloader, DirectDataLoader), "dataloader must be an instance of DirectDataLoader"
    assert model.var_to_dim == dataloader.var_to_dim, "model and dataloader must have the same target variable dimension"
    assert model.var_cond_dim == dataloader.var_cond_dim, "model and dataloader must have the same conditional variable dimension"
    assert model.var_from_dim == dataloader.var_from_dim, "model and dataloader must have the same var_from dimension"

    
    # get the features
    features_var_to = dataloader.data[batch_index][0][traj_idx]
    if model.var_from_dim > 0:
        features_var_from = dataloader.data[batch_index][1][traj_idx]
    else:
        features_var_from = None
    if model.var_cond_dim > 0:
        features_var_cond = dataloader.data[batch_index][2][traj_idx]
    else:
        features_var_cond = None
    # run and extract the predictions
    predicted = model(features_var_to, features_var_from, features_var_cond)
    predicted = predicted.detach().cpu().numpy()

    # get the target array
    target_array = dataloader.data[batch_index][3][traj_idx].detach().cpu().numpy()

    # plot the results
    if plot_end is None:
        plot_end = target_array.shape[0]

    model_type = model.model_type
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
            for i in range(model.var_to_dim):
                ci_low = means[plot_start:plot_end, i]-2*stds[plot_start:plot_end, i]
                ci_high = means[plot_start:plot_end, i]+2*stds[plot_start:plot_end, i]
                plt.figure(figsize=(18, 3))
                plt.title(f'Model vs target for dimension {i}')
                plt.plot(means[plot_start:plot_end, i])
                plt.fill_between(np.arange(plot_start, plot_end), ci_low, ci_high, alpha=0.5, label="95% CI")
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
            for i in range(model.var_to_dim):
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
            for i in range(model.var_to_dim):
                plt.figure(figsize=(18, 3))
                plt.title(f'Model vs target for dimension {i}')
                plt.plot(means[plot_start:plot_end, i])
                plt.plot(target_array[plot_start:plot_end, i], linestyle='--')
                plt.show()