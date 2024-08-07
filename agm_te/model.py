import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from agm_te.dataset import AgmTrainingData




# The RNN model
class ApproxGenModel(nn.Module):
    def __init__(self, rnn_type, model_type, input_dim, target_dim, hidden_size, num_layers):
        """
        Initialize the dynamics model as an RNN with the given parameters.
        
        Parameters:
        ----------
        rnn_type :  str 
            the type of RNN to use (RNN, LSTM, GRU)
        model_type : str
            the type of model to parametrise using the RNN. Can be \\
            'gaussian', in which case the output is the mean and variance of the distribution \\
            'regression' in which case the output is the mean only \\
            'poisson' in which case the output is the rate parameter of a Poisson distribution
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
        assert rnn_type in ['RNN', 'LSTM', 'GRU'], "rnn_type must be either 'RNN', 'LSTM' or 'GRU'"
        assert isinstance(num_layers, int), "num_layers must be an integer"
        assert num_layers > 0, "num_layers must be a positive integer"
        
        if rnn_type == 'RNN':
            self.dynamicsmodel = nn.RNN( input_dim, hidden_size, batch_first=True, num_layers=num_layers)
        
        elif rnn_type == 'LSTM':
            self.dynamicsmodel = nn.LSTM(input_dim, hidden_size, batch_first=True, num_layers=num_layers)
       
        elif rnn_type == 'GRU':
            self.dynamicsmodel = nn.GRU( input_dim, hidden_size, batch_first=True, num_layers=num_layers)

        ### Readout Section is responsible for mapping the latent space to the target space ###
        assert model_type in ['gaussian', 'regression', 'poisson'], "model_type must be either 'gaussian','regression', or 'poisson'"
        self.model_type = model_type
        # set up the readout layer according to the model type
        if model_type == 'gaussian':
            self.readout = nn.Linear(  hidden_size, target_dim*2)   # mean and variance for each target dimension
        
        elif model_type == 'regression':
            self.readout = nn.Linear(  hidden_size, target_dim)     # direct prediction for each target dimension
        
        elif model_type == 'poisson':
            self.readout = RateReadout(hidden_size, target_dim)     # rate parameter for each target dimension

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, input):
        """
        Forward pass through the model.
        """
        out, _ = self.dynamicsmodel(input)
        out = self.readout(out)
        return out


class RateReadout(nn.Module):
    """
    Rate readout layer for Poisson models. The latent state z is read out as:\\
    r = exp(Wz+b) * r0
    """
    def __init__(self, input_features, output_features):
        super(RateReadout, self).__init__()
        self.linear = nn.Linear(input_features, output_features, bias=True)
        # Initialize the rate vector
        self.rate_vector = nn.Parameter(torch.ones(output_features))

    def forward(self, z):
        return torch.exp(self.linear(z)) * self.rate_vector


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
TRAINING FUNCTION
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
    if   model.model_type == 'gaussian':
        criterion = gaussian_neg_log_likelihood
    elif model.model_type == 'regression':
        criterion = nn.MSELoss()
    elif model.model_type == 'poisson':
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

    return model, np.round(np.mean(losses, axis=1), 8).reshape(-1)
     

def train_agms(
        model_1:        ApproxGenModel,
        data_1:         AgmTrainingData,
        model_2:        ApproxGenModel,
        data_2:         AgmTrainingData,
        batch_size =    1,
        epochs =        1000, 
        learning_rate = 0.001,
        lr_decay_step = 100,
        lr_decay_gamma =1.0,
        l2_penalty =    0.0,
        optimize =      'adam',
        plot_loss =     False,
        loss_history =  False,
        ):
    """
    Train the RNN model on the given data. Wrapper function around the _train_agm function, that tries to ensure compatibility between the models and the data.
    """

    # check the models
    assert isinstance(model_1, ApproxGenModel),         "model_1 must be an instance of ApproxGenModel"
    assert isinstance(model_2, ApproxGenModel),         "model_2 must be an instance of ApproxGenModel"
    assert model_1.model_type == model_2.model_type,    "model_1 and model_2 must have the same model type"
    assert model_1.get_device() == model_2.get_device(),"model_1 and model_2 must be on the same device"
    
    # check the data, and compatibility with the models
    assert isinstance(data_1, AgmTrainingData),    "data_1 must be an instance of AgmTrainingData"
    assert isinstance(data_2, AgmTrainingData),    "data_2 must be an instance of AgmTrainingData"
    assert data_1.feature_dim == model_1.input_dim, "data_1 and model_1 must have the same feature dimension"
    assert data_1.target_dim == model_1.target_dim, "data_1 and model_1 must have the same target variable dimension"
    assert data_2.feature_dim == model_2.input_dim, "data_2 and model_2 must have the same feature dimension"
    assert data_2.target_dim == model_2.target_dim, "data_2 and model_2 must have the same target variable dimension"
    
    # check the hyperparameters of the training procedure
    assert isinstance(epochs, int),                 "epochs must be an integer"
    assert epochs > 50,                             "epochs must be greater than 50"
    assert isinstance(learning_rate, float),        "learning_rate must be a float"
    assert learning_rate > 0,                       "learning_rate must be positive"
    assert isinstance(l2_penalty, float),           "l2_penalty must be a float"
    assert l2_penalty >= 0,                         "l2_penalty must be non-negative"
    assert optimize == 'adam' or optimize == 'sgd', "optimize must be either 'adam' or 'sgd'"
    
    # check the plotting and loss history options
    assert isinstance(plot_loss, bool), "plot must be a boolean"
    assert isinstance(loss_history, bool), "loss_history must be a boolean"
   

    model_1, losses_1 = _train_agm(
        model           = model_1,
        data            = data_1,
        batch_size      = batch_size,
        epochs          = epochs,
        learning_rate   = learning_rate,
        lr_decay_step   = lr_decay_step,
        lr_decay_gamma  = lr_decay_gamma,
        optimize        = optimize,
        l2_penalty      = l2_penalty
        )
    
    model_2, losses_2 = _train_agm(
        model           = model_2,
        data            = data_2,
        batch_size      = batch_size,
        epochs          = epochs,
        learning_rate   = learning_rate,
        lr_decay_step   = lr_decay_step,
        lr_decay_gamma  = lr_decay_gamma,
        optimize        = optimize,
        l2_penalty      = l2_penalty
        )
    
    # calculate the transfer entropy estimate
    TE_hist = np.round(losses_1 - losses_2, 4)
    TE = np.round(np.mean(TE_hist[-10:]),4)
    if loss_history:
        loss_return = TE_hist
    elif not loss_history:
        loss_return = TE
    
    # print the final results
    print(f'Loss 1: {np.round(np.mean(losses_1[-10:]), 6)}, Loss 2: {np.round(np.mean(losses_2[-10:]), 6)}, Est. TE: {TE}')

    # plot loss over training
    if plot_loss:
        # plot individual losses
        plt.figure(figsize=(12, 3))
        plt.title('Train loss vs epoch')
        plt.plot(losses_1, alpha = 0.5, label = 'model 1 loss')
        plt.plot(losses_2, alpha = 0.5, label = 'model 2 loss')
        plt.legend()
        plt.show()
        # plot the TE estimate
        plt.figure(figsize=(12, 3))
        plt.title('TE estimate vs epoch')
        plt.plot(np.round(losses_1 - losses_2, 4), alpha = 0.5, label = 'TE estimate')
        plt.legend()
        plt.show()

    return model_1, model_2, loss_return

def get_means_variances_torch(output:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process the raw outputs of the model to separate the means and variances.

    :param output: A tensor of shape (t, 2k) containing interleaved predicted means and log variances.
    :return: Two tensors, one for the means and one for the variances.
    """
    # Separate means and log variances
    means = output[:, 0::2]             # Even indexed outputs: means
    log_variances = output[:, 1::2]     # Odd indexed outputs: log variances
    variances = torch.exp(log_variances)

    return means, variances

"""
USER FRIENDLY INIT
"""

def init_agms_from_loaders(
        data_1:         AgmTrainingData,
        data_2:         AgmTrainingData,
        hidden_size, 
        rnn_type =      "RNN", 
        model_type =    "gaussian", 
        num_layers =    1, 
        ) ->            tuple[ApproxGenModel, ApproxGenModel]:
    """
    Create a pair of probabilistic dynamics models from the given data loaders.
    - If estimating TE, data_1, and therefore model_1 will be used for estimating H(Y+|Y-), while data_2 and model_2 will be used for estimating H(Y+|Y-,X).
    - If estimating CTE, data_1, and therefore model_1 will be used for estimating H(Y+|Y-,Z-), while data_2 and model_2 will be used for estimating H(Y+|Y-,Z-,X-).

    Parameters:
    ----------
    data_1 : AgmTrainingData
        the data loader object that contains the data to train the first model
    data_2 : AgmTrainingData
        the data loader object that contains the data to train the second model
    rnn_type : str 
        the type of RNN to use (RNN, LSTM, GRU)
    model_type : str
        the type of model to parametrise using the RNN. Can be \\
        'gaussian', in which case the output is the mean and variance of the distribution \\
        'regression' in which case the output is the mean only \\
        'poisson' in which case the output is the rate parameter of a Poisson distribution
    num_layers : int
        the number of layers in the RNN

    Returns:
    -------
    tuple[ApproxGenModel, ApproxGenModel]
        the two models that are to be trained on the given data
    """

    assert isinstance(data_1, AgmTrainingData), "data_1 must be an instance of AgmTrainingData"
    assert isinstance(data_2, AgmTrainingData), "data_2 must be an instance of AgmTrainingData"

    assert data_1.var_to_dim == data_2.var_to_dim, "data_1 and data_2 must have the same target variable dimension"
    assert data_1.var_cond_dim == data_2.var_cond_dim, "data_1 and data_2 must have the same conditional variable dimension"
    assert data_1.var_from_dim == 0, "data_1 must have no var_from data"
    assert data_2.var_from_dim > 0, "data_2 must have non-zero var_from dimension"

    model_1 = ApproxGenModel(
        rnn_type    = rnn_type,
        model_type  = model_type,
        input_dim   = data_1.feature_dim,
        target_dim  = data_1.target_dim,
        hidden_size = hidden_size,
        num_layers  = num_layers,
    )

    model_2 = ApproxGenModel(
        rnn_type    = rnn_type,
        model_type  = model_type,
        input_dim   = data_2.feature_dim,
        target_dim  = data_2.target_dim,
        hidden_size = hidden_size,
        num_layers  = num_layers,
    )

    return model_1, model_2



"""
VISUALIZATION FUNCTIONS
"""

def plot_agm_output(
        model:          ApproxGenModel, 
        input_data:     np.ndarray,
        target:         np.ndarray|None = None,
        plot_start:     int = 0, 
        plot_end        = None, 
        by_dim:         bool = True
        ):
    
    """
    Plot the predictions of the approximate generative model (against the target if provided) for the given input data.

    Parameters:
    ----------
    model : ApproxGenModel
        a trained instance of the ApproxGenModel class
    input_data : np.ndarray
        the input from which to generate the predictions
    target : np.ndarray | None
        the target data, if available
    plot_start : int
        the starting timestep for the plot
    plot_end : int | None
        the ending timestep for the plot. If None, the entire input data will be plotted
    by_dim : bool
        whether to plot the target dimensions separately or aggregated

    Returns:
    -------
    None
    """

    assert isinstance(model, ApproxGenModel), "model must be an instance of ApproxGenModel"
    
    # check the input data
    assert isinstance(input_data, np.ndarray), "dataloader must be a numpy array"
    assert len(input_data.shape) == 2, "the input dataset must be a 2D array"
    assert input_data.shape[1] == model.input_dim, "model and dataloader must have the same feature dimension"
    
    # check target data if it is provided
    assert target is None or isinstance(target, np.ndarray), "target must be a numpy array or None"
    if target is not None:
        assert len(target.shape) == 2, "the target dataset must be a 2D array"
        assert target.shape[1] == model.target_dim, "model and dataloader must have the same target variable dimension"
        plot_title = 'Model vs target'
    else:
        plot_title = 'Model prediction'

    # check the plotting parameters
    assert plot_start >= 0, "plot_start must be a non-negative integer"
    if plot_end is None:
        plot_end = input_data.shape[0]
    else:
        assert plot_end <= input_data.shape[0], "plot_end must be less than the length of the input"
    assert isinstance(by_dim, bool), "by_dim must be a boolean. If True, each dimension of the target is plotted seperately, otherwise it will be aggregated"


    # run and extract the predictions
    input = np.zeros((1, input_data.shape[0], input_data.shape[1]))
    input[0, :, :] = input_data
    input = torch.tensor(input, dtype=torch.float32).to(model.get_device())
    predicted = model(input)
    predicted = predicted.detach().cpu()[0] # since output is a 1 x T x output_dim tensor

    # check the model type and plot accordingly
    if   model.model_type == 'gaussian':
        """
        Gaussian models have two outputs per target dimension: mean and variance.
        """
        means, variances = get_means_variances_torch(predicted)
        means = means.numpy()
        variances = variances.numpy()
        stds = np.sqrt(variances)
        
        if by_dim == False:
            
            plt.figure(figsize=(18, 3))
            plt.title(plot_title)
            plt.plot(means[plot_start:plot_end])
            if target is not None:
                plt.plot(target[plot_start:plot_end], linestyle='--')
            plt.show()
        
        elif by_dim == True:
            for i in range(model.target_dim):
                ci_low = means[plot_start:plot_end, i]-2*stds[plot_start:plot_end, i]
                ci_high = means[plot_start:plot_end, i]+2*stds[plot_start:plot_end, i]
                plt.figure(figsize=(18, 3))
                plt.title(f'{plot_title} for dim. {i}')
                plt.plot(means[plot_start:plot_end, i])
                plt.fill_between(np.arange(plot_start, plot_end), ci_low, ci_high, alpha=0.5, label="95% CI")
                if target is not None:
                    plt.plot(target[plot_start:plot_end, i], linestyle='--')
                plt.show()
    
    elif model.model_type == 'regression':
        """
        Regression models simply output a point estimate.
        """
        pest = predicted.numpy()
        if by_dim == False:
            plt.figure(figsize=(18, 3))
            plt.title(plot_title)
            plt.plot(pest[plot_start:plot_end])
            if target is not None:
                plt.plot(target[plot_start:plot_end], linestyle='--')
            plt.show()
        
        elif by_dim == True:
            for i in range(model.target_dim):
                plt.figure(figsize=(18, 3))
                plt.title(f'{plot_title} for dim. {i}')
                plt.plot(pest[plot_start:plot_end, i])
                if target is not None:
                    plt.plot(target[plot_start:plot_end, i], linestyle='--')
                plt.show()

    elif model.model_type == 'poisson':
        """
        Poisson models output the rate parameter of a Poisson distribution.
        """
        rates = predicted.numpy()
        if by_dim == False:
            plt.figure(figsize=(18, 3))
            plt.title(plot_title)
            plt.plot(rates[plot_start:plot_end])
            if target is not None:
                plt.plot(target[plot_start:plot_end], linestyle='--')
            plt.show()
        
        elif by_dim == True:
            for i in range(model.target_dim):
                plt.figure(figsize=(18, 3))
                plt.title(f'{plot_title} for dim. {i}')
                plt.plot(rates[plot_start:plot_end, i])
                if target is not None:
                    plt.plot(target[plot_start:plot_end, i], linestyle='--')
                plt.show()


'''
COUNTERFACTUAL PREDICTION
'''

def agm_predict_counterfactual(
        model:          ApproxGenModel,
        var_from:       np.ndarray,
        var_to_start:   np.ndarray,
        var_cond:       np.ndarray = None,
        random_rep:     int = 0,
        random_noise:   float = 0.00
        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    
    """
    Use a learnt dynamics model to perform simulations of counterfactual trajectories.
    """

    assert isinstance(model, ApproxGenModel), "model must be an instance of ApproxGenModel"
    assert random_rep == 0 or random_rep > 10, "random_rep must be 0 (deterministic operation) or greater than 10 for sufficient samples for stochastic operation"
    assert model.model_type == 'gaussian', "simulations only implemented for gaussian models"
    
    var_from_dim        = var_from.shape[1]
    var_to_dim          = var_to_start.shape[1]
    assert var_to_dim == model.target_dim, "var_to_start dimension must match the target dimension of the model"
    var_cond_dim        = var_cond.shape[1] if var_cond is not None else 0
    total_input_dim     = var_from_dim + var_to_dim + var_cond_dim
    assert model.input_dim == total_input_dim, "model input dimension must match the total input dimension"
    var_to_timesteps    = var_to_start.shape[0]
    total_timesteps     = var_from.shape[0]

    # contruct the input tensor, given the user supplied data
    def construct_input(var_to_start, var_from, var_cond):
        input = np.zeros((total_timesteps, total_input_dim))
        input[:var_to_timesteps, :var_to_dim] = var_to_start
        input[:, var_to_dim:var_to_dim+var_from_dim] = var_from
        if var_cond is not None:
            input[:, var_to_dim+var_from_dim:] = var_cond
        input = torch.tensor(input, dtype=torch.float32).to(model.get_device()).contiguous()

        return input
    
    with torch.no_grad():

        # deterministic operation, always take the mean of the distribution as the next value. Always gives the same result
        if random_rep == 0:
            # set up the input
            input = construct_input(var_to_start, var_from, var_cond)
            # reset the hidden state
            model.dynamicsmodel.hidden_state = None
            # iterate over the timesteps
            for i in range(var_to_timesteps, total_timesteps):
                predicted = model(input[:i])
                means, variances = get_means_variances_torch(predicted)
                input[i, :var_to_dim] = means[-1]

            return input[:,:var_to_dim].detach().cpu().numpy()
        
        # stochastic operation, sample from the distribution at each timestep. Gives different results each time
        else:
            res_holder = np.zeros((random_rep, total_timesteps, var_to_dim))
            for r in range(random_rep):
                # set up the input
                input = construct_input(var_to_start, var_from, var_cond)
                # reset the hidden state
                model.dynamicsmodel.hidden_state = None
                # iterate over the timesteps
                for i in range(var_to_timesteps, total_timesteps):
                    predicted = model(input[:i])
                    means, variances = get_means_variances_torch(predicted)
                    sample = torch.normal(means[-1, :], torch.sqrt(variances[-1, :])) + torch.randn_like(means[-1, :])*random_noise
                    input[i, :var_to_dim] = sample

                res_holder[r, :, :] = input[:,:var_to_dim].detach().cpu().numpy()

            pred_mean = np.mean(res_holder, axis=0)
            pred_std = np.std(res_holder, axis=0)
            return pred_mean, pred_std
