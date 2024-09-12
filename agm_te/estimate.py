import copy 
import numpy as np
import torch
import matplotlib.pyplot as plt

from agm_te.dataset import AgmTrainingData, DataSet
from agm_te.model import ApproxGenModel, _train_agm

"""
USER FRIENDLY INIT
"""

def init_agms_from_loaders(
        data_1:         AgmTrainingData,
        data_2:         AgmTrainingData,
        hidden_size, 
        dyn_model_type =      "RNN", 
        obs_model_type =    "gaussian", 
        num_layers =    1,
        timestep =      None,
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
    dyn_model_type : str 
        the type of RNN to use (MLP, RNN, LSTM, GRU)
    obs_model_type : str
        the type of model to parametrise using the RNN. Can be \\
        'gaussian', in which case the output is the mean and variance of the distribution \\
        'regression' in which case the output is the mean only \\
        'poisson' in which case the output is the rate parameter of a Poisson distribution
    num_layers : int
        the number of layers in the RNN
    timestep : float | None
        the timestep of the Poisson model. Only required if obs_model_type is 'poisson'

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
        dyn_model_type    = dyn_model_type,
        obs_model_type  = obs_model_type,
        input_dim   = data_1.feature_dim,
        target_dim  = data_1.target_dim,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        timestep    = timestep,
    )

    model_2 = ApproxGenModel(
        dyn_model_type    = dyn_model_type,
        obs_model_type  = obs_model_type,
        input_dim   = data_2.feature_dim,
        target_dim  = data_2.target_dim,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        timestep    = timestep,
    )

    return model_1, model_2

"""
USER FRIENDLY TRAINING OF PAIR
"""

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
        ):
    """
    Train the AGM on the given data. Wrapper function around the _train_agm function, that tries to ensure compatibility between the models and the data.
    """

    # check the models
    assert isinstance(model_1, ApproxGenModel),         "model_1 must be an instance of ApproxGenModel"
    assert isinstance(model_2, ApproxGenModel),         "model_2 must be an instance of ApproxGenModel"
    assert model_1.obs_model_type == model_2.obs_model_type,    "model_1 and model_2 must have the same model type"
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

    return model_1, losses_1, model_2, losses_2




"""
TRANSFER ENTROPY ESTIMATION PROCESS
"""
def _TE_from_losses(losses_1, losses_2) -> float:
    """
    Calculate the transfer entropy estimate from the loss history of the two models.
    Final TE estimate is the difference in the mean negative log likelihood of the two 
    competing models in the last 10% of the epochs.
    """
    last10p_epochs = int(len(losses_1)/10)
    TE_hist = losses_1 - losses_2
    TE = np.round(np.mean(TE_hist[-last10p_epochs:]),4)
    return TE

def agm_estimate_TE(
        dataset:        DataSet,
        model_params:   dict,
        train_params:   dict,
        var_from:       str,
        var_to:         str,
        var_cond: str = None,
        ) -> tuple[ApproxGenModel, ApproxGenModel, float]:
    """
    Estimate the (conditional) transfer entropy between two variables using AGM-TE.
    (C)TE is estimated as the difference in the mean negative log likelihood of the two competing probabilistic models.

    Parameters:
    ----------
    dataset : DataSet
        the dataset object that contains the timeseries data to train the models
    model_params : dict
        shared parameters of the AGM models (e.g. type, hidden size, number of layers, probability model type, etc.) see `ApproxGenModel.__init__` for details
    train_params : dict
        shared parameters of the training process (e.g. batch size, number of epochs, learning rate, etc.) see `_train_agm` for details
    var_from : str
        the name of the (potentially) causal variable
    var_to : str
        the name of the target variable
    var_cond : str | None
        if estimating conditional transfer entropy, the name of the conditioning variable
   

    Returns:
    -------
    tuple[ApproxGenModel, ApproxGenModel, float]
        the two models that were trained on the given data, and the transfer entropy estimate
    """
    # Get dataloaders that contain the data from which to estimate the transfer entropy
    if var_cond is None:
        dataloader_1, dataloader_2 = dataset.get_TE_dataloaders(var_from, var_to)
    else:
        dataloader_1, dataloader_2 = dataset.get_CTE_dataloaders(var_from, var_to, var_cond)

    # Create the models
    device = model_params['compute_device']
    model_params_copy = copy.deepcopy(model_params) # needed to prevent modification of the original dictionary
    del model_params_copy['compute_device']
    model_1, model_2 = init_agms_from_loaders(dataloader_1, dataloader_2, **model_params_copy)
    
    model_1.to(device)
    model_2.to(device)

    # Train the models
    model_1, losses_1, model_2, losses_2 = train_agms(model_1, dataloader_1, model_2, dataloader_2, **train_params)
    
    # Calculate the transfer entropy estimate
    TE = _TE_from_losses(losses_1, losses_2)

    return model_1, model_2, TE
    


"""
VISUALIZATION FUNCTIONS
"""

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
    if   model.obs_model_type == 'gaussian':
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
    
    elif model.obs_model_type == 'regression':
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

    elif model.obs_model_type == 'poisson':
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
        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    
    """
    Use a trained AGM to perform simulations of counterfactual trajectories.
    Should only be used, if TE X->Y is positive, and TE Y->Y is 0. 
    
    Parameters:
    ----------
    model : ApproxGenModel
        the trained model to use for the counterfactual prediction. Should be model_2 from the training process!!
    var_from : np.ndarray
        manipulated history for the causal variable, shape (pred_timesteps, causalvar_dim). Counterfactual will be predicted for all timesteps of the causal variable.
    var_to_start : np.ndarray
        starting values for the target variable, shape (start_timesteps, target_dim)
    var_cond : np.ndarray | None
        history of the conditioning variable, shape (pred_timesteps, condvar_dim)
    random_rep : int
        number of random samples to draw from the predicted distribution. If 0, deterministic predictions are made. If >10, stochastic predictions are made.

    Returns:
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        the predicted counterfactual trajectories. If random_rep is 0, a single array is returned. If random_rep is >10, a tuple of arrays is returned, containing the mean and standard deviation of the samples.
    """

    # Check basic types and inputs
    assert isinstance(model, ApproxGenModel), "model must be an instance of ApproxGenModel"
    
    assert model.obs_model_type == 'gaussian', "simulations only implemented for gaussian models"
    
    assert isinstance(var_from, np.ndarray), "var_from must be a numpy array"
    assert len(var_from.shape) == 2, "var_from must be a 2D array"
    assert isinstance(var_to_start, np.ndarray), "var_to_start must be a numpy array"
    assert len(var_to_start.shape) == 2, "var_to_start must be a 2D array"
    if var_cond is not None:
        assert isinstance(var_cond, np.ndarray), "var_cond must be a numpy array"
        assert len(var_cond.shape) == 2, "var_cond must be a 2D array"

    assert random_rep == 0 or random_rep > 10, "random_rep must be 0 (deterministic operation) or greater than 10 for sufficient samples for stochastic operation"

    # Check the dimensions of the inputs
    var_from_dim        = var_from.shape[1]
    var_to_dim          = var_to_start.shape[1]
    assert var_to_dim == model.target_dim, "var_to_start dimension must match the target dimension of the model"
    var_cond_dim        = var_cond.shape[1] if var_cond is not None else 0
    total_input_dim     = var_from_dim + var_to_dim + var_cond_dim
    assert model.input_dim == total_input_dim, "model input dimension must match the total input dimension"
    assert var_to_start.shape[0] <= var_from.shape[0], "the number of timesteps provided for the target variable must be <= the number of timesteps provided for the causal variable"
    if var_cond is not None:
        assert var_cond.shape[0] == var_from.shape[0], "the number of timesteps provided for the conditioning variable must match the number of timesteps provided for the causal variable"
    var_to_timesteps    = var_to_start.shape[0] # number of timesteps for which the causal variable is provided
    t_to_sim    = var_from.shape[0] + 1  # prediction happens for all timesteps of the provided causal variable, plus one extra timestep

    # contruct the input tensor, given the user supplied data
    def construct_input(var_to_start, var_from, var_cond):
        """
        For causal variable X [with d dimensions], target variable Y [with k dimensions] (and conditioning variable Z [with l dimensions]), 
        construct the input tensor for the model. Tensor is size (t_to_sim, total_input_dim). 
        
        Contents and structure are as follows:
        
            from t = 0:var_to_timesteps:
                [ Y_0, Y_1, ..., Y_d,   X_0, X_1, ..., X_k,   (Z_0, Z_1, ..., Z_l) ]

            from t = var_to_timesteps:t_to_sim-1:
                [ 0,   0,   ..., 0,     X_0, X_1, ..., X_k,   (Z_0, Z_1, ..., Z_l) ]

            final row:
                [ 0,   0,   ..., 0,     0,   0,   ..., 0,     (0,   0,   ..., 0  ) ]
        """
        input = np.zeros((t_to_sim, total_input_dim))
        # Write in the starting values for the target variable
        input[0:var_to_timesteps,0:var_to_dim] = var_to_start
        # Write in the values for the causal variable
        input[0:-1,var_to_dim:var_to_dim+var_from_dim] = var_from
        # Write in the values for the conditioning variable
        if var_cond is not None:
            input[0:-1,var_to_dim+var_from_dim:] = var_cond
        
        # Convert to torch tensor and move to the device
        input = torch.tensor(input, dtype=torch.float32).to(model.get_device()).contiguous()

        return input
    
    def predict(model, input, probabilistic):
        """
        Using an input prepared by `construct_input`, predict the target variable for all timesteps.

        The input tensor is modified to have the following structure:

            from t = 0:var_to_timesteps:
                [ Y_0, Y_1, ..., Y_d,   X_0, X_1, ..., X_k,   (Z_0, Z_1, ..., Z_l) ]

            from t = var_to_timesteps:t_to_sim-1:
                [ ^Y0, ^Y1, ..., ^Yd,   X_0, X_1, ..., X_k,   (Z_0, Z_1, ..., Z_l) ]

            final row:
                [ ^Y0, ^Y1, ..., ^Yd,   0,   0,   ..., 0,     (0,   0,   ..., 0  ) ]

        """

        with torch.no_grad():
            # reset the hidden state (for RNN models)
            model.dynamicsmodel.hidden_state = None

            # Iterate over all timesteps for which the causal variable is provided.
            for i in range(var_to_timesteps, t_to_sim):
                # Use the input up to the current timestep (only makes a difference for RNN models, MLP could just use the current step)
                predicted = model(input[:i]) 

                # Get the means and variances of the predicted distribution
                latest_prediction = predicted[[-1],:]

                means, variances = get_means_variances_torch(latest_prediction)
                
                if probabilistic:
                    # Sample from the distribution
                    predicted_tplus1 = torch.normal(means, torch.sqrt(variances))
                else:
                    # Take the mean of the distribution
                    predicted_tplus1 = means

                input[i,0:var_to_dim] = predicted_tplus1
            
        pred_target_seq = input[:,0:var_to_dim].detach().cpu().numpy()

        return pred_target_seq
    

    # Deterministic 
    if random_rep == 0:
        # set up the input
        input = construct_input(var_to_start, var_from, var_cond)
        
        # predict the target variable
        pred_target_seq = predict(model, input, probabilistic=False)

        return pred_target_seq
    
    # Probabilistic
    else:
        res_holder = np.zeros((random_rep, t_to_sim, var_to_dim))
        
        for r in range(random_rep):
            # set up the input
            input = construct_input(var_to_start, var_from, var_cond)
            
            # predict the target variable
            pred_target_seq = predict(model, input, probabilistic=True)

            res_holder[r, :, :] = pred_target_seq

        pred_mean = np.mean(res_holder, axis=0)
        pred_std = np.std(res_holder, axis=0)
        
        return pred_mean, pred_std