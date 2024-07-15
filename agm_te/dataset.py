import numpy as np
import torch

class DataSet():
    """
    Class which holds sampled data for a system of multiple variables.
    The raw data is stored as a dictionary, with variable names as keys and lists of numpy arrays as values.
    The class provides methods which yield the subsets of data for estimating transfer entropy (TE) and conditional transfer entropy (CTE) between variables in the DataSet.
    Some methods yield DirectDataLoader objects, which contain the formatted and batched data ready for training the pytorch models.
    """
    def __init__(self, data: dict):
        # check basic types
        assert isinstance(data, dict), "data must be a dictionary, with variable names as keys and numpy arrays as values"
        
        assert len(data) > 1, "data must have at least two variables"
        
        for key, value in data.items():
            assert isinstance(key, str), "variable names (keys) in data must be strings"
            assert isinstance(value, list), "values in data must be lists of numpy arrays"
        
        assert 'remaining' not in data.keys(), "variable name 'remaining' is reserved for conditioning on all remaining variables"

        for key, value in data.items():
            for array in value:
                assert isinstance(array, np.ndarray), "values in data must be lists of numpy arrays"
        # check shapes of numpy arrays
        for key, value in data.items():
            for array in value:
                assert array.ndim == 2, "all numpy arrays in data must be 2D, with timesteps as axis 0 (rows), and features on axis 1 (columns)"
        
        for key, value in data.items():
            var_dim = []
            for array in value:
                var_dim.append(array.shape[1])
            assert len(set(var_dim)) == 1, f"numpy arrays for variable {key} have different dimensions (# of columns)"

        n_traj = [len(value) for value in data.values()]
        assert len(set(n_traj)) == 1, "all variables in data must have the same number of unique trajectories"

        total_timesteps = [0 for _ in range(len(data))]
        for key, value in data.items():
            for array in value:
                total_timesteps[list(data.keys()).index(key)] += array.shape[0]
        assert len(set(total_timesteps)) == 1, "all variables in data must have the same number of timesteps"

        timestep_per_traj = []
        for key, value in data.items():
            timestep_per_traj.append([array.shape[0] for array in value])
        assert len(set([tuple(x) for x in timestep_per_traj])) == 1, "all variables in data must have the same number of timesteps per trajectory"

        self.data = data
        self.n_traj = n_traj[0] # number of trajectories is the number of list elements
        self.n_timesteps = total_timesteps[0] # number of timesteps is the sum of the number of timesteps in each trajectory
        self.n_vars = len(data) # number of variables is the number of keys in the dictionary

    def __getitem__(self, var) -> list[np.ndarray]:
        return self.data[var]
    
    def __str__(self) -> str:
        str = f'DataSet of {self.n_timesteps} timesteps across {self.n_traj} trajectories for {len(list(self.data.keys()))} variables:\n'
        for key, value in self.data.items():
            str += f'\t {key} is {value[0].shape[1]} dimensional\n'
        
        return str
    
    def get_TE_data(self, var_from, var_to):
        """
        Returns datasets used for estimating the transfer entropy (TE) from the variable 'var_from' to the variable 'var_to'.

        Say var_from is X, and var_to is Y, then these datasets allow us to estimate the TE:

        T(X -> Y) = H(Y+|Y-) - H(Y+|X-,Y-)

        - feat_var_to is Y- (the past of Y), used as input when estimating H(Y+|Y-)
        - feat_var_both is X- and Y- (the past of both X and Y), used as input when estimating H(Y+|X-,Y-)
        - target_var_to is Y+ (the future of Y), used as the target in both H(Y+|Y-) and H(Y+|X-,Y-)

        Features are offset by -1 timestep, so that the target is the next timestep in the sequence.

        Parameters:
        ----------
        var_from :          str
            Variable name for the source variable.
        var_to :            str
            Variable name for the target variable.

        Returns:
        -------
        feat_var_to :   list[np.ndarray]
            Values of var_to[:-1], forming the sample of Y-
        feat_var_from : list[np.ndarray]
            Values of var_from[:-1] forming the sample of X-
        target_var_to :     list[np.ndarray]
            Values of var_to[1:], forming the sample of Y+
        """

        assert var_from in self.data.keys(), f"variable {var_from} not found in the dataset"
        assert var_to in self.data.keys(), f"variable {var_to} not found in the dataset"
        assert var_from != var_to, "var_from and var_to must be different"

        feat_var_to   = []
        feat_var_from = []
        target_var_to     = []


        for traj_idx in range(self.n_traj):
            f_var_to      = self.data[var_to][traj_idx][:-1]
            f_var_from    = self.data[var_from][traj_idx][:-1]
            t_var_to      = self.data[var_to][traj_idx][1:]

            feat_var_to.append(f_var_to)
            feat_var_from.append(f_var_from)
            target_var_to.append(t_var_to)

        return feat_var_to, feat_var_from, target_var_to

    def get_CTE_data(self, var_from, var_to, var_cond='remaining'):
        """
        Returns datasets for estimating the conditional transfer entropy (CTE) of the variable 'var_from' to the variable 'var_to', conditioned on the variable (or set of variables) 'var_cond'.

        Say var_from is X, var_to is Y, and var_cond is Z, then these dataloaders allow us to estimate the CTE:

        T(X -> Y|Z) = H(Y+|Y-,Z-) - H(Y+|X-,Y-,Z-)
        
        - feat_var_to is Y-, Z- (the past of Y and Z), used as input when estimating H(Y+|Y-,Z-)
        - feat_var_both is X-, Y-, and Z- (the past of  X, Y, and Z), used as input when estimating H(Y+|X-,Y-,Z-)
        - target_var_to is Y+ (the future of Y), used as the target in both H(Y+|Y-,Z-) and H(Y+|X-,Y-,Z-)

        Features are offset by -1 timestep, so that the target is the next timestep in the sequence.

        Parameters:
        ----------
        var_from :          str
            Variable name for the source variable.
        var_to :            str
            Variable name for the target variable.
        var_cond :          str, optional
            Variable name for the conditioning variable. Default is 'remaining', which means conditioning on all remaining variables.
        
        Returns:
        -------
        feat_var_to :   list[np.ndarray]
            Values of var_to[:-1], forming the sample of Y-.
        feat_var_from : list[np.ndarray]
            Values of var_from[:-1], forming the sample of X-.
        feat_var_cond : list[np.ndarray]
            Values of var_cond[:-1], forming the sample of Z-.
        target_var_to :     list[np.ndarray]
            Values of var_to[1:], forming the sample of Y+.
        """

        assert self.n_vars > 2, "dataset must have more than 2 variables to condition on a third variable"
        assert var_from in self.data.keys(), f"variable {var_from} not found in the dataset"
        assert var_to in self.data.keys(), f"variable {var_to} not found in the dataset"
        assert var_from != var_to, "var_from and var_to must be different"
        assert var_cond != var_from, "var_cond cannot be the same as var_from"
        assert var_cond != var_to, "var_cond cannot be the same as var_to"

        if var_cond != 'remaining':
            raise NotImplementedError("Conditioning to specific variables is not yet implemented.")
        
        feat_var_to   = []
        feat_var_from = []
        feat_var_cond = []
        target_var_to     = []

        for traj_idx in range(self.n_traj):
            f_var_to   = self.data[var_to][traj_idx][:-1]
            f_var_from = self.data[var_from][traj_idx][:-1]
            f_var_cond = np.concatenate([self.data[var][traj_idx][:-1] for var in self.data.keys() if var not in [var_from, var_to]])
            t_var_to = self.data[var_to][traj_idx][1:]

            feat_var_to.append(f_var_to)
            feat_var_from.append(f_var_from)
            feat_var_cond.append(f_var_cond)
            target_var_to.append(t_var_to)

        return feat_var_to, feat_var_from, feat_var_cond, target_var_to

    def get_TE_dataloaders(self, device, var_from, var_to, batch_size=1):
        """
        Returns dataloaders for estimating the transfer entropy (TE) from the variable 'var_from' to the variable 'var_to'.

        Say var_from is X, and var_to is Y, then:

        T(X -> Y) = H(Y+|Y-) - H(Y+|X-,Y-)

        - dataloader_1 is for estimating the conditional entropy H(Y+|Y-)
        - dataloader_2 is for estimating the conditional entropy H(Y+|X-,Y-)

        Parameters:
        ----------
        device :        torch.device
            Device on which the data is stored. This should be the device on which the model is run.
        var_from :      str
            Variable name for the source variable.
        var_to :        str
            Variable name for the target variable.
        batch_size :    int
            Batch size for the dataloaders.
        
        Returns:
        -------
        dataloader_1 :  DirectDataLoader
            Dataloader containing Y- as features and Y+ as target.
        dataloader_2 :  DirectDataLoader
            Dataloader containing X- and Y- as features, and Y+ as target.
        """
        feat_var_to, feat_var_from, target_var_to = self.get_TE_data(var_from, var_to)

        # dataloader for H(Y+|Y-), batch_feat_var_cond is a list of None and batch_feat_var_from is a list of None
        batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to = \
            prepare_training_data(batch_size, feat_var_to, target_var_to)
        dataloader_1 = DirectDataLoader(batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to, device)
        
        # dataloader for H(Y+|X-,Y-), batch_feat_var_cond is a list of None
        batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to = \
            prepare_training_data(batch_size, feat_var_to, target_var_to, f_var_from=feat_var_from)
        dataloader_2 = DirectDataLoader(batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to, device)

        return dataloader_1, dataloader_2
    
    def get_CTE_dataloaders(self, device, var_from, var_to, var_cond, batch_size=1):
        """
        Returns dataloaders for estimating the conditional transfer entropy (CTE) of the variable 'var_from' to the variable 'var_to', conditioned on the variable (or set of variables) 'var_cond'.

        Say var_from is X, var_to is Y, and var_cond is Z, then these dataloaders allow us to estimate the CTE:

        T(X -> Y|Z) = H(Y+|Y-,Z-) - H(Y+|X-,Y-,Z-)
        
        - dataloader_1 is for estimating the conditional entropy H(Y+|Y-,Z-)
        - dataloader_2 is for estimating the conditional entropy H(Y+|X-,Y-,Z-)

        Parameters:
        ----------
        device :        torch.device
            Device on which the data is stored. This should be the device on which the model is run.
        var_from :      str
            Variable name for the source variable.
        var_to :        str
            Variable name for the target variable.
        var_cond :      str, optional
            Variable name for the conditioning variable. Default is None, which means conditioning on all remaining variables.
        batch_size :    int
            Batch size for the dataloaders.
        
        Returns:
        -------
        dataloader_1 :  DirectDataLoader
            Dataloader containing Y- and Z- as features, and Y+ as target.
        dataloader_2 :  DirectDataLoader
            Dataloader containing X-, Y-, and Z- as features, and Y+ as target.
        """
        feat_var_to, feat_var_from, feat_var_cond, target_var_to = self.get_CTE_data(var_from, var_to, var_cond)
        
        
        # dataloader for H(Y+|Y-,Z-)
        batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to = \
            prepare_training_data(batch_size, feat_var_to, target_var_to, f_var_cond=feat_var_cond)
        dataloader_1 = DirectDataLoader(batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to, device)
        
        # dataloader for H(Y+|X-,Y-,Z-)
        batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to = \
            prepare_training_data(batch_size, feat_var_to, target_var_to, f_var_from=feat_var_from, f_var_cond=feat_var_cond)
        dataloader_2 = DirectDataLoader(batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to, device)

        return dataloader_1, dataloader_2

def batch_data_list(data_list: list[np.ndarray], batch_size: int) -> list[np.ndarray]:
    """
    Take a list of t*d numpy arrays (each a unique trajectory), and return a list of batch_size*t*d numpy arrays.
    """
    assert isinstance(data_list, list), "data_list must be a list"
    assert all(isinstance(x, np.ndarray) for x in data_list), "all elements in data_list must be numpy arrays"
    assert all(x.ndim == 2 for x in data_list), "all numpy arrays in data_list must be 2D (timesteps x features)"
    assert all(x.shape == data_list[0].shape for x in data_list), "all numpy arrays in data_list must have the same number of rows (timesteps) and columns (features)"
    assert isinstance(batch_size, int), "batch_size must be an integer"
    assert batch_size >= 1, "batch_size must be a positive integer"
    assert len(data_list)%batch_size == 0, "number of unique trajectories must be divisible by batch_size"

    batched_data = []
    batch_counter = 0 # keeps track of the index in the current batch
    for data in data_list:
        if batch_counter == 0:
            # start a new batch as all 0s
            batched_data.append(np.zeros((batch_size, data.shape[0], data.shape[1])))
        
        if batch_counter < batch_size:
            # append to the batch (starting from index 0)
            batched_data[-1][batch_counter] = data
            batch_counter += 1
        
        if batch_counter == batch_size:
            batch_counter = 0

    return batched_data

def prepare_training_data(batch_size, f_var_to, t_var_to, f_var_from=None, f_var_cond=None):
    """
    We wish to estimate the transfer entropy (TE) from the source variable X to the target variable Y, possibly conditioned on the variable Z.
    
    The TE is defined as:
        TE(X -> Y) = H(Y+|Y-) - H(Y+|X-,Y-)
    The CTE is defined as:
        CTE(X -> Y|Z) = H(Y+|Y-,Z-) - H(Y+|X-,Y-,Z-)

    These four possible conditional entropies each require a different set of input data, which is prepared here.

    - H(Y+|Y-) requires f_var_to = Y- and t_var_to = Y+
    - H(Y+|X-,Y-) requires f_var_to = Y-, f_var_from = X-, and t_var_to = Y+
    - H(Y+|Y-,Z-) requires f_var_to = Y-, f_var_cond = Z-, and t_var_to = Y+
    - H(Y+|X-,Y-,Z-) requires f_var_to = Y-, f_var_from = X-, f_var_cond = Z-, and t_var_to = Y+

    Parameters:
    ----------
    f_var_to : list[np.ndarray]
        List of numpy arrays containing the sample representing Y- (the past of the target variable).
    t_var_to : list[np.ndarray]
        List of numpy arrays containing the sample representing Y+ (the future of the target variable).
    f_var_from : list[np.ndarray], optional
        List of numpy arrays containing the sample representing X- (the past of the source variable). Default is None.
    f_var_cond : list[np.ndarray], optional
        List of numpy arrays containing the sample representing Z- (the past of the conditioning variable). Default is None.
    """

    assert isinstance(f_var_to, list), "f_var_to must be a list"
    assert all(isinstance(x, np.ndarray) for x in f_var_to), "all elements in f_var_to must be numpy arrays"
    assert all(x.ndim == 2 for x in f_var_to), "all numpy arrays in f_var_to must be 2D (timesteps x features)"
    assert all(x.shape == f_var_to[0].shape for x in f_var_to), "all numpy arrays in f_var_to must have the same number of rows (timesteps) and columns (features)"
    timesteps = f_var_to[0].shape[0]
    var_to_dim = f_var_to[0].shape[1]

    assert isinstance(t_var_to, list), "t_var_to must be a list"
    assert all(isinstance(x, np.ndarray) for x in t_var_to), "all elements in t_var_to must be numpy arrays"
    assert all(x.ndim == 2 for x in t_var_to), "all numpy arrays in t_var_to must be 2D (timesteps x features)"
    assert all(x.shape == t_var_to[0].shape for x in t_var_to), "all numpy arrays in t_var_to must have the same number of rows (timesteps) and columns (features)"
    assert all(x.shape[0] == timesteps for x in t_var_to), "all numpy arrays in t_var_to must have the same number of rows (timesteps) as as the other variables"
    assert all(x.shape[1] == var_to_dim for x in t_var_to), "all numpy arrays in t_var_to must have the same number of columns as f_var_to"

    if f_var_from is not None:
        assert isinstance(f_var_from, list), "f_var_from must be a list"
        assert all(isinstance(x, np.ndarray) for x in f_var_from), "all elements in f_var_from must be numpy arrays"
        assert all(x.ndim == 2 for x in f_var_from), "all numpy arrays in f_var_from must be 2D (timesteps x features)"
        assert all(x.shape == f_var_from[0].shape for x in f_var_from), "all numpy arrays in f_var_from must have the same number of rows (timesteps) and columns (features)"
        assert all(x.shape[0] == timesteps for x in f_var_from), "all numpy arrays in f_var_from must have the same number of rows (timesteps) as the other variables"
        
    if f_var_cond is not None:
        assert isinstance(f_var_cond, list), "f_var_cond must be a list"
        assert all(isinstance(x, np.ndarray) for x in f_var_cond), "all elements in f_var_cond must be numpy arrays"
        assert all(x.ndim == 2 for x in f_var_cond), "all numpy arrays in f_var_cond must be 2D (timesteps x features)"
        assert all(x.shape == f_var_cond[0].shape for x in f_var_cond), "all numpy arrays in f_var_cond must have the same number of rows (timesteps) and columns (features)"
    
    batched_var_to   = batch_data_list(f_var_to, batch_size)
    batched_var_from = batch_data_list(f_var_from, batch_size) if f_var_from is not None else [None]*len(batched_var_to)
    batched_var_cond = batch_data_list(f_var_cond, batch_size) if f_var_cond is not None else [None]*len(batched_var_to)
    batched_target   = batch_data_list(t_var_to, batch_size)

    return batched_var_to, batched_var_from, batched_var_cond, batched_target


class DirectDataLoader():
    """
    Custom Training Data Loader for direct data loading.
    If the complete dataset fits into GPU VRAM, using this class avoids the overhead of PyTorch's DataLoader class.

    Iterating over this class yields batches of data, where each batch is a tuple of the form (var_to, var_from, var_cond, target).
    var_to and target are always present, while var_from and var_cond are optional, depending on the data provided to the DirectDataLoader.
    present variables are always torch tensors of size batch_size x timesteps x variable_dimensions.
    non-present variables are lists of None of size batch_size.

    This is used to train the pytorch models for estimating transfer entropy (TE) and conditional transfer entropy (CTE).
    """
    def __init__(self, batched_var_to, batched_var_from, batched_var_cond, batched_target, device):
        assert isinstance(device, torch.device), "device must be a torch.device object"
        self.device = device
        
        assert len(batched_var_to) == len(batched_target) == len(batched_var_from) == len(batched_var_cond), f"all input lists must have the same length, but they have lengths {len(batched_var_to)}, {len(batched_target)}, {len(batched_var_from)}, and {len(batched_var_cond)}"
        assert len(batched_var_to) > 0, "input lists must have at least one element"
        
        # data is a list of tuples, each of the form (var_to, var_from, var_cond, target)
        data = []
        for var_to, var_from, var_cond, target in zip(batched_var_to, batched_var_from, batched_var_cond, batched_target):
            batch = ()
            # var_to
            batch += (torch.tensor(var_to, dtype=torch.float32).contiguous().to(device),)
            # var_from
            if var_from is not None:
                batch += (torch.tensor(var_from, dtype=torch.float32).contiguous().to(device),)
            else:
                batch += (None,)
            # var_cond
            if var_cond is not None:
                batch += (torch.tensor(var_cond, dtype=torch.float32).contiguous().to(device),)
            else:
                batch += (None,)
            # target
            batch += (torch.tensor(target, dtype=torch.float32).contiguous().to(device),)
            
            data.append(batch)

        self.data = data
        self.index = 0

        # set attributes for dimensions of the data, this is used to check compatibility with the model and the pair dataset
        self.var_to_dim = batched_var_to[0].shape[2]
        self.var_from_dim = batched_var_from[0].shape[2] if batched_var_from[0] is not None else 0
        self.var_cond_dim = batched_var_cond[0].shape[2] if batched_var_cond[0] is not None else 0

    def __iter__(self):
        self.index = 0
        return self  # Instance itself is the iterator

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.data[index]
        elif isinstance(index, int):
            return self.data[index]
        else:
            raise TypeError('Index must be an integer or slice')
        
    def __str__(self):
        str_ret = f'DirectDataLoader with {len(self.data)} batches of data\n'
        str_ret += f'\t var_to dimension: {self.var_to_dim}\n'
        str_ret += f'\t var_from dimension: {self.var_from_dim}\n'
        str_ret += f'\t var_cond dimension: {self.var_cond_dim}\n'
        return str_ret