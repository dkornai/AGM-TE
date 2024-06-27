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

        - features_var_to is Y- (the past of Y), used as input when estimating H(Y+|Y-)
        - features_var_both is X- and Y- (the past of both X and Y), used as input when estimating H(Y+|X-,Y-)
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
        features_var_to :   list[np.ndarray]
            Values of var_to[:-1], used in H(target_var_to|features_var_to).
        features_var_both : list[np.ndarray]
            Values of var_to[:-1] and var_from[:-1] concatenated, used in H(target_var_to|features_var_both).
        target_var_to :     list[np.ndarray]
            Values of var_to[1:], used in H(target_var_to|features_var_to) and H(target_var_to|features_var_both).
        traj_idxs :         list[int]
            List containing the trajectory index of each array in the lists
        """

        assert var_from in self.data.keys(), f"variable {var_from} not found in the dataset"
        assert var_to in self.data.keys(), f"variable {var_to} not found in the dataset"
        assert var_from != var_to, "var_from and var_to must be different"

        features_var_to   = []
        features_var_both = []
        target_var_to = []
        for traj_idx in range(self.n_traj):
            f_var_to      = self.data[var_to][traj_idx][:-1]
            f_var_both    = np.concatenate([f_var_to, self.data[var_from][traj_idx][:-1]], axis=1)
            t_var_to      = self.data[var_to][traj_idx][1:]

            features_var_to.append(f_var_to)
            features_var_both.append(f_var_both)
            target_var_to.append(t_var_to)
        
        traj_idxs = list(range(self.n_traj))

        return features_var_to, features_var_both, target_var_to, traj_idxs

    def get_CTE_data(self, var_from, var_to, var_cond='remaining'):
        """
        Returns datasets for estimating the conditional transfer entropy (CTE) of the variable 'var_from' to the variable 'var_to', conditioned on the variable (or set of variables) 'var_cond'.

        Say var_from is X, var_to is Y, and var_cond is Z, then these dataloaders allow us to estimate the CTE:

        T(X -> Y|Z) = H(Y+|Y-,Z-) - H(Y+|X-,Y-,Z-)
        
        - features_var_to is Y-, Z- (the past of Y and Z), used as input when estimating H(Y+|Y-,Z-)
        - features_var_both is X-, Y-, and Z- (the past of  X, Y, and Z), used as input when estimating H(Y+|X-,Y-,Z-)
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
        features_var_to :   list[np.ndarray]
            Values of var_to[:-1], used in H(target_var_to|features_var_to).
        features_var_both : list[np.ndarray]
            Values of var_to[:-1] and var_from[:-1] concatenated, used in H(target_var_to|features_var_both).
        target_var_to :     list[np.ndarray]
            Values of var_to[1:], used in H(target_var_to|features_var_to) and H(target_var_to|features_var_both).
        traj_idxs :         list[int]
            List containing the trajectory index of each array in the lists
        """
        assert self.n_vars > 2, "dataset must have more than 2 variables to condition on a third variable"
        assert var_from in self.data.keys(), f"variable {var_from} not found in the dataset"
        assert var_to in self.data.keys(), f"variable {var_to} not found in the dataset"
        assert var_from != var_to, "var_from and var_to must be different"
        assert var_cond != var_from, "var_cond cannot be the same as var_from"
        assert var_cond != var_to, "var_cond cannot be the same as var_to"

        if var_cond != 'remaining':
            raise NotImplementedError("Conditioning to specific variables is not yet implemented.")
        
        features_var_to   = []
        features_var_both = []
        target_var_to = []

        for traj_idx in range(self.n_traj):
            f_var_to   = self.data[var_to][traj_idx][:-1]
            for var in self.data.keys():
                if var not in [var_from, var_to]:
                    f_var_to = np.concatenate([f_var_to, self.data[var][traj_idx][:-1]], axis=1)

            f_var_both = np.concatenate([f_var_to, self.data[var_from][traj_idx][:-1]], axis=1)
            t_var_to = self.data[var_to][traj_idx][1:]

            features_var_to.append(f_var_to)
            features_var_both.append(f_var_both)
            target_var_to.append(t_var_to)

        traj_idxs = list(range(self.n_traj))

        return features_var_to, features_var_both, target_var_to, traj_idxs
    
    def get_TE_dataloaders(self, device, var_from, var_to, batch_size):
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
            Dataloader containing features_var_to and target_var_to.
        dataloader_2 :  DirectDataLoader
            Dataloader containing features_var_both and target_var_to.
        """
        features_var_to, features_var_both, target_var_to, traj_idxs = self.get_TE_data(var_from, var_to)
        
        dataloader_1 = get_DirectDataLoader_fromdata(device, features_var_to, target_var_to, traj_idxs, batch_size)
        dataloader_2 = get_DirectDataLoader_fromdata(device, features_var_both, target_var_to, traj_idxs, batch_size)

        return dataloader_1, dataloader_2
    
    def get_CTE_dataloaders(self, device, var_from, var_to, var_cond, batch_size):
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
            Dataloader containing features_var_to and target_var_to.
        dataloader_2 :  DirectDataLoader
            Dataloader containing features_var_both and target_var_to.
        """
        features_var_to, features_var_both, target_var_to, traj_idxs = self.get_CTE_data(var_from, var_to, var_cond)
        
        dataloader_1 = get_DirectDataLoader_fromdata(device, features_var_to, target_var_to, traj_idxs, batch_size)
        dataloader_2 = get_DirectDataLoader_fromdata(device, features_var_both, target_var_to, traj_idxs,  batch_size)

        return dataloader_1, dataloader_2

class DirectDataLoader():
    """
    Custom Training Data Loader for direct data loading.
    If the complete dataset fits into GPU VRAM, using this class avoids the overhead of PyTorch's DataLoader class.
    """
    def __init__(self, data):
        assert isinstance(data, list), "data must be a list"
        assert len(data) > 0, "data must have at least one element"
        assert all(isinstance(x, tuple) for x in data), "data must be a list of tuples"
        assert all(len(x) == 3 for x in data), "each element in data must be a tuple of length 3"
        assert all(isinstance(x[0], torch.Tensor) for x in data), "first element in each tuple must be a torch.Tensor"
        assert all(isinstance(x[1], torch.Tensor) for x in data), "second element in each tuple must be a torch.Tensor"
        #assert all(isinstance(x[2], int) for x in data), "third element in each tuple must be an integer"

        self.data = data
        self.feature_dim = data[0][0].shape[-1]  # dimensionality of the input
        self.target_dim = data[0][1].shape[-1] # dimensionality of the target
        #self.n_traj = len(set([x[2] for x in data])) # number of unique trajectories in the data
        
        self.index = 0

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


def get_DirectDataLoader_fromdata(
        device:         torch.device, 
        feature_list:   list[np.ndarray],
        target_list:    list[np.ndarray],
        traj_idxs:      list[int],
        batch_size:     int = 1,
        ) ->            DirectDataLoader:
    """
    Create a DirectDataLoader object (used to train the dynamics models) from raw numpy arrays.    

    Parameters:
    ----------
    device : torch.device
        the device to run the model on
    feature_list : list[np.ndarray]
        list of numpy arrays containing the input data. Each array should represent a unique trajectory of the system.
    target_list : list[np.ndarray]
        list of numpy arrays containing the target data. Each array should represent a unique trajectory of the system.
    traj_idxs : list[int]
        which unique trajectory does the n-th element in feature_list and target_list belong to.
    slice_size : None|int
        if the data comes from a single trajectory, slice the data into chunks of size slice_size. Default is None, used when data comes from multiple trajectories to begin with.
    batch_size : None|int
        batch 'batch_size' sequences into 3d tensors. Default is 1, which means no batching effectively.

    Returns:
    -------
    : DirectDataLoader
        DirectDataLoader object containing references to the torch tensors on the declared device.
    """

    assert isinstance(device, torch.device), "device must be a torch.device object"
    # check data types
    assert isinstance(feature_list, list), "feature_list must be a list"
    assert all(isinstance(x, np.ndarray) for x in feature_list), "all elements in feature_list must be numpy arrays"
    assert isinstance(target_list, list), "target_list must be a list"
    assert all(isinstance(x, np.ndarray) for x in target_list), "all elements in target_list must be numpy arrays"
    assert isinstance(traj_idxs, list), "traj_idxs must be a list"
    assert all(isinstance(x, int) for x in traj_idxs), "all elements in traj_idxs must be integers"
    
    # check trajectory indices
    assert len(feature_list) == len(traj_idxs), "feature_list and traj_idxs must have the same length"
    assert (len(traj_idxs) == 1) or (len(traj_idxs) == len(set(traj_idxs))), "all elements in traj_idxs must be unique."
    if len(traj_idxs) == 1:
        assert traj_idxs[0] == 0, "if there is only one trajectory, the trajectory index must be 0"
    else:
        assert list(np.arange(len(traj_idxs))) == traj_idxs, "trajectory indices must be consecutive integers starting from 0"
    
    # check data shapes 
    assert all(x.ndim == 2 for x in feature_list), "all numpy arrays in feature_list must be 2D (timesteps x features)"
    assert all(x.ndim == 2 for x in target_list), "all numpy arrays in target_list must be 2D (timesteps x features)"
    assert len(feature_list) == len(target_list), "feature_list and target_list must have the same length"
    n_vars_feature = [x.shape[1] for x in feature_list]
    assert len(set(n_vars_feature)) == 1, "all numpy arrays in feature_list must have the same number of columns (dimensions)"
    n_vars_target = [x.shape[1] for x in target_list]
    assert len(set(n_vars_target)) == 1, "all numpy arrays in target_list must have the same number of columns (dimensions)"
    len_feat_traj = [x.shape[0] for x in feature_list]
    len_targ_traj = [x.shape[0] for x in target_list]
    assert len(set(len_feat_traj)) == 1, "all numpy arrays in feature_list must have the same number of rows (timesteps), this is to ensure batching works correctly"
    assert len(set(len_targ_traj)) == 1, "all numpy arrays in target_list must have the same number of rows (timesteps), this is to ensure batching works correctly"
    assert len_feat_traj == len_targ_traj, "the n-th element of feature_list and target_list must have the same number of timesteps for all n"

    # check batch_size
    assert isinstance(batch_size, int), "batch_size must be an integer"
    if isinstance(batch_size, int): 
        assert batch_size >= 1, "batch_size must greater than 1"
        assert len(feature_list)%batch_size == 0, f"number of unique trajectories {len(feature_list)} must be divisible by batch_size"

    # batch 2D tensors into 3D tensors, and move to device as contiguous tensors
    data_tuples = []
    batch_counter = 0 # keeps track of the index in the current batch
    for feat, targ, traj_idx in zip(feature_list, target_list, traj_idxs):
        if batch_counter == 0:
            # start a new batch as all 0s
            feature_batch = np.zeros((batch_size, feat.shape[0], feat.shape[1]))
            target_batch = np.zeros((batch_size, targ.shape[0], targ.shape[1]))
            traj_idx_batch = []
        
        if batch_counter < batch_size:
            # append to the batch (starting from index 0)
            feature_batch[batch_counter] = feat
            target_batch[batch_counter] = targ
            traj_idx_batch.append(traj_idx)
            batch_counter += 1
        
        if batch_counter == batch_size:
            # move to device as contiguous tensors
            feature_seq = torch.tensor(feature_batch, dtype=torch.float32).contiguous().to(device)
            target_seq = torch.tensor(target_batch, dtype=torch.float32).contiguous().to(device)
            data_tuples.append((feature_seq, target_seq, traj_idx_batch))
            batch_counter = 0 # reset the batch counter to 0 and start a new batch
            continue

    # convert into a DirectDataLoader object
    return DirectDataLoader(data_tuples)