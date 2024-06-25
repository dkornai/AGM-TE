import numpy as np
import torch

class DataSet():
    def __init__(self, data):
        # check basic types
        assert isinstance(data, dict), "data must be a dictionary, with variable names as keys and numpy arrays as values"
        
        assert len(data) > 1, "data must have at least two variables"
        
        for key, value in data.items():
            assert isinstance(key, str), "variable names (keys) in data must be strings"
            assert isinstance(value, list), "values in data must be lists of numpy arrays"
        
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

    def __getitem__(self, var):
        return self.data[var]
    
    def __str__(self):
        return f'DataSet with {len(list(self.data.keys()))} variables: {list(self.data.keys())}'
    
    def get_TE_data(self, var_from, var_to):
        """
        Returns data for estimating the transfer entropy from 'var_from' to 'var_to'.
        This means taking values :-1 of var_from, and values 1: of var_to.

        Parameters:
        ----------
        var_from : str
            Variable name for the source variable.
        var_to : str
            Variable name for the target variable.

        Returns:
        -------
        features_var_to : list[np.ndarray]
            Values of var_to[:-1], used in H(target_var_to|features_var_to).
        features_var_both : list[np.ndarray]
            Values of var_to[:-1] and var_from[:-1] concatenated, used in H(target_var_to|features_var_both).
        target_var_to : list[np.ndarray]
            Values of var_to[1:], used in H(target_var_to|features_var_to) and H(target_var_to|features_var_both).
        list
            List of which trajectory a given data point belongs to.
        """
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
            
        return features_var_to, features_var_both, target_var_to, list(range(self.n_traj))

    def get_CTE_data(self, var_from, var_to, var_cond=None):
        """
        Returns data for estimating the conditional transfer entropy from 'var_from' to 'var_to', conditioned on 'var_cond'.

        Parameters:
        ----------
        var_from : str
            Variable name for the source variable.
        var_to : str
            Variable name for the target variable.
        var_cond : str, optional
            Variable name for the conditioning variable. Default is None, which means conditioning on all remaining variables.
        
        Returns:
        -------
        features_var_to : list[np.ndarray]
            Values of var_to[:-1], used in H(target_var_to|features_var_to).
        features_var_both : list[np.ndarray]
            Values of var_to[:-1] and var_from[:-1] concatenated, used in H(target_var_to|features_var_both).
        target_var_to : list[np.ndarray]
            Values of var_to[1:], used in H(target_var_to|features_var_to) and H(target_var_to|features_var_both).
        list
            List of which trajectory a given data point belongs to.
        """
        if var_cond != None:
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

        return features_var_to, features_var_both, target_var_to, list(range(self.n_traj))
    
    def get_TE_dataloaders(self, var_from, var_to, device, slice_size=None):
        """
        Returns dataloaders for estimating the transfer entropy from 'var_from' to 'var_to'.
        This means taking values :-1 of var_from, and values 1: of var_to.

        Parameters:
        ----------
        var_from : str
            Variable name for the source variable.
        var_to : str
            Variable name for the target variable.
        device : torch.device
            Device to run the model on.
        slice_size : int
            if the data comes from a single trajectory, slice the data into chunks of size slice_size. Default is None, used when data comes from multiple trajectories to begin with.

        Returns:
        -------
        dataloaders : list[torch.utils.data.DataLoader]
            List of dataloaders for each type of data.
        """
        features_var_to, features_var_both, target_var_to, traj_idxs = self.get_TE_data(var_from, var_to)
        
        dataloader_1 = data_loader_direct(device, features_var_to, target_var_to, traj_idxs, slice_size)
        dataloader_2 = data_loader_direct(device, features_var_both, target_var_to, traj_idxs, slice_size)

        return dataloader_1, dataloader_2
    
    def get_CTE_dataloaders(self, var_from, var_to, var_cond, device, slice_size=None):
        """
        Returns dataloaders for estimating the conditional transfer entropy from 'var_from' to 'var_to', conditioned on 'var_cond'.

        Parameters:
        ----------
        var_from : str
            Variable name for the source variable.
        var_to : str
            Variable name for the target variable.
        var_cond : str, optional
            Variable name for the conditioning variable. Default is None, which means conditioning on all remaining variables.
        device : torch.device
            Device to run the model on.
        slice_size : int
            if the data comes from a single trajectory, slice the data into chunks of size slice_size. Default is None, used when data comes from multiple trajectories to begin with.

        Returns:
        -------
        dataloaders : list[torch.utils.data.DataLoader]
            List of dataloaders for each type of data.
        """
        features_var_to, features_var_both, target_var_to, traj_idxs = self.get_CTE_data(var_from, var_to, var_cond)
        
        dataloader_1 = data_loader_direct(device, features_var_to, target_var_to, traj_idxs, slice_size)
        dataloader_2 = data_loader_direct(device, features_var_both, target_var_to, traj_idxs, slice_size)

        return dataloader_1, dataloader_2

class DirectDataLoader():
    """
    Custom Training Data Loader for direct data loading.
    """
    def __init__(self, data):
        assert isinstance(data, list), "data must be a list"
        assert len(data) > 0, "data must have at least one element"
        assert all(isinstance(x, tuple) for x in data), "data must be a list of tuples"
        assert all(len(x) == 3 for x in data), "each element in data must be a tuple of length 3"
        assert all(isinstance(x[0], torch.Tensor) for x in data), "first element in each tuple must be a torch.Tensor"
        assert all(isinstance(x[1], torch.Tensor) for x in data), "second element in each tuple must be a torch.Tensor"
        assert all(isinstance(x[2], int) for x in data), "third element in each tuple must be an integer"

        self.data = data
        self.feature_dim = data[0][0].shape[1]  # dimensionality of the input
        self.target_dim = data[0][1].shape[1] # dimensionality of the target
        self.n_traj = len(set([x[2] for x in data])) # number of unique trajectories in the data
        
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


def data_loader_direct(
        device:         torch.device, 
        feature_list:   list[np.ndarray],
        target_list:    list[np.ndarray],
        traj_idxs:      list[int],
        slice_size:     None|int = None
        ) ->            DirectDataLoader:
    """
    Create a list representing the sequences of data to be fed to the model.
    If the complete dataset fits into GPU VRAM, this function speeds up training considerably, compared to using torch DataLoader.

    Parameters:
    ----------
    device : torch.device
        the device to run the model on
    feature_list : list[np.ndarray]
        list of numpy arrays containing the input data
    target_list : list[np.ndarray]
        list of numpy arrays containing the target data
    traj_idxs : list[int]
        list of trajectory indices for each element in the feature_list and target_list
    slice_size : None|int
        if the data comes from a single trajectory, slice the data into chunks of size slice_size. Default is None, used when data comes from multiple trajectories to begin with.

    Returns:
    -------
    sequences : list[tuple[torch.Tensor, torch.Tensor, int]]
        list of tuples containing the feature and target tensors, and the trajectory index for each element in the feature_list and target_list
    """

    assert isinstance(device, torch.device), "device must be a torch.device object"

    # if no slicing is required, just create a list of tuples from the original data
    if slice_size == None:
        sequences = []
        for feat, targ, traj_idx in zip(feature_list, target_list, traj_idxs):
            
            # move to devices as contiguous tensors
            feature_seq = torch.tensor(feat, dtype=torch.float32).contiguous().to(device)
            target_seq = torch.tensor(targ, dtype=torch.float32).contiguous().to(device)
            
            sequences.append((feature_seq, target_seq, traj_idx))

    # if slicing is required (data comes from a single trajectory), slice the data into chunks of size slice_size
    elif slice_size != None:
        assert len(set(traj_idxs)) == 1, "slicing can only be used if all data comes from a single trajectory"
        assert isinstance(slice_size, int), "slice_size must be an integer"
        assert slice_size > 0, "slice_size must be a positive integer"
        assert slice_size < feature_list[0].shape[0], "slice_size must be smaller than the number of timesteps in the trajectory"
        
        sequences = []

        # ensure tensors are contiguous in memory
        features_contiguous = torch.tensor(feature_list[0], dtype=torch.float32).contiguous()
        targets_contiguous  = torch.tensor(target_list[0], dtype=torch.float32).contiguous()

        num_timesteps = feature_list[0].shape[0]
        
        for start_idx in range(0, num_timesteps, slice_size):
            end_idx = min(start_idx + slice_size, num_timesteps)
            
            # move to devices as contiguous tensors
            feature_seq   = features_contiguous[start_idx:end_idx].contiguous().to(device)
            target_seq    = targets_contiguous[start_idx:end_idx].contiguous().to(device)
            
            sequences.append((feature_seq, target_seq, 0))
    
    return DirectDataLoader(sequences)