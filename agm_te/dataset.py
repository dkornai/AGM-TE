import numpy as np
import torch

class DataSet():
    """
    Class which holds sampled data for a system with multiple variables.
    The raw data is stored as a dictionary, with variable names as keys and lists of numpy arrays as values.
    The class provides methods which yield the subsets of data for estimating transfer entropy (TE) and conditional transfer entropy (CTE) between variables in the DataSet.
    Some methods yield AgmTrainingData objects, which contain the formatted and batched data ready for training the pytorch models.
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
        target_var_to : list[np.ndarray]
            Values of var_to[1:], forming the sample of Y+.
        """

        assert self.n_vars > 2, "dataset must have more than 2 variables to condition on a third variable"
        assert var_from in self.data.keys(), f"variable {var_from} not found in the dataset"
        assert var_to in self.data.keys(), f"variable {var_to} not found in the dataset"
        assert var_cond in self.data.keys() or var_cond == 'remaining', f"variable {var_cond} not found in the dataset"
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
            if var_cond == 'remaining':
                f_var_cond = np.concatenate([self.data[var][traj_idx][:-1] for var in self.data.keys() if var not in [var_from, var_to]], axis=-1)
            else:
                f_var_cond = self.data[var_cond][traj_idx][:-1]
            t_var_to = self.data[var_to][traj_idx][1:]

            feat_var_to.append(f_var_to)
            feat_var_from.append(f_var_from)
            feat_var_cond.append(f_var_cond)
            target_var_to.append(t_var_to)

        return feat_var_to, feat_var_from, feat_var_cond, target_var_to

    def get_TE_dataloaders(self, var_from, var_to):
        """
        Returns dataloaders for estimating the transfer entropy (TE) from the variable 'var_from' to the variable 'var_to'.

        Say var_from is X, and var_to is Y, then:

        T(X -> Y) = H(Y+|Y-) - H(Y+|X-,Y-)

        - dataloader_1 is for estimating the conditional entropy H(Y+|Y-)
        - dataloader_2 is for estimating the conditional entropy H(Y+|X-,Y-)

        Parameters:
        ----------
        var_from :      str
            Variable name for the source variable.
        var_to :        str
            Variable name for the target variable.
        
        Returns:
        -------
        dataloader_1 :  AgmTrainingData
            Dataloader containing Y- as features and Y+ as target.
        dataloader_2 :  AgmTrainingData
            Dataloader containing X- and Y- as features, and Y+ as target.
        """
        feat_var_to, feat_var_from, target_var_to = self.get_TE_data(var_from, var_to)

        # dataloader for H(Y+|Y-), batch_feat_var_cond is a list of None and batch_feat_var_from is a list of None
        batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to = \
            prepare_training_data(feat_var_to, target_var_to)
        dataloader_1 = AgmTrainingData(batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to)
        
        # dataloader for H(Y+|X-,Y-), batch_feat_var_cond is a list of None
        batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to = \
            prepare_training_data(feat_var_to, target_var_to, f_var_from=feat_var_from)
        dataloader_2 = AgmTrainingData(batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to)

        return dataloader_1, dataloader_2
    
    def get_CTE_dataloaders(self, var_from, var_to, var_cond):
        """
        Returns dataloaders for estimating the conditional transfer entropy (CTE) of the variable 'var_from' to the variable 'var_to', conditioned on the variable (or set of variables) 'var_cond'.

        Say var_from is X, var_to is Y, and var_cond is Z, then these dataloaders allow us to estimate the CTE:

        T(X -> Y|Z) = H(Y+|Y-,Z-) - H(Y+|X-,Y-,Z-)
        
        - dataloader_1 is for estimating the conditional entropy H(Y+|Y-,Z-)
        - dataloader_2 is for estimating the conditional entropy H(Y+|X-,Y-,Z-)

        Parameters:
        ----------
        var_from :      str
            Variable name for the source variable.
        var_to :        str
            Variable name for the target variable.
        var_cond :      str, optional
            Variable name for the conditioning variable. Default is None, which means conditioning on all remaining variables.
        
        Returns:
        -------
        dataloader_1 :  AgmTrainingData
            Dataloader containing Y- and Z- as features, and Y+ as target.
        dataloader_2 :  AgmTrainingData
            Dataloader containing X-, Y-, and Z- as features, and Y+ as target.
        """
        feat_var_to, feat_var_from, feat_var_cond, target_var_to = self.get_CTE_data(var_from, var_to, var_cond)
        
        
        # dataloader for H(Y+|Y-,Z-)
        batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to = \
            prepare_training_data(feat_var_to, target_var_to, f_var_cond=feat_var_cond)
        dataloader_1 = AgmTrainingData(batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to)
        
        # dataloader for H(Y+|X-,Y-,Z-)
        batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to = \
            prepare_training_data(feat_var_to, target_var_to, f_var_from=feat_var_from, f_var_cond=feat_var_cond)
        dataloader_2 = AgmTrainingData(batch_feat_var_to, batch_feat_var_from, batch_feat_var_cond, batch_target_var_to)

        return dataloader_1, dataloader_2

def stack_data_list(data_list: list[np.ndarray]) -> np.ndarray:
    """
    Take a list of t*d numpy arrays (each a unique trajectory), and return a combined concatenated array.
    """
    assert isinstance(data_list, list), "data_list must be a list"
    assert all(isinstance(x, np.ndarray) for x in data_list), "all elements in data_list must be numpy arrays"
    assert all(x.ndim == 2 for x in data_list), "all numpy arrays in data_list must be 2D (timesteps x features)"
    assert all(x.shape == data_list[0].shape for x in data_list), "all numpy arrays in data_list must have the same number of rows (timesteps) and columns (features)"

    timesteps, dim = data_list[0].shape
    n_traj = len(data_list)

    data = np.zeros((n_traj, timesteps, dim), dtype=np.float32)
    for i, element in enumerate(data_list):
        data[i, :, :] = np.array(element, dtype=np.float32)

    return data 
 
def prepare_training_data(f_var_to, t_var_to, f_var_from=None, f_var_cond=None):
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
    
    feature_var_to   = stack_data_list(f_var_to)
    feature_var_from = stack_data_list(f_var_from) if f_var_from is not None else None
    feature_var_cond = stack_data_list(f_var_cond) if f_var_cond is not None else None
    target_var_to    = stack_data_list(t_var_to)

    return feature_var_to, feature_var_from, feature_var_cond, target_var_to


class AgmTrainingData():
    """
    Custom Training Data class. 

    Attributes:

    - self.input is a 3D numpy array of shape (n_traj, n_timesteps, feature_dim)
    - self.target is a 3D numpy array of shape (n_traj, n_timesteps, target_dim)
    - self.var_to_dim is the dimension of the target variable
    - self.var_from_dim is the dimension of the source variable (0 if no source variable is used)
    - self.var_cond_dim is the dimension of the conditioning variable (0 if no conditioning variable is used)
    - self.feature_dim is the total dimension of the input data (var_to_dim + var_from_dim + var_cond_dim)
    - self.target_dim is the total dimension of the target data (var_to_dim)
    
    """
    def __init__(self, f_var_to, f_var_from, f_var_cond, target_var_to):
        
        n_traj_per_dataset = [element.shape[0] for element in [f_var_to, f_var_from, f_var_cond, target_var_to] if isinstance(element, np.ndarray) ]
        assert len(set(n_traj_per_dataset)) == 1, 'all datasets must have the same number of trajectories'
        timesteps_per_dataset = [element.shape[1] for element in [f_var_to, f_var_from, f_var_cond, target_var_to] if isinstance(element, np.ndarray) ]
        assert len(set(timesteps_per_dataset)) == 1, 'all datasets must have the same number of timesteps'
        
        # input data (features)
        input = np.array(f_var_to, dtype=np.float32)
        if f_var_from is not None: input = np.concatenate((input, np.array(f_var_from, dtype=np.float32)), axis=-1)
        if f_var_cond is not None: input = np.concatenate((input, np.array(f_var_cond, dtype=np.float32)), axis=-1)
    
        # target data
        target = np.array(target_var_to, dtype=np.float32)

        self.input = input
        self.target = target

        # set attributes for dimensions of the data, this is used to check compatibility with the model and the pair dataset
        self.var_to_dim     = f_var_to.shape[2]
        self.var_from_dim   = f_var_from.shape[2] if f_var_from is not None else 0
        self.var_cond_dim   = f_var_cond.shape[2] if f_var_cond is not None else 0

        # set attributes for total input and target dimensions (used for checking the match to a given model instance)
        self.feature_dim    = self.var_to_dim + self.var_from_dim + self.var_cond_dim
        self.target_dim     = self.var_to_dim
        
    def __str__(self):
        return f'AgmTrainingData with feature of size ({self.input.shape[0]}, {self.input.shape[1]}, {self.input.shape[2]}) and target of size ({self.target.shape[0]}, {self.target.shape[1]}, {self.target.shape[2]})'