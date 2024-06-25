import numpy as np

class DataSet():
    def __init__(self, data, trajectory_length=None):
        # check basic types
        assert isinstance(data, dict), "data must be a dictionary, with variable names as keys and numpy arrays as values"
        assert len(data) > 1, "data must have at least two variables"
        for key, value in data.items():
            assert isinstance(key, str), "variable names (keys) in data must be strings"
            assert isinstance(value, np.ndarray), "values in data must be numpy arrays"
        # check shapes of numpy arrays
        for key, value in data.items():
            assert value.ndim == 2, "all numpy arrays in data must be 2D, with timesteps as axis 0 (rows), and features on axis 1 (columns)"
        len_data = [value.shape[0] for value in data.values()]
        assert len(set(len_data)) == 1, "all numpy arrays in data must have the same number of timesteps (rows)"
        assert trajectory_length == None or isinstance(trajectory_length, int), "trajectory_length must be an integer or None"

        self.data = data 
        self.trajectory_length = trajectory_length

    def __getitem__(self, var):
        return self.data[var]
    
    def __str__(self):
        return f'DataSet with {len(list(self.data.keys()))} variables: {[list(self.data.keys())]}'
    
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
        features_var_to : numpy array
            Values of var_to[:-1], used in H(target_var_to|features_var_to).
        features_var_both : numpy array
            Values of var_to[:-1] and var_from[:-1] concatenated, used in H(target_var_to|features_var_both).
        target_var_to : numpy array
            Values of var_to[1:], used in H(target_var_to|features_var_to) and H(target_var_to|features_var_both).
        """
        features_var_to   = self.data[var_to][:-1]
        features_var_both = np.concatenate([features_var_to, self.data[var_from][:-1]], axis=1)
        target_var_to = self.data[var_to][1:]
        return features_var_to, features_var_both, target_var_to

    def get_CTE_data(self, var_from, var_to, var_cond=None):
        if var_cond != None:
            raise NotImplementedError("Conditioning to specific variables is not yet implemented.")
        
        features_var_to   = self.data[var_to][:-1]
        for var in self.data.keys():
            if var not in [var_from, var_to]:
                features_var_to = np.concatenate([features_var_to, self.data[var][:-1]], axis=1)

        features_var_both = np.concatenate([features_var_to, self.data[var_from][:-1]], axis=1)

        target_var_to = self.data[var_to][1:]
        return features_var_to, features_var_both, target_var_to