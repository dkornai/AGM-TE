# agm_te

Approximate Generative Model estimation of Transfer Entropy or `agm_te` is a python package for estimating transfer entropy between variables of interest from time series. Transfer entropy estimation can be used in the process of causal discovery. 

Transfer entropy from a variable $X$ to a variable $Y$, given an observation of length $T$, is estimated as: 

$$
\hat{\mathcal{T}}_{X \to Y }&= \frac{1}{T}\sum\nolimits_{t=1}^{T}[-\log q_1(y_{t+1}|\mathbf{y_t})] - \frac{1}{T}\sum\nolimits_{t=1}^{T}[-\log q_2(y_{t+1}|\mathbf{y_t}, \mathbf{x_t})]
$$

where $q_1$ and $q_2$ are competing probabilistic forecasting models over $y_{t+1}$, given either the past of $Y$ [$\mathbf{y_t}$], or the past of both $Y$ and $X$.  [ $\mathbf{y_t}, \mathbf{x_t}$ ]

## Installation

`agm_te` is installable as a package. Clone the repository, navigate to the folder, and run `pip install .` to install the package. 

## TE Estimation Workflow

For a specific example of a TE estimation workflow, see the `\demo\test_sim_bivar.ipynb` file. The following is just a general description.

### 1) Data Processing

Time series observations are handled using the `DataSet` class. The raw data is stored as a dictionary, with variable names as keys and lists of numpy arrays as values. The class provides methods which yield the subsets of data for estimating transfer entropy (TE) and conditional transfer entropy (CTE) between variables in the `DataSet`.

### 2A) Model Type Specification

The Approximate Generative Model [AGM] from which the method is named are probabilistic time series forecasters driven by neural networks. An AGM is composed of a neural network dynamics model, and a probabilistic observation model. Parameters for the dynamics model include the type [MLP, GRU, RNN, LSTM], the size and number of hidden layers. The type of the observation model [e.g. Gaussian, Poisson] is also determined at this stage. See the docstring of the `ApproxGenModel` class for more details. This should result in a dictionary of model parameters.

### 2B) Training Parameter Specification

Specify the batch sizes, number of epochs, etc that will be used during the training process. This should result in a dictionary of training parameters.

### 3) Estimate Transfer entropy

Use the `agm_estimate_TE(dataset=, model_params=, train_params=, var_from=, var_to=)` method to  estimate TE using the two competing models


