import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import os 
from tqdm.auto import tqdm # tqdm is a library for creating progress bars in Python, useful for tracking the progress of loops and iterations
# required packages: mostly pytorch (torch) for deep learning methods and pandas and numpy for data handling and optimized vector operations, respectively
# if run locally, requires all this packages to be installed. In particular, requires the CUDA enabled pytorch. This in turn requires an Nvidia GPU, the CUDA toolkit and the Graphics Drivers

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available")

if device != "cuda":
    assert False, "Aborting execution. Do not torture your CPU, run this script with a GPU. Connect to T4 on Google Colab or run locally on a GPU equipped system.)" # DO NOT TORTURE YOUR CPU!!

# Debugging tool for checking backpropagation gradients of the computational graph (do not worry about this)

#torch.autograd.set_detect_anomaly(True)

torch.set_float32_matmul_precision("high")

##########################################################
##########################################################
#### CODE FOR SOLVING KRUSELL-SMITH VIA DEEP LEARNING ####
##########################################################
##########################################################


"""

Pytorch implementation of the DeepHAM algorithm by Han, Yang, and E (2021) for the Krusell-Smith model (1998). The replicated configuration is "game_nn_n50.json".
Please check original code at: https://github.com/frankhan91/DeepHAM
Code by: Marko Irisarri (UPF (PhD) / Manchester (AP))

"""

"""

Overall Structure of the code:

- Model Class: contains the declaration and definition of the inputs of the model and the methods (functions) required to solve the model
- Main(): DeepHam algorithm:
    - Step 1: Simulate economy given policy function guesses and obtain invariant distribution of the econmy
    - Step 2: Given the invariant distribution according to the guess on the policy functions, train the value NN via regression by drawing samples and computing the realized value over those trajectories
    - Step 3: Given the current guesses on policy functions and value function, draw samples from the ergodic distribution and train policy function NN along the simulated paths by building the computational graph and taking the derivatives w.r.t NN parameters
- "As is", the ergodic distribution is similar to the one obtained via the KS method Den Haan (2010) (Maliar KS implementation) by step 50 in about 1 hour of execution on Google Collab's T4 GPU.

"""

"""

Overall Comments:

  - Keep track of VRAM consumption! Exceeding VRAM will result in either (i) abrupt execution abortion (ii) using the shared CPU-GPU memory (slow PCI-E transfers of memory).
  - The output is quite verbose, contains: histograms for the individual distribution of assets per iteration of the DeepHam algorithm.
  - When working with pytorch, be extremely careful with in-place operators (=*, =+, clamp_max_, ...) as they might break the BPTT (Backpropagation-Through-Time) gradient flow!
  - Getting the right solution requires extensive testing of normalization, width and depth of NN, tuning the settings of the optimizer...
  - The Assets Policy NN employs a sigmoid output layer, which is constrained to be between 0 and 1. This will tell us the fraction of total resources that go to next period's assets and guarantees that c > 0.
  - [-1,1] normalization is employed 2*(x-min)/(max-min)-1. This is common practice with hidden tanh() activation functions.
  - If debugging is desired, then turning the torchscript compilation off is required.

"""

def set_seed(seed: int):

    ### Set all random seeds for reproducibility ###

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # NumPy
    np.random.seed(seed)

    # Python's built-in random
    random.seed(seed)

    # cuDNN configurations (deterministic mode)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed for reproducibility
set_seed(42)


class KrusellSmith(torch.nn.Module):

    """

    This a python class, it allows to create objects (instances of a class) with their own attributes (data) and methods (functions)
    Our Class will be called Krusell Smith, and an instance of this class will be the "model" that we define in the main() function

    Inputs:
    -torch.nn.Module: this means that our class will inherit functionalities from the torch.nn.Module class, so that we can use deep learning methods

    """

    def __init__(self): # This is the constructor for our class (Krusell Smith)
        super().__init__() # This calls the constructor of torch.nn.Module (allows us to actually use the deep learning methods)
        self.device = device # This is the device where we will perform the computations, and defined globally above. For optimal performance, this should be set to "cuda" (requires Nvidia GPU and the adequate graphics drivers and CUDA toolkit installed)

        # Standard Krusell-Smith parameters

        # "Self" means that it will be a parameter or element of the created instance of the class

        self.beta = torch.tensor(0.99, device=self.device, dtype=torch.float32)
        self.alpha = torch.tensor(0.36, device=self.device, dtype=torch.float32)
        self.delta = torch.tensor(0.025, device=self.device, dtype=torch.float32)
        self.agg_productivity = torch.tensor([0.99, 1.01], device=self.device, dtype=torch.float32)
        self.mu = torch.tensor(0.15, device=self.device, dtype=torch.float32)
        self.l_bar = torch.tensor(1.0 / 0.9, device=self.device, dtype=torch.float32)
        self.idio_productivity = torch.tensor([0, 1], device=self.device, dtype=torch.float32)
        self.ur_b = torch.tensor(0.1, device=self.device, dtype=torch.float32) # unemployment rate in the bad state
        self.ur_g = torch.tensor(0.04, device=self.device, dtype=torch.float32) # unemployment rate in the good state
        self.labour_taxes = torch.tensor([self.mu * self.ur_b / (self.l_bar * (1.0 - self.ur_b)),
                                           self.mu * self.ur_g / (self.l_bar * (1.0 - self.ur_g))],
                                          device=self.device, dtype=torch.float32) # labour taxes in the bad and good state
        self.k_ss = ((1.0 / self.beta - (1.0 - self.delta)) / self.alpha) ** (1.0 / (self.alpha - 1.0))
        self.L = torch.tensor([0.9, 0.96], device=self.device, dtype=torch.float32)
        self.prob_trans_agg = torch.tensor([[0.875, 0.125], [0.125, 0.875]], device=self.device, dtype=torch.float32)
        self.prob_trans_idio = torch.tensor( # Joint transition matrix for idio. and agg. shocks (taken directly from the DeepHAM paper)
            [
                [0.525, 0.35, 0.03125, 0.09375],
                [0.038889, 0.836111, 0.002083, 0.122917],
                [0.09375, 0.03125, 0.291667, 0.583333],
                [0.009115, 0.115885, 0.024306, 0.850694] # note that the idio and agg shocks are not independent
            ], device=self.device, dtype=torch.float32
        )

        # Compute the invariant distribution for the idiosyncratic and aggregate shocks

        self.stationary_agg = self.compute_invariant_dist(self.prob_trans_agg)
        self.stationary_idio = self.compute_invariant_dist(self.prob_trans_idio)

        """

        This set of parameters is part of the NN model proper and should therefore be scalars, not tensors (for compatibility with torchscript)

        """

        ### NN structure ###

        self.n_agents = 50 # Simulate an economy with 50 agents (the more agents, the closer to the theoretical atomistic behaviour) 50 is the default in the original paper
        self.n_features = 4 # This is the individual state space, defined over individual assets and productivity, and aggregate productivity and capital

        # --- Features: 0: Individual Capital, 1: Idiosyncratic Productivity, 2: Aggregate Productivity, 3: Aggregate Capital --- #

        self.output_dim = 1 # The NN will return a one dimensional scalar (the share that goes to next period capital out of the total resources in the budget constraint)
        self.size_hidden_layers = 24 # This is the number of neurons per hidden layer of the NN (more on this later)

        ### MC Paths Lengths by Step ###

        self.n_periods_simulation = 11000 # Length of simulation in step 1
        self.burn_in = 6000 # Burn-in period for the simulation (step 1)
        self.n_periods_value = 550 # Length of the simulation to approximate the value function (step 2)
        self.n_periods_monte_carlo = 150 # Length of the computatinal graph for (step 3)

        ### MC Paths Batches by Step ###

        self.n_batches_monte_carlo = 350 # Number of economies simulated, each with self.n_agents. This is important for the NN to learn the actual policy function, need to simulate many economies
        self.n_batches_value = 5000 # Likewise, to train the value function, need to simulate many economies, each with self.n_agents
        self.n_batches_simulation = 1 # For the simulation (step 1) we only require 1 economy

        """

        Preallocation of tensors for each step of the algorithm

        """

        ### Preallocate tensors for the simulation (Step 1) ###

        self.inputs_simulation = torch.zeros([self.n_batches_simulation, self.n_agents, self.n_periods_simulation, self.n_features], device=self.device, dtype=torch.float32) # Preallocated tensor (stored on GPU, of type float32 for best performance) for the simulation
        self.inputs_simulation[:, :, :, 0] = self.k_ss * torch.ones_like(self.inputs_simulation[:, :, :, 0]) # Set initial individual capital to the steady - state value of capital
        self.agg_simulation = torch.zeros([self.n_batches_simulation, self.n_periods_simulation], device=self.device, dtype=torch.float32) # Preallocated tensor for aggregate productivity over the simulation (step 1)

        ### Preallocate tensors for the Value Function Training (Step 2) ###

        self.inputs_value = torch.zeros(self.n_batches_value, self.n_agents, self.n_periods_value, self.n_features, device = device, dtype = torch.float32) # Same as above for training the value function (step 2)
        self.agg_value = torch.zeros(self.n_batches_value, self.n_periods_value, device = device, dtype = torch.float32) # Same as above, aggregate shocks in step 2

        ### Preallocate tensors for the Policy Training (Step 3) ###

        self.inputs_monte_carlo = torch.zeros([self.n_batches_monte_carlo, self.n_agents, self.n_periods_monte_carlo, self.n_features], device=self.device, dtype=torch.float32, requires_grad=False) # Same as above but for step 3
        self.agg_monte_carlo = torch.zeros([self.n_batches_monte_carlo, self.n_periods_monte_carlo], device=self.device, dtype=torch.float32) # Aggregate productivity over the different economies for step 3
        self.util_value = torch.zeros(self.n_batches_value,self.n_agents, device = device, dtype = torch.float32)

        ### Preallocate tensors for IRF generation ###
        self.n_batches_irf = 5000
        self.n_periods_irf = 100
        self.inputs_irf = torch.zeros(self.n_batches_irf, self.n_agents, self.n_periods_irf, self.n_features, device=self.device, dtype=torch.float32)
        self.agg_shocks_irf = torch.zeros(self.n_batches_irf, self.n_periods_irf, device=self.device, dtype=torch.float32)
        self.agg_capital_irf = torch.zeros(self.n_batches_irf, self.n_periods_irf, device=self.device, dtype=torch.float32)
        self.agg_output_irf = torch.zeros(self.n_batches_irf, self.n_periods_irf, device=self.device, dtype=torch.float32)
        self.agg_interest_rate_irf = torch.zeros(self.n_batches_irf, self.n_periods_irf, device=self.device, dtype=torch.float32)

        ### Normalization Tensors : Critical for well-behaved training of the NNs ###

        # These are taken from the standard solution method
        # Why fixed bounds? After extensive testing, find that this leads to the most stable training compared to endogenous adaptive bounds

        self.normalization_min = torch.tensor([10.0, 0.0, 0.0, 30.0], device=self.device, dtype=torch.float32) # Minimum values per feature (k,z,Z,K)
        self.normalization_max = torch.tensor([150.0, 1.0, 1.0, 50.0], device=self.device, dtype=torch.float32) # Maximum values per feature (k,z,Z,K)

        # Values for normalizing the output from the Value Function (step 2)

        self.value_min = torch.zeros(1, device = device, dtype = torch.float32)
        self.value_max = torch.zeros(1, device = device, dtype = torch.float32)

        """

        Validation tensor: this is a tensor to evaluate the policy for assets post step 3 and verify our results in debugging (validation, not essential)

        """

        # This is a tensor to evaluate the policy for assets post step 3 and verify our results in debugging (validation, not essential)

        self.linspace_k = torch.tensor([20,40,50,60,100,300], device = device, dtype = torch.float32)
        self.inspace_z = self.idio_productivity
        self.linspace_Z = torch.tensor([0,1], device = device, dtype = torch.float32)
        self.linspace_K = torch.tensor([40],device = device, dtype = torch.float32)

        # Generate coordinate grids for all combinations
        self.grids = torch.meshgrid(
            self.linspace_k,
            self.inspace_z,
            self.linspace_Z,
            self.linspace_K,
            indexing="ij"  # Use "ij" indexing for matrix-style (not Cartesian)
        )

        self.nd_tensor = torch.stack(self.grids, dim=-1).reshape(-1, 4)

        """

        This block defines the policy NN and the value NN.

        In both cases, we choose a tanh() activation function for the hidden layers

        This is common with a sigmoid() output layer, which will tell us the fraction of total resources that go to next period's assets (why is a linear output not a good idea in this case?)

        """

        # This is a "trick" to scale the sigmoid function so that it has higher accuracy at output range (0.96-0.99)
        # DO NOT CHANGE: REQUIRED FOR STABLE CONVERGENCE

        class ScaledSigmoid(torch.nn.Module):
            def __init__(self, scale=0.1):
                super(ScaledSigmoid, self).__init__()
                self.scale = scale

            def forward(self, x):
                return torch.sigmoid(self.scale * x)

        ### NN for the policy function ###

        # This is a feed-forward neural network with two hidden layers and tanh activation functions for the assets policy function

        self.policy_assets = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, self.size_hidden_layers),
            torch.nn.Tanh(),
            torch.nn.Linear(self.size_hidden_layers, self.size_hidden_layers),
            torch.nn.Tanh(),
            torch.nn.Linear(self.size_hidden_layers, self.output_dim),
            ScaledSigmoid(scale=0.15)
        )


        ### NN for the value function ###

        # In the case of the value function, note that we use a linear output function, since we are trainig it via regression (step 2)

        self.policy_value = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, self.size_hidden_layers),
            torch.nn.Tanh(),
            torch.nn.Linear(self.size_hidden_layers, self.size_hidden_layers),
            torch.nn.Tanh(),
            torch.nn.Linear(self.size_hidden_layers, self.output_dim),
        )

    """

    These are the methods of the Krusell-Smith Class

    """

    @torch.jit.export  # torchscript compiler pragma
    def compute_invariant_dist(self, transition_matrix):

        """

        Standard function to compute the ergodic distribution of a Markov Chain. No Gradients Required.

        Inputs:
        - Markov Chain of either the idio. or agg. shocks


        """

        with torch.no_grad(): # This means that this function requires no computation of gradients (not optimizing NN parameters in this function)
        # Good practice to wrap the methods that do not require gradients under this context manager, avoids leakage of gradients across methods

            eigvals, eigvecs = torch.linalg.eig(transition_matrix.T)
            eigvals = eigvals.real
            eigvecs = eigvecs.real
            idx = torch.isclose(eigvals, torch.tensor(1.0, device=self.device))
            if not idx.any():
                raise ValueError("No eigenvalue found close to 1, check matrix properties.")
            stationary = eigvecs[:, idx].squeeze()
            stationary = stationary / stationary.sum()
            return stationary

    @torch.jit.export
    def initialize(self):

        """

        This function initializes the idiosyncratic and aggregate shocks for the simulation of the model (step 1 of the algorithm). No Gradients Required.

        Inputs:
        -None (class instance passed by reference)

        """

        with torch.no_grad():

          ergodic_matrix = torch.cumsum(self.stationary_agg, dim=0)
          ergodic_cumsum = torch.cumsum(self.stationary_idio, dim=0)

          for t in range(self.n_periods_simulation):

              if t == 0: # In the first period, draw from the ergodic distribution
                  probs = ergodic_matrix.repeat(self.n_batches_simulation, 1)
                  self.agg_simulation[:, 0] = torch.multinomial(probs, num_samples=1).squeeze(dim=1)  # Sample next states
              else:
                  probs = self.prob_trans_agg[self.agg_simulation[:, t - 1].to(torch.int32)]
                  self.agg_simulation[:, t] = torch.multinomial(probs, num_samples=1).squeeze(1)  # Sample next states
              if t == 0: # Since the Markov Chain for the idio. shocks depends on the realization of the aggregate shock, we need to precompute the aggregate shocks every period
                  combined_states = torch.multinomial(ergodic_cumsum, num_samples=self.n_batches_simulation * self.n_agents, replacement=True)\
                      .reshape(self.n_batches_simulation, self.n_agents)
                  self.inputs_simulation[:, :, t, 1] = combined_states % 2
              else:
                # This is just book-keeping: note that in the KS method the distribution of the idio shocks is not independent but depends on the aggregate state
                  agg_current = self.agg_simulation[:, t - 1].to(torch.int32).reshape([self.n_batches_simulation, 1])
                  agg_future = self.agg_simulation[:, t].to(torch.int32).reshape([self.n_batches_simulation, 1])
                  row_index = 2 * agg_current + self.inputs_simulation[:, :, t - 1, 1]
                  col_index = 2 * agg_future
                  col_index = col_index.repeat(1, self.n_agents)
                  row_index = row_index.to(torch.int32)
                  col_index = col_index.to(torch.int32)
                  probs = torch.cat([self.prob_trans_idio[row_index, col_index].reshape(self.n_batches_simulation * self.n_agents, 1),
                                    self.prob_trans_idio[row_index, col_index + 1].reshape(self.n_batches_simulation * self.n_agents, 1)], dim=1)
                  probs_flat = probs.reshape(-1, 2) / probs.reshape(-1, 2).sum(dim=1).reshape(self.n_batches_simulation * self.n_agents, 1)
                  sampled = torch.multinomial(probs_flat, num_samples=1, replacement=True).reshape(self.n_batches_simulation, self.n_agents)
                  self.inputs_simulation[:, :, t, 1] = sampled

          self.inputs_simulation[:, :, :, 2] = self.agg_simulation.repeat(1,1,self.n_agents).reshape(self.n_batches_simulation, self.n_agents,self.n_periods_simulation)


    @torch.jit.export
    def simulation(self):

        """

        Simulates the economy forward using the current policy network (Step 1). No Gradients Required.

        Inputs: - None

        """

        with torch.no_grad():

            for t in range(self.n_periods_simulation - 1):

            # This is the within-periods KS economy, we have certain agents with certain wealth and productivity and we compute the aggregates and the prices based on those
            # Then, given individual and aggregate states, we evaluate the assets policy NN to obtain next period's states

                agg_labour = self.L[self.inputs_simulation[:, 0, t, 2].to(torch.int32)] * self.l_bar
                agg_shock = self.agg_productivity[self.inputs_simulation[:, 0, t, 2].to(torch.int32)]
                labour_tax = self.labour_taxes[self.inputs_simulation[:, 0, t, 2].to(torch.int32)]
                self.inputs_simulation[:, :, t, 3] = self.inputs_simulation[:, :, t, 0].mean(dim=1).repeat(1,self.n_agents)
                K = self.inputs_simulation[:, :, t, 0].mean(dim=1)
                R = 1 + agg_shock * self.alpha * (K / agg_labour) ** (self.alpha - 1) - self.delta
                W = agg_shock * (1 - self.alpha) * (K / agg_labour) ** self.alpha
                upper_bound = R * self.inputs_simulation[:, :, t, 0] + (1 - labour_tax) * W * self.l_bar * self.inputs_simulation[:, :, t, 1] + self.mu * W * (1 - self.inputs_simulation[:, :, t, 1])
                state = self.inputs_simulation[:, :, t, :].reshape(self.n_batches_simulation*self.n_agents, self.n_features)
                state = 2.0*(state - self.normalization_min) / (self.normalization_max - self.normalization_min) - 1
                next_capital = self.policy_assets(state).view(self.n_batches_simulation, self.n_agents) * upper_bound
                self.inputs_simulation[:, :, t + 1, 0] = next_capital


    @torch.jit.export
    def simulation_value(self):

     """
     This function is similar to the previous one, only that now the path of simulation is the one to construct the lifetime utility given the assets policy NN
     The output from this simulation will allow us to train our Value NN via regression. That is, given some states in period 0, the output from this function tells us the expected lifetime utilty given those initial conditions,
     and then we can use that to tell the NN that given those inputs it should generate that lifetime utility

     Inputs:
     -None

     """
     with torch.no_grad():

        self.util_value[:,:] = 0 # reset the terminal values to 0

        for t in range(self.n_periods_value - 1):

            agg_labour = self.L[self.inputs_value[:, 0, t, 2].to(torch.int32)] * self.l_bar
            agg_shock = self.agg_productivity[self.inputs_value[:, 0, t, 2].to(torch.int32)]
            labour_tax = self.labour_taxes[self.inputs_value[:, 0, t, 2].to(torch.int32)]
            labour_tax = labour_tax.unsqueeze(1)
            self.inputs_value[:, :, t, 3] = self.inputs_value[:, :, t, 0].mean(dim=1).unsqueeze(1).repeat(1,self.n_agents)
            K = self.inputs_value[:, :, t, 0].mean(dim=1)
            R = 1 + agg_shock * self.alpha * (K / agg_labour) ** (self.alpha - 1) - self.delta
            R = R.unsqueeze(1)
            W = agg_shock * (1 - self.alpha) * (K / agg_labour) ** self.alpha
            W = W.unsqueeze(1)
            upper_bound = R * self.inputs_value[:, :, t, 0] + (1 - labour_tax) * W * self.l_bar * self.inputs_value[:, :, t, 1] + self.mu * W * (1 - self.inputs_value[:, :, t, 1])
            state = self.inputs_value[:, :, t, :].reshape(self.n_batches_value*self.n_agents, self.n_features)

            state = 2.0*(state - self.normalization_min) / (self.normalization_max - self.normalization_min) - 1
            next_capital = self.policy_assets(state).view(self.n_batches_value, self.n_agents) * upper_bound
            self.inputs_value[:, :, t + 1, 0] = next_capital
            consumption = upper_bound - self.inputs_value[:, :, t + 1, 0]
            self.util_value[:,:] += self.beta**t * torch.log(consumption)

        # Normalize the target values for more stable training

        self.value_min = self.util_value[:].min()
        self.value_max = self.util_value[:].max()

        self.util_value[:] = (self.util_value[:] - self.value_min) / (self.value_max - self.value_min)



    @torch.jit.export
    def forward_value(self):

        """

        Calculates the loss for training the value NN (Step 2 regression).
        This is the regression setup for training the value NN. As outlined above, given say (a0, z0, Z0, K0) with expected lifetime utilty V_end(a0, z0, Z0, K0), here we want to train the value NN so that NN_Value(a0, z0, Z0, K0) returns V_end(a0, z0, Z0, K0) after training
        Gradients Required.

        Inputs: - None

        """

        loss_regression = torch.zeros(self.n_batches_value,self.n_agents, device = self.device, dtype = torch.float32)

        state = self.inputs_value[:,:,0,:].reshape(self.n_batches_value*self.n_agents, self.n_features)

        state = 2.0*(state - self.normalization_min) / (self.normalization_max - self.normalization_min) - 1

        loss_regression = (self.policy_value(state).squeeze(1) - self.util_value.reshape(self.n_batches_value*self.n_agents))**2

        return loss_regression.mean()


    @torch.jit.export
    def drawShocksFeedForward(self):

        """
        Draws shocks for Monte Carlo simulation (Step 3). No Gradients.

        Inputs: - None

        """

        with torch.no_grad():

          for t in range(self.n_periods_monte_carlo - 1):

              probs = self.prob_trans_agg[self.agg_monte_carlo[:, t].to(torch.int32)]
              self.agg_monte_carlo[:, t + 1] = torch.multinomial(probs, num_samples=1).squeeze(1)
              self.inputs_monte_carlo[:, :, t + 1, 2] = self.agg_monte_carlo[:, t + 1].reshape(self.n_batches_monte_carlo, 1).repeat(1, self.n_agents)
              agg_current = self.agg_monte_carlo[:, t].to(torch.int32).reshape([self.n_batches_monte_carlo, 1])
              agg_future = self.agg_monte_carlo[:, t + 1].to(torch.int32).reshape([self.n_batches_monte_carlo, 1])
              row_index = 2 * agg_current + self.inputs_monte_carlo[:, :, t, 1]
              col_index = 2 * agg_future
              col_index = col_index.repeat(1, self.n_agents)
              row_index = row_index.to(torch.int32)
              col_index = col_index.to(torch.int32)
              probs = torch.cat([self.prob_trans_idio[row_index, col_index].reshape(self.n_batches_monte_carlo * self.n_agents, 1),
                                self.prob_trans_idio[row_index, col_index + 1].reshape(self.n_batches_monte_carlo * self.n_agents, 1)], dim=1)
              probs_flat = probs.reshape(-1, 2) / (probs.reshape(-1, 2).sum(dim=1)).reshape(self.n_batches_monte_carlo * self.n_agents, 1)
              sampled = torch.multinomial(probs_flat, num_samples=1, replacement=True).reshape(self.n_batches_monte_carlo, self.n_agents)
              self.inputs_monte_carlo[:, :, t + 1, 1] = sampled

    @torch.jit.export
    def drawShocksValue(self):

        """

        Draws shocks for value function target simulation (Step 2). No Gradients.

        Inputs: - None

        """

        with torch.no_grad():

            for t in range(self.n_periods_value - 1):

                probs = self.prob_trans_agg[self.agg_value[:, t].to(torch.int32)]
                self.agg_value[:, t + 1] = torch.multinomial(probs, num_samples=1).squeeze(1)
                self.inputs_value[:, :, t + 1, 2] = self.agg_value[:, t + 1].reshape(self.n_batches_value, 1).repeat(1, self.n_agents)
                agg_current = self.agg_value[:, t].to(torch.int32).reshape([self.n_batches_value, 1])
                agg_future = self.agg_value[:, t + 1].to(torch.int32).reshape([self.n_batches_value, 1])
                row_index = 2 * agg_current + self.inputs_value[:, :, t, 1]
                col_index = 2 * agg_future
                col_index = col_index.repeat(1, self.n_agents)
                row_index = row_index.to(torch.int32)
                col_index = col_index.to(torch.int32)
                probs = torch.cat([self.prob_trans_idio[row_index, col_index].reshape(self.n_batches_value * self.n_agents, 1),
                                    self.prob_trans_idio[row_index, col_index + 1].reshape(self.n_batches_value * self.n_agents, 1)], dim=1)
                probs_flat = probs.reshape(-1, 2) / (probs.reshape(-1, 2).sum(dim=1)).reshape(self.n_batches_value * self.n_agents, 1)
                sampled = torch.multinomial(probs_flat, num_samples=1, replacement=True).reshape(self.n_batches_value, self.n_agents)
                self.inputs_value[:, :, t + 1, 1] = sampled


    @torch.jit.export
    def forward(self):

        """

        Performs the forward pass for training the assets policy NN (Step 3).
        Builds the computational graph. Gradients Required.

        Gradients are computed.

        Inputs: - None

        """

        utility = torch.zeros(self.n_batches_monte_carlo,self.n_agents, device=self.device, dtype=torch.float32)

        current_capital = self.inputs_monte_carlo[:, :, 0, 0].detach().clone()

        for t in range(self.n_periods_monte_carlo):

            target_capital = current_capital[:, 0].unsqueeze(1).clone() # Since we are solving for the Competitive Equilibrium, we only optimize w.r.t the agent 0's capital and detach the rest from the computational graph
            other_capital = current_capital[:, 1:].detach().clone() # Rest of the agents' capital is detached
            current_capital = torch.cat([target_capital, other_capital], dim=1)

            current_idio = self.inputs_monte_carlo[:, :, t, 1].detach().clone()
            current_agg = self.inputs_monte_carlo[:, :, t, 2].detach().clone()
            K = current_capital.mean(dim=1).detach()

            state = torch.stack([current_capital, current_idio, current_agg, K.unsqueeze(1).repeat(1, self.n_agents)], dim=2)
            state = state.reshape(self.n_batches_monte_carlo*self.n_agents, self.n_features)
            state = 2.0*(state - self.normalization_min) / (self.normalization_max - self.normalization_min) - 1

            # Employ our auxiliary value function NN in the last period to capture the value for the remaining periods

            if t == self.n_periods_monte_carlo-1:

                # Note here that we are training the normalized value function, therefore we need to unnormalize it to use it on the computational graph

                value = self.policy_value(state).reshape(self.n_batches_monte_carlo,self.n_agents) * (self.value_max - self.value_min) + self.value_min

                utility = utility + self.beta**t * value

                continue

            next_capital = self.policy_assets(state).view(self.n_batches_monte_carlo, self.n_agents) # note that this is the fraction of total resources that go to next period's capital, not the actual capital

            agg_shock_idx = current_agg[:, 0].to(torch.int32)
            agg_labour = self.L[agg_shock_idx] * self.l_bar
            agg_labour = agg_labour.flatten()
            agg_shock_val = self.agg_productivity[agg_shock_idx]
            labour_tax = self.labour_taxes[agg_shock_idx]
            labour_tax = labour_tax.unsqueeze(1)
            R = 1.0 + agg_shock_val * self.alpha * (K / agg_labour) ** (self.alpha - 1) - self.delta
            R = R.unsqueeze(1)
            W = agg_shock_val * (1 - self.alpha) * (K / agg_labour) ** self.alpha
            W = W.unsqueeze(1)

            upper_bound = R * current_capital + (1 - labour_tax) * W * self.l_bar * current_idio + self.mu * W * (1 - current_idio)

            current_capital = next_capital * upper_bound
            consumption = upper_bound - current_capital

            utility = utility + ((self.beta ** t) * torch.log(consumption))

        # This is the objective function w.r.t to which the NN will be trained, we want to maximize the expected lifetime utility of agent 0
        return utility.mean(dim=0)[0] # Since we are solving the competitive equilibrium, we only care about a single agent's utility (in this case agent 0 but it could any)


    @torch.jit.export
    def sampler(self):

        """

        Samples starting states from the long simulation (Step 1) for Step 3 training. No gradients.

        Inputs: - None

        """

        with torch.no_grad():

            n_periods = self.n_periods_simulation
            n_batches_mc = self.n_batches_monte_carlo
            burn_in = self.burn_in
            starting_periods = torch.randperm(n_periods - burn_in)[:n_batches_mc] + burn_in
            initial_conditions = self.inputs_simulation[0, :, starting_periods, :] # 0th index is 0 given we have a single batch economy for the simulation in Step 1
            initial_conditions = initial_conditions.permute(1, 0, 2) # need to permute given the non-monotonous slicing of the previous tensor
            self.inputs_monte_carlo[:, :, 0, :] = initial_conditions
            self.agg_monte_carlo[:, 0] = self.inputs_monte_carlo[:, 0, 0, 2]

    @torch.jit.export
    def sampler_value(self):

        """

        Samples starting states from the long simulation (Step 1) for Step 2 value target generation. No gradients.

        Inputs: - None

        """

        with torch.no_grad():

            n_periods = self.n_periods_simulation
            n_batches_mc = self.n_batches_value
            burn_in = self.burn_in
            starting_periods = torch.randperm(n_periods - burn_in)[:n_batches_mc] + burn_in
            initial_conditions = self.inputs_simulation[0, :, starting_periods, :]
            initial_conditions = initial_conditions.permute(1, 0, 2)
            self.inputs_value[:, :, 0, :] = initial_conditions
            self.agg_value[:, 0] = self.inputs_value[:, 0, 0, 2]

    # No torch.jit.export decorator here as it employs non-supported operations. Fine in terms of performance, done only once after training.
    def bellman_error_sampled(self, n_periods: int = 200, clamp_min: float = 1e-8):
        
        """
    
        CRITICAL NOTE I: This is method is only to be called after TRAINING!! 

        CRITICAL NOTE II: This method computes the Bellman Error on the sampled states! It "assumes" that the "true" ergodic set has been attained!!

        Computes the Bellman error as given by V(s) - [u(c) + beta * E[V(s') | s]] under the trained (optimal) policy and value functions.

        - Draw `n_periods` random periods after burn_in from self.inputs_simulation.
        - Compute a'_i for all agents, get K' as mean(a'_pop) per period. (K' is pre-determined by the policy function)
        - Evaluate the expectation E[V(s') | s] across the 4 (z', Z') combinations. 
        
        Inputs:

            - None
        
        Returns:
            residuals: Tensor shape (n_periods, n_agents)
            stats: Tensor([mean_abs, rmse, max_abs])
            states: Tensor shape (n_periods, n_agents, n_features) (to be used for plotting obtained BE as a function of the individual sates)

        """

        device = self.device
        with torch.no_grad():
            # sample period indices after burn_in
            available = self.n_periods_simulation - self.burn_in
            S = min(n_periods, available) # Number of sampled periods 
            perm = torch.randperm(available, device=device)[:S] + self.burn_in  # shape (S,)
            # states from simulation: inputs_simulation shape [n_batches_simulation, n_agents, T0, features]
            states = self.inputs_simulation[0, :, perm, :]        # shape (n_agents, S, 4)
            states = states.permute(1, 0, 2).contiguous()        # shape (M, n_agents, 4)

            S, A, F = states.shape
            assert A == self.n_agents and F == self.n_features

            # unpack current states
            a0 = states[:, :, 0]                # (S, n_agents)
            z0 = states[:, :, 1].to(torch.int32)  # (S, n_agents) {0,1}
            Z0 = states[:, :, 2].to(torch.int32)  # (S, n_agents) (same across agents per period but fine)
            K0_period = states[:, 0, 3]         # (S,) pick first agent's K (same for all agents)

            # compute period-level aggregates (for all drawn samples S)
            agg_shock_val = self.agg_productivity[Z0[:, 0]]   # (S,)
            agg_labour = self.L[Z0[:, 0]] * self.l_bar        # (S,)
            labour_tax = self.labour_taxes[Z0[:, 0]]          # (S,)

            R = 1.0 + agg_shock_val * self.alpha * (K0_period / agg_labour) ** (self.alpha - 1.0) - self.delta  # (S,)
            W = agg_shock_val * (1.0 - self.alpha) * (K0_period / agg_labour) ** (self.alpha)                  # (S,)

            # expand to per-agent shape
            R_exp = R.unsqueeze(1).repeat(1, A)           # (S, A)
            W_exp = W.unsqueeze(1).repeat(1, A)           # (S, A)
            labour_tax_exp = labour_tax.unsqueeze(1).repeat(1, A)  # (S, A)

            # compute upper bound for each agent at t (vectorized)
            z0_float = z0.to(torch.float32)
            upper_bound = R_exp * a0 + (1.0 - labour_tax_exp) * W_exp * self.l_bar * z0_float + self.mu * W_exp * (1.0 - z0_float)  # (S,A)

            # normalize states to feed into NNs
            norm_min = self.normalization_min.view(1, 1, -1)   # (1,1,4)
            norm_max = self.normalization_max.view(1, 1, -1)   # (1,1,4)
            states_norm = 2.0 * (states - norm_min) / (norm_max - norm_min) - 1.0  # (S,A,4)

            # flatten to evaluate networks in batch
            states_norm_flat = states_norm.view(S * A, F)      # (S*A, 4)

            # policy_assets -> fraction -> next capital
            frac_flat = self.policy_assets(states_norm_flat).view(S, A)  # (S,A)
            a1 = frac_flat * upper_bound                               # (S,A)
            Kprime = a1.mean(dim=1)                                    # (S,)

            # consumption today
            consumption0 = (upper_bound - a1).clamp(min=clamp_min)     # (S,A)

            # compute V(s) at t (unnormalize)
            v0_norm = self.policy_value(states_norm_flat).view(S, A)   # (S,A)
            v0 = v0_norm * (self.value_max - self.value_min) + self.value_min  

            # Prepare to compute expected V(s') across the 4 (Z', z') combinations
            # row_index depends on current (Z0, z0): row = 2*Z + z
            row_index = (2 * Z0 + z0).to(torch.int32)  # (S,A)

            expected_v1 = torch.zeros_like(v0, device=device, dtype = torch.float32)  # (S,A)

            # iterate over 2x2 combos (only 4 iterations)
            for Zp in range(0, self.agg_productivity.numel()):  # Zp is the next period's aggregate state (0 or 1)
                for zp in range(0, self.idio_productivity.numel()):
                    col_idx = 2 * Zp + zp
                    # prob for each (M,A)
                    probs = self.prob_trans_idio[row_index, col_idx].to(device)  # (M,A)

                    # build s' = (a1, z', Z', K') for all samples
                    # a1 is (M,A), zp scalar, Zp scalar, Kprime is (M,) Note: zp and Zp are scalars as we are computing the expectation at each (S, A) pair
                    zp_tensor = torch.full((S, A), float(zp), device=device, dtype=torch.float32)
                    Zp_tensor = torch.full((S, A), float(Zp), device=device, dtype=torch.float32)
                    Kp_tensor = Kprime.unsqueeze(1).repeat(1, A)

                    sp = torch.stack([a1, zp_tensor, Zp_tensor, Kp_tensor], dim=2)  # (S,A,4)
                    sp_norm = 2.0 * (sp - norm_min) / (norm_max - norm_min) - 1.0   # (S,A,4)
                    sp_flat = sp_norm.view(S * A, F)

                    v1_norm = self.policy_value(sp_flat).view(S, A)                 # (S,A)
                    v1 = v1_norm * (self.value_max - self.value_min) + self.value_min  # (S,A)

                    expected_v1 += probs * v1

            # Bellman RHS and residuals
            rhs = torch.log(consumption0) + self.beta * expected_v1   # (S,A)
            residuals = v0 - rhs                                      # (S,A)

            # summary stats over all samples
            mean_abs = residuals.abs().mean()
            rmse = torch.sqrt((residuals ** 2).mean())
            max_abs = residuals.abs().max()
            stats = torch.stack([mean_abs, rmse, max_abs])

            return residuals, stats, states

    def bellman_error_by_bins(self, residuals, states, n_bins, feature_idx):
        """
        residuals: (S, A) from bellman_error_sampled
        states:    (S, A, 4) sampled states used for BE (same periods & agents)
        feature_idx: 0=a, 3=K

        Returns: list of (bin_edges, mean_abs_BE_per_bin)

        Inputs: 

        - Obtained residuals 
        - Employed states for the residuals

        """
        with torch.no_grad():
            x = states[:, :, feature_idx].reshape(-1)   # Flattened feature
            r_abs = residuals.reshape(-1).abs()         # Flattened absolute residuals
            r_signed = residuals.reshape(-1)            # Flattened signed residuals

            # Compute quantile-based bin edges
            quantiles = torch.linspace(0, 1, n_bins + 1, device=x.device)
            bins = torch.quantile(x, quantiles)

            abs_means = []
            signed_means = []
            for i in range(n_bins):
                mask = (x >= bins[i]) & (x <= bins[i+1] if i == n_bins-1 else x < bins[i+1])
                if mask.any():
                    abs_means.append(r_abs[mask].mean().item())
                    signed_means.append(r_signed[mask].mean().item())
                else:
                    abs_means.append(float("nan"))
                    signed_means.append(float("nan"))

            return bins.cpu().numpy(), abs_means, signed_means


  
    @torch.jit.export
    def sampler_irf(self):
        """
        Samples starting states from the long simulation for IRF generation.

        Inputs: - None

        """
        with torch.no_grad():
            n_periods = self.n_periods_simulation
            n_batches = self.n_batches_irf
            burn_in = self.burn_in
            
            # Ensure there are enough periods to sample from
            if n_periods - burn_in < n_batches:
                raise ValueError("Not enough simulation periods after burn-in to sample for IRF.")

            starting_periods = torch.randperm(n_periods - burn_in, device=self.device)[:n_batches] + burn_in
            initial_conditions = self.inputs_simulation[0, :, starting_periods, :]
            initial_conditions = initial_conditions.permute(1, 0, 2)
            
            self.inputs_irf[:, :, 0, :] = initial_conditions
            self.agg_shocks_irf[:, 0] = self.inputs_irf[:, 0, 0, 2]

    @torch.jit.export
    def drawShocksIRF(self):
        """
        Draws aggregate and idiosyncratic shock paths for the IRF simulation horizon.

        Inputs: - None
        """
        with torch.no_grad():
            for t in range(self.n_periods_irf - 1):
                # Draw next aggregate shock
                probs_agg = self.prob_trans_agg[self.agg_shocks_irf[:, t].to(torch.int32)]
                self.agg_shocks_irf[:, t + 1] = torch.multinomial(probs_agg, num_samples=1).squeeze(1)
                self.inputs_irf[:, :, t + 1, 2] = self.agg_shocks_irf[:, t + 1].unsqueeze(1).repeat(1, self.n_agents)

                # Draw next idiosyncratic shocks conditional on aggregate path
                agg_current = self.agg_shocks_irf[:, t].to(torch.int32).unsqueeze(1)
                agg_future = self.agg_shocks_irf[:, t + 1].to(torch.int32).unsqueeze(1)
                
                row_index = 2 * agg_current + self.inputs_irf[:, :, t, 1].to(torch.int32)
                col_index = 2 * agg_future
                
                col_index = col_index.repeat(1, self.n_agents)
                
                probs_idio_slice1 = self.prob_trans_idio[row_index, col_index]
                probs_idio_slice2 = self.prob_trans_idio[row_index, col_index + 1]

                probs = torch.cat([
                    probs_idio_slice1.reshape(self.n_batches_irf * self.n_agents, 1),
                    probs_idio_slice2.reshape(self.n_batches_irf * self.n_agents, 1)
                ], dim=1)
                
                probs_sum = probs.sum(dim=1, keepdim=True)
                # Avoid division by zero if a state is absorbing for a period
                probs_flat = torch.where(probs_sum > 0, probs / probs_sum, torch.full_like(probs, 0.5))

                sampled = torch.multinomial(probs_flat, num_samples=1, replacement=True).reshape(self.n_batches_irf, self.n_agents)
                self.inputs_irf[:, :, t + 1, 1] = sampled.to(torch.float32)

    @torch.jit.export
    def _run_irf_path(self, inputs_irf, agg_shocks_irf):
        """
        Helper function to run a simulation path for IRFs and return aggregates.

        Inputs: 
        - inputs_irf: Tensor of shape [n_batches_irf, n_agents, n_periods_irf, n_features]
        - agg_shocks_irf: Tensor of shape [n_batches_irf, n_period

        """
        # Tensors to store results for this specific path
        agg_capital = torch.zeros_like(self.agg_capital_irf)
        agg_output = torch.zeros_like(self.agg_output_irf)
        agg_interest_rate = torch.zeros_like(self.agg_interest_rate_irf)

        with torch.no_grad():
            for t in range(self.n_periods_irf - 1):
                # 1. Calculate prices and aggregates for the current period t
                agg_shock_idx = agg_shocks_irf[:, t].to(torch.int32)
                agg_labour = self.L[agg_shock_idx] * self.l_bar
                agg_shock_val = self.agg_productivity[agg_shock_idx]
                labour_tax = self.labour_taxes[agg_shock_idx]

                K = inputs_irf[:, :, t, 0].mean(dim=1)
                inputs_irf[:, :, t, 3] = K.unsqueeze(1).repeat(1, self.n_agents)
                
                # Store aggregate capital
                agg_capital[:, t] = K

                # Calculate R and W
                R = 1 + agg_shock_val * self.alpha * (K / agg_labour) ** (self.alpha - 1) - self.delta
                W = agg_shock_val * (1 - self.alpha) * (K / agg_labour) ** self.alpha

                # Store prices
                agg_interest_rate[:, t] = R
                
                # Store aggregate output
                agg_output[:, t] = agg_shock_val * K**self.alpha * agg_labour**(1 - self.alpha)

                # 2. Determine next period's capital using the policy function
                state = inputs_irf[:, :, t, :].reshape(self.n_batches_irf * self.n_agents, self.n_features)
                state_norm = 2.0 * (state - self.normalization_min) / (self.normalization_max - self.normalization_min) - 1.0
                
                upper_bound = R.unsqueeze(1) * inputs_irf[:, :, t, 0] + \
                              (1 - labour_tax.unsqueeze(1)) * W.unsqueeze(1) * self.l_bar * inputs_irf[:, :, t, 1] + \
                              self.mu * W.unsqueeze(1) * (1 - inputs_irf[:, :, t, 1])

                next_capital_policy = self.policy_assets(state_norm).view(self.n_batches_irf, self.n_agents)
                next_capital = next_capital_policy * upper_bound
                
                inputs_irf[:, :, t + 1, 0] = next_capital

                # 3. Calculate and store other aggregates
                consumption = upper_bound - next_capital
                
                # Investment = K' - (1-delta)*K
                investment = next_capital.mean(dim=1) - (1-self.delta)*K

        return agg_capital, agg_output, agg_interest_rate




    def simulationIRF(self):
        """
        Generates and saves IRFs for a negative productivity shock.
        1. Simulates a benchmark path without shock.
        2. Simulates a shocked path with a negative shock at t=0.
        3. Saves both paths to CSV files.
        """
        print("Starting IRF generation...")
        with torch.no_grad():
            # --- 1. BENCHMARK PATH ---
            print("Simulating benchmark IRF path...")
            (agg_capital_bench, agg_output_bench,
              agg_interest_rate_bench) = self._run_irf_path(
                self.inputs_irf.clone(), self.agg_shocks_irf.clone()
            )

            # Save benchmark results
            np.savetxt("irf/agg_capital_benchmark.csv", agg_capital_bench.cpu().numpy(), delimiter=",")
            np.savetxt("irf/agg_output_benchmark.csv", agg_output_bench.cpu().numpy(), delimiter=",")
            np.savetxt("irf/agg_interest_rate_benchmark.csv", agg_interest_rate_bench.cpu().numpy(), delimiter=",")
            np.savetxt("irf/agg_shocks_benchmark.csv", self.agg_shocks_irf.cpu().numpy(), delimiter=",")

            # --- 2. SHOCKED PATH ---
            print("Simulating shocked IRF path...")
            inputs_irf_shocked = self.inputs_irf.clone()
            agg_shocks_irf_shocked = self.agg_shocks_irf.clone()

            # Introduce the shock at t=0: force the economy into the bad state (Z=0.99)
            agg_shocks_irf_shocked[:, 0] = self.agg_productivity[0]
            inputs_irf_shocked[:, :, 0, 2] = self.agg_productivity[0]
            
            # Re-draw shocks from t=1 onwards, conditional on the new t=0 state
            for t in range(self.n_periods_irf - 1):
                probs_agg = self.prob_trans_agg[agg_shocks_irf_shocked[:, t].to(torch.int32)]
                agg_shocks_irf_shocked[:, t + 1] = torch.multinomial(probs_agg, num_samples=1).squeeze(1)
                inputs_irf_shocked[:, :, t + 1, 2] = agg_shocks_irf_shocked[:, t + 1].unsqueeze(1).repeat(1, self.n_agents)

                agg_current = agg_shocks_irf_shocked[:, t].to(torch.int32).unsqueeze(1)
                agg_future = agg_shocks_irf_shocked[:, t + 1].to(torch.int32).unsqueeze(1)
                row_index = 2 * agg_current + inputs_irf_shocked[:, :, t, 1].to(torch.int32)
                col_index = 2 * agg_future.repeat(1, self.n_agents)
                
                probs_idio_slice1 = self.prob_trans_idio[row_index, col_index]
                probs_idio_slice2 = self.prob_trans_idio[row_index, col_index + 1]
                probs = torch.cat([
                    probs_idio_slice1.reshape(self.n_batches_irf * self.n_agents, 1),
                    probs_idio_slice2.reshape(self.n_batches_irf * self.n_agents, 1)
                ], dim=1)
                probs_sum = probs.sum(dim=1, keepdim=True)
                probs_flat = torch.where(probs_sum > 0, probs / probs_sum, torch.full_like(probs, 0.5))

                sampled = torch.multinomial(probs_flat, num_samples=1, replacement=True).reshape(self.n_batches_irf, self.n_agents)
                inputs_irf_shocked[:, :, t + 1, 1] = sampled.to(torch.float32)
            
            # Run the simulation with the new shocked paths
            (agg_capital_shock, agg_output_shock,
               agg_interest_rate_shock) = self._run_irf_path(
                inputs_irf_shocked, agg_shocks_irf_shocked
            )
            
            # Save shocked results
            np.savetxt("irf/agg_capital_shocked.csv", agg_capital_shock.cpu().numpy(), delimiter=",")
            np.savetxt("irf/agg_output_shocked.csv", agg_output_shock.cpu().numpy(), delimiter=",")
            np.savetxt("irf/agg_interest_rate_shocked.csv", agg_interest_rate_shock.cpu().numpy(), delimiter=",")
            np.savetxt("irf/agg_shocks_shocked.csv", agg_shocks_irf_shocked.cpu().numpy(), delimiter=",")

            print("IRF generation complete. Results saved in 'irf/' directory.")

def plot_irfs():
    """
    Loads IRF data from CSV files, calculates the percentage deviation
    for each path, averages them, computes confidence intervals, and plots the results.
    """
    variables_to_plot = {
        'capital': 'Capital (K)',
        'output': 'Output (Y)',
        'interest_rate': 'Interest Rate (R)'
    }

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 12))
    axes = axes.flatten()
    fig.suptitle('Impulse Response to a Negative Productivity Shock', fontsize=16)

    for i, (var, title) in enumerate(variables_to_plot.items()):
        ax = axes[i]
        try:
            # Load benchmark and shocked data (shape: [n_batches, n_periods])
            benchmark_data = np.loadtxt(f"irf/agg_{var}_benchmark.csv", delimiter=',')
            shocked_data = np.loadtxt(f"irf/agg_{var}_shocked.csv", delimiter=',')

            # Calculate percentage deviation for EACH path
            # Using np.divide to handle potential division by zero safely
            all_irfs = np.divide(
                shocked_data - benchmark_data,
                benchmark_data,
                out=np.zeros_like(benchmark_data), # Output array for the result
                where=benchmark_data!=0 # Condition to perform the division
            ) * 100

            # Compute the mean across paths
            mean_irf = np.mean(all_irfs, axis=0)

            # Plotting
            periods = range(len(mean_irf))
            ax.plot(periods, mean_irf, marker='o', linestyle='-', label='Mean Response', markersize=4)
            
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel('Periods after Shock')
            ax.set_ylabel('% Deviation from Benchmark')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()

        except FileNotFoundError:
            print(f"Warning: Could not find IRF data for '{var}'. Skipping plot.")
            ax.text(0.5, 0.5, f"Data not found for\n{title}", ha='center', va='center', fontsize=12)
            ax.set_title(title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():

    """

    This is the main function, it:

        - Creates an instance of the Krusell Smith model class
        - Sets up the optimizers for the value and policy NNs
        - Solves the model following the DeepHAM approach
            - During the execution, it will:
                - Display a histogram with the current distribution over individual assets
                - In the terminal, you will see the policy and value functions evaluated at some validation points.
                    - These should be as close as possible as in the standard Krusell Smith solution method by Den Haan (2010) (Maliar's Implementation)
                - It saves the current distribution over individual assets every 50 iterations for convenience
            - After execution:
                - Save the results (distribution of individual capital) in "results.txt" file
                - Saves the NN weights and biases in "NN_weights_biases.pth"
        - After training the model, once the weights are stored, it will load the weights and compute the Bellman Errors on sampled states as well as the IRF to a negative aggregate productivity shock.

    """


    # Paths for saved model & results
    weights_path = "NN_weights_biases.pth" # CRITICAL NOTE: ONLY THE WEIGHTS AND BIASES ARE SAVED, NOT THE ENTIRE MODEL! IMPLICATIONS: STATE OF THE OPTIMIZER, NORMALIZATION BOUNDS NOT SAVED! DO NOT CHANGE FROM TRAINING VALUES!
    # Boolean continue training from saved weights
    continue_training = True
    # Decide on outer DeepHAM iterations (Obtainig the invariant distribution (step 1), training the value NN (step 2), training the assets policy NN (step 3))
    num_outer_steps = 51 # Number of outer steps (DeepHAM iterations)

    # Initialize the model and optimizers
    model = KrusellSmith().to(device) # Create instance of the model and pass parameters (scalars and NNs) to GPU
    optim_capital = torch.optim.Adam(model.policy_assets.parameters(), lr = 0.001, betas = (0.8,0.99)) # Optimizer settings for the assets policy NN (momentum, std persistence)
    optim_value = torch.optim.Adam(model.policy_value.parameters(), lr = 0.0001, betas = (0.9,0.99)) # Optimizer settings for the value policy NN



    if os.path.exists(weights_path) and not continue_training:
        print(f"Found saved weights at {weights_path}, skipping training...")

        # Load weights into the model
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model loaded, straight to Bellman Error validation and IRF computation.")

        # Setting up required model parts for Bellman Error validation        
        model.initialize()
        model.simulation()

        model.sampler_value() # Choose random periods for simulation to start value training
        model.drawShocksValue() # Draw shocks for the chosen random start points
        model.simulation_value() # Simulate forward from chosen periods and compute the realized utility

        # Note: here the model is run in eager mode, as we are not training it anymore, but rather validating the Bellman Error

        # --- IRF Analysis ---
        model.sampler_irf()
        model.drawShocksIRF()
        model.simulationIRF()
        
        # --- Plot the generated IRFs ---
        plot_irfs()

        # --- Validation ---- Evaluating the Bellman Errors on sampled states 
        print("Evaluating Bellman Errors on Sampled States...")
        residuals, stats, states = model.bellman_error_sampled(n_periods=200, clamp_min=1e-8)
        print(f"Bellman Error Stats: Mean Abs: {stats[0].item():.6f}, RMSE: {stats[1].item():.6f}, Max Abs: {stats[2].item():.6f}")

        # Running Diagnostics: Plotting Bellman Error by bins

        # Plot Bellman Error by bins for the first feature (e.g., assets)
        feature_idx = 0  # Index of the feature to plot (0 for assets, 3 for capital)  
        n_bins = 25  # Number of bins for the histogram
        bins, means, signed_means = model.bellman_error_by_bins(residuals, states, n_bins, feature_idx)
        plt.figure(figsize=(10, 5))
        plt.bar(bins[:-1], means, width=np.diff(bins), align='edge', edgecolor='black', alpha=0.7)
        plt.xlabel('Individual Asset Bins')
        plt.ylabel('Mean Absolute Bellman Error')
        plt.title('Bellman Error by AAsset Bins')
        plt.xticks(bins, rotation=45)
        plt.tight_layout()
        plt.show()

    else: 

        if os.path.exists(weights_path) and continue_training:
            print(f"Found saved weights at {weights_path}, resuming training with given weights...")

            # Load weights into the model
            model.load_state_dict(torch.load(weights_path, map_location=device))

        try: # Compile model using torchscript (requires the code to be torchscript compatible)
            model = torch.jit.script(model) # Successful compilation will result in faster execution (>100%), as we avoid expensive interpreter overhead (we obtain a static computation graph and CUDA kernel launches are faster)
            print("Model successfully compiled with TorchScript.")
        except Exception as e:
            print(f"Warning: TorchScript compilation failed: {e}. Running in eager mode.")

        # Boolean: ignore verbose printing of timings for Forward method and Monte-Carlo re-draws in Step 3  (for debugging purposes)

        ignore_timings_step_2 = True

        # Boolean: for ignoring the printing of the validation points (for debugging purposes)

        ignore_validation = True

        # Enable interactive mode for plotting the histogram showing the distribution of individual assets
        plt.ion()
        fig, ax = plt.subplots()

        num_epochs_step_2 = 3000 # Number of epochs for training the value NN (step 2)
        num_epochs_step_3 = 150 # Number of epochs for training the assets policy NN (step 3)

        ### Outer Steps of the DeepHAM Algorithm ###

        for step in range(num_outer_steps):

            ### Step 1 - Obtain the invariant distribution under the assets policy NN ###

            # Note that the scope of the functions is within the krusell smith class, therefore, the functions must be called within the model instance, (model.function())

            print(f"Outer DeepHam iteration {step}," "Step 1: Simulating Economy - Obtain Ergodic Distribution")

            model.initialize() # Draw shocks (idio and agg)
            model.simulation() # Given drawn shocks, simulate economy forward to recover individual and aggregate capitals

            ### Step 2 - Given the policy for assets NN, train the value NN ###

            print(f"Outer DeepHam iteration {step},""Step 2: Train the Value Function NN")

            model.sampler_value() # Choose random periods for simulation to start value training
            model.drawShocksValue() # Draw shocks for the chosen random start points
            model.simulation_value() # Simulate forward from chosen periods and compute the realized utility

            time_value = time.time()
            for epoch in range(0,num_epochs_step_2):

                error = model.forward_value() # Train value NN so that it generates the utility obtained in the last function for those given inputs (note: the input is the one for period 0)
                optim_value.zero_grad() # Set gradients to zero
                error.backward() # Compute gradients
                optim_value.step() # Apply a step of the optimizer given gradients
                if epoch % 1000 == 0:
                    print(f"Epoch {epoch}: loss {error}") # This is a regression problem, should be as close to 0 as possible

            print(f"final loss {error}")
            time_value_end = time.time()

            print(f"{time_value_end - time_value}")

            ### Step 3 - Given both current guesses on assets and value NNs, train the assets policy NN ###
            print(f"Outer DeepHam iteration {step},""Step 3: Train the Assets Policy NN")

            model.sampler() # Choose random starting periods from simulation steps
            model.drawShocksFeedForward() # Simulate idio and agg shocks forward from starting points for Monte Carlo paths

            with tqdm(range(num_epochs_step_3), desc=f"Step {step}", unit="iter") as pbar:
                for inner in pbar:
                    time_forward = time.time()
                    utility = model.forward()  # Simulate forward the Krusell Smith economy computational graph

                    loss = -utility  # We are maximizing, take negative of utility
                    optim_capital.zero_grad()  # Reset gradients to zero
                    loss.backward()  # Compute gradients over the batch of economies
                    optim_capital.step()  # Update the NN given the realized gradients

                    time_forward_end = time.time()
                    time_simul = time.time()
                    model.sampler()  # For next step, draw new batches of economies for the next Monte Carlo step
                    model.drawShocksFeedForward()  # Same as above
                    time_simul_end = time.time()

                    # optional timing/log prints (keeps same behavior as your original code)
                    if not ignore_timings_step_2:
                        # use tqdm.write so prints do not break the progress bar
                        tqdm.write(f"time required by: forward {time_forward_end - time_forward:.3f}s simul: {time_simul_end - time_simul:.3f}s")
                        tqdm.write(f"Step {step}, inner {inner}, Loss {-loss.item():.6f}")

            # show the latest loss in the progress bar postfix (human-friendly)
            pbar.set_postfix({"loss": f"{-loss.item():.6f}"})

            print(f"Average Agg Capital: {model.inputs_simulation[:, :, model.burn_in:, 3].mean()}")

            # Convert simulation data to NumPy
            data = model.inputs_simulation[:, :, model.burn_in:, 0].flatten().cpu().detach().numpy() # For validation purposes, take the simulation data to CPU and plot it, the distribution over individual assets should be the same as obtained via grid-based methods

            if step % 50 == 0:
                np.savetxt("results.txt", data, delimiter=",")

            # Update histogram
            fig, ax = plt.subplots()
            ax.hist(data, bins=50, density=True, alpha=0.6, color='b')
            ax.set_xlabel('Individual Capital (a)')
            ax.set_ylabel('Density')
            ax.set_title(f'Ergodic Distribution of Assets (Step {step})')

            # Redraw plot properly
            plt.draw()
            plt.pause(0.01)
            fig.canvas.flush_events()  # Ensure GUI updates


            if not ignore_validation:
                # Check: evaluate policy function on the created ND tensor (should match the ones from the standard grid-based method)
                print("Policy Function at Validation Points")
                print(f"{torch.cat([model.nd_tensor, model.policy_assets(2.0*(model.nd_tensor - model.normalization_min) / (model.normalization_max - model.normalization_min) - 1)], dim=1)}")
                print("Value Function at Validation Points")
                print(f"{torch.cat([model.nd_tensor, model.policy_value(2.0*(model.nd_tensor - model.normalization_min) / (model.normalization_max - model.normalization_min)- 1) * (model.value_max - model.value_min) + model.value_min], dim=1)}")


        ### End of DeepHAM Training ###

        # Convert simulation data to NumPy
        data = model.inputs_simulation[:, :, model.burn_in:, 0].flatten().cpu().detach().numpy() # To plot the data we need to pass it to the cpu memory (RAM vs VRAM)

        plt.ioff()
        fig, ax = plt.subplots()
        # Final Histogram
        ax.clear()
        ax.hist(data, bins=50, density=True, alpha=0.6, color='b')
        ax.set_xlabel('Individual Capital (a)')
        ax.set_ylabel('Density')
        ax.set_title('Final Ergodic Distribution of Assets')
        plt.show()

        # Save the results for the ergodic distribution
        np.savetxt("results.txt", data, delimiter=",")  # Saves as a CSV-style text file
        # Save the model (NN) weights and biases. Note: state of the optimizer or normalization bounds is not saved!
        torch.save(model.state_dict(), "NN_weights_biases.pth")
        print("Model weights and biases saved to 'NN_weights_biases.pth'")
        print("End of Training.")


    # END OF THE EXECUTABLE


if __name__ == "__main__":
    main()
