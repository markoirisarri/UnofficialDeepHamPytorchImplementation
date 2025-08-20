# Unofficial DeepHAM (Han, Yang, and E, 2021) Implementation in Pytorch (repo under construction)

## Description

This repository contains a sample code for computing the Krusell Smith (1998) model by following the DeepHAM algorithm in Pytorch. Authorization from the original authors has been granted for this repository.

This repository is built entirely on the methodology of the original authors, [Original Paper](https://yangycpku.github.io/files/DeepHAM_paper.pdf). I am NOT an author of the original paper and this is purely an implementation for educational purposes. Please check the official repo at: [Original Repository](https://github.com/frankhan91/DeepHAM).

### Why DeepHAM and Deep Learning for Macroeconomics? 

Modern macroeconomic models are increasingly complex. To better understand economic phenomena, we build models with many state variables to account for:

    Heterogeneity: Different types of households and firms (e.g., varying income, wealth, age, or productivity).

    Frictions: Incomplete markets, borrowing constraints, and sticky prices.

    Aggregate Shocks: Multiple sources of uncertainty, such as productivity, monetary policy, and disaster risks.

While these models are richer and more realistic, they pose a significant computational challenge. Traditional solution methods, which often rely on creating a grid over the entire state space, suffer from the "Curse of Dimensionality": the computational and memory requirements grow exponentially with the number of state variables. This makes solving high-dimensional models practically impossible with standard techniques.

To overcome these challenges, this repository implements the DeepHAM algorithm (Han, Yang, and E, 2021), which leverages neural networks to find the model's solution. The advantages of this approach are threefold:

* Immunity to the Curse of Dimensionality

Traditional grid-based methods scale exponentially with the number of dimensions. In contrast, the training of a Neural Network (NN) relies on data sampling, which scales linearly.

This is a game-changer. Consider a model with 30 state variables:

    Grid Method: Discretizing each dimension with just 5 points would require storing and iterating over 5^30 ≈ 9.31 x 10²⁰ grid points—an impossible task.

    Deep Learning Method: A sampling-based approach would only require a dataset of size 30 x N, where N is the number of sample points (e.g., a few million).

Furthermore, the ability to train NNs using batches (small subsets of the data) ensures that memory consumption remains low and manageable, regardless of the total dataset size.

*  Accuracy Where It Matters: The Ergodic Set

Grid methods waste most of their computational effort. They strive for uniform accuracy across a vast hypercube of possible states, even in regions the economy will never realistically visit.

The DeepHAM algorithm's embedded simulation methodology ensures that training data is naturally drawn from the model's ergodic set—the region of the state space where the solution "lives" in the long run. As argued by Maliar et al. (2021), this is crucial for efficiency.

To illustrate, imagine the ergodic set is a hypersphere inside a hypercube:

    With d=2 dimensions, the sphere's area is 79% of the square's. A grid is reasonably efficient.

    With d=30 dimensions, the hypersphere's volume is just 2 x 10⁻¹⁴ of the hypercube's.

In high dimensions, a grid-based method spends virtually all its time refining the solution in irrelevant areas. A simulation-based approach, however, focuses its power exclusively on the states that matter, leading to far greater effective accuracy.

*  A Theoretical Guarantee: The Universal Approximation Theorem

We can be confident that a neural network is capable of representing the complex policy or value functions of our models. The Universal Approximation Theorem (Cybenko, 1989; Hornik et al., 1989) states that a simple feed-forward network with a single hidden layer and a sufficient number of neurons can approximate any continuous function to an arbitrary degree of accuracy.

A key condition is that the network's hidden layers use activation functions (like ReLU, tanh, or sigmoid) that are non-polynomial. In economics, this theorem provides a powerful theoretical guarantee: a sufficiently large NN can learn the true, potentially highly non-linear, functions that govern the behavior of agents in our models.

References:

    Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4), 303-314.

    Han, J., Yang, Y., & E, W. (2021). DeepHAM: A global solution method for heterogeneous agent models with aggregate shocks. arXiv preprint arXiv:2112.14377.

    Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. Neural networks, 2(5), 359-366.

    Maliar, L., Maliar, S., & Winant, P. (2021). Deep learning for solving dynamic economic models. Journal of Monetary Economics, 122, 76-101.

### Quick Visual Validation Exercise: Does Deep Learning Get the Invariant Distribution Right?

A key concern is whether we can trust the results obtained by Deep Learning, as there is little theoretical guarantee that the obtained results should converge. To address these concerns, the figure below provides a quick visual validation exercise of the DeepHAM methodology. 

<p align="center">
     <img src = https://github.com/markoirisarri/UnofficialDeepHAMPytorchImplementation/blob/main/distribution_individual_assets_validation_grid_based_deepham.png  >
</p>

The figure shows the invariant distribution over individual assets under the traditional grid-based methods (the Maliar 2010 implementation) and the results obtained by employing the present Pytorch DeepHAM implementation. There are two key points worth stressing:

*  First, the domain covered by both methodologies is similar (except at the upper tail, more details on this on the code). This suggests that the DeepHAM methodology can properly learn the true ergodic set of the model's solution.
*  Second, we obtain close distributions under both methodologies. This is a de-facto validation exercise on the entire policy function, as it would not be possible to obtain the right invariant distribution unless the policy function is correct at every point on the state space. 


## Structure

The structure of the contents is as follows:

Main folder: 

- "UnofficialDeepHAMPytorchImplementation.py": Source file for the implementation. Further comments and indications can be found within the file. It solves for the risky-steady state and then computes the obtained Bellman Errors and computes the IRFs to a negative aggregate productivity shock.
- "NN_weights_biases.pth": These are the obtained weights and biases of the Policy and Value NN after 51 outer DeepHAM iterations.
- "results.txt": This file containts the distribution of individual assets after 51 outer DeepHAM iterations. Its purpose is to serve as a visual validation tool.
- "distribution_individual_assets_validation_grid_based_DeepHAM.png": Figure showing the distribution over individual assets under both methodologies.  

/irf: Stores the time series of the aggregates across all simulated paths. Employed for plotting the IRFs. 

## Acknowledgements 

* I thank Yucheng Yang for his insightful comments that have improved this repository in every respect.
  
## Requirements:

* Cloud-based computing (Google Collab):
  * Please select mode GPU T4. Failure to do so will result in an error message. 

* Running the code locally 

  * CUDA-compatible GPU.
  * Pytorch GPU install.
    * Please download it from the official webpage: [link](https://pytorch.org/get-started/locally/)
  * CUDA toolkit (might also be installed in conjunction with Pytorch)
    * Can be downloaded from: [link](https://developer.nvidia.com/cuda-toolkit)
  * Nvidia Graphics Drivers
    * Will be often installed or updated as part of the CUDA toolkit

## Useful References

* Pytorch Documentation [link](https://docs.pytorch.org/docs/stable/index.html)
* TorchScript Documentation [link](https://docs.pytorch.org/docs/stable/jit.html)

