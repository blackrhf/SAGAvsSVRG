# SAGAvsSVRG
Comparative Analysis of SAG, SAGA, and SVRG on the a9a Dataset

This study presents a systematic comparison of three prominent stochastic optimization algorithms - Stochastic Average Gradient (SAG), Stochastic Average Gradient Augmented (SAGA), and Stochastic Variance Reduced Gradient (SVRG) - using the a9a benchmark dataset. The a9a dataset, derived from UCI Adult data for income prediction tasks, provides an ideal testbed with its 32,561 training samples and 123 sparse features, representing characteristic challenges in real-world machine learning applications.

Key aspects of our comparative framework include:

Convergence Properties:

Initial convergence speed (first 5-10 epochs)

Final solution quality after 30 epochs

Stability of optimization trajectory

Computational Efficiency:

Time per epoch

Memory requirements

Scaling with dataset size

Implementation Characteristics:

Sensitivity to learning rate

Handling of sparse features

Adaptability to non-convex scenarios

Our analysis reveals that while all three algorithms belong to the variance-reduced gradient family, they demonstrate markedly different performance profiles on this dataset. SVRG shows particular advantages in early-stage convergence, while SAGA offers better stability in later training phases. The standard SAG algorithm, while conceptually simpler, proves less competitive in both convergence speed and final solution quality.
