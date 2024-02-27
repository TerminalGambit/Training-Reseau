# TensorFlow Probability (probabilistic programming library built on TensorFlow)

# Application of the Monte Carlo Markov Chain using TensorFlow Probability

# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(0)
tf.random.set_seed(0)

# Define the model
def model():
    # Define the model
    model = tfp.distributions.JointDistributionSequential([
        tfp.distributions.Normal(loc=0., scale=1.), # Prior
        lambda loc: tfp.distributions.Normal(loc=loc, scale=1.) # Likelihood
    ])
    return model

# Define the log joint probability
def log_joint_prob(loc, data):
    model = model()
    return model.log_prob((loc, data))

# Define the MCMC
def mcmc():
    # Define the number of samples
    num_samples = 1000

    # Define the initial state
    initial_state = 0.

    # Define the step size
    step_size = 0.5

    # Define the number of burn-in steps
    num_burnin_steps = 500

    # Define the kernel
    kernel = tfp.mcmc.RandomWalkMetropolis(log_joint_prob)

    # Define the trace
    trace = tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=initial_state,
        kernel=kernel,
        num_burnin_steps=num_burnin_steps,
        trace_fn=None
    )

    return trace

# Define the main function
def main():
    # Define the data
    data = 1.

    # Define the trace
    trace = mcmc()

    # Plot the trace
    plt.plot(trace)
    plt.show()

# Execute the main function
if __name__ == "__main__":
    main()