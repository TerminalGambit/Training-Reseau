# TensorFlow Probability (probabilistic programming library built on TensorFlow)

# Application of the Monte Carlo Markov Chain using TensorFlow Probability

# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Define the model using TensorFlow Probability's JointDistributionSequential
model = tfp.distributions.JointDistributionSequential([
    tfp.distributions.Normal(loc=0., scale=1.),  # Prior distribution
    lambda loc: tfp.distributions.Normal(loc=loc, scale=1.)  # Likelihood given the prior
])

# Define a log joint probability function with data
def make_log_joint_prob(data):
    def log_joint_prob(loc):
        return model.log_prob((loc, data))
    return log_joint_prob

# Define the Monte Carlo Markov Chain (MCMC) function
def mcmc(data):
    # Define the number of samples and the number of burn-in steps
    num_samples = 1000
    num_burnin_steps = 500

    # Initial state for the chain
    initial_state = np.array(0., dtype=np.float32)

    # Define the MCMC kernel
    kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=make_log_joint_prob(data),
    )

    # Sample from the Markov chain
    states, _ = tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=initial_state,
        kernel=kernel,
        num_burnin_steps=num_burnin_steps,
        trace_fn=None  # Adjust here if you're interested in diagnostics
    )

    return states


# Main function to execute the MCMC simulation and plot the results
def main():
    # Example data point
    data = np.array(1., dtype=np.float32)

    # Run MCMC to sample from the posterior given the data
    trace = mcmc(data)

    # Plotting the trace of the samples
    plt.plot(trace)
    plt.xlabel('Sample index')
    plt.ylabel('Position')
    plt.title('MCMC Trace')
    plt.show()

# Execute the main function
if __name__ == "__main__":
    main()

