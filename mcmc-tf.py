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
model = tfp.distributions.JointDistributionSequential([
    tfp.distributions.Normal(loc=0., scale=1.),  # Prior
    lambda loc: tfp.distributions.Normal(loc=loc, scale=1.)  # Likelihood
])

# Define the log joint probability with data parameter
def make_log_joint_prob(data):
    def log_joint_prob(loc):
        return model.log_prob((loc, data))
    return log_joint_prob

# Define the MCMC with data parameter
def mcmc(data):
    # Define the number of samples and burn-in steps
    num_samples = 1000
    num_burnin_steps = 500

    # Define the initial state and step size
    initial_state = np.array(0., dtype=np.float32)
    step_size = np.array(0.5, dtype=np.float32)

    # Define the kernel with the log joint probability function specific to the given data
    kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=make_log_joint_prob(data),
    )

    # Sample from the chain
    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=initial_state,
        kernel=kernel,
        num_burnin_steps=num_burnin_steps,
        trace_fn=None
    )

    return states

# Main function to run the MCMC and plot results
def main():
    # Define the data
    data = np.array(1., dtype=np.float32)

    # Run MCMC
    trace = mcmc(data)

    # Plot the trace
    plt.plot(trace)
    plt.xlabel('Sample index')
    plt.ylabel('Position')
    plt.title('MCMC Trace')
    plt.show()

# Execute the main function
if __name__ == "__main__":
    main()
