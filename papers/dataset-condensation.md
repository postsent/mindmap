# What did the authors tried to accomplished?

A **training set synehtsis technioque** called **data condensation** that learns to condense **large dataset** into a **small set** of informative synthetic samples.

# Key elements of the approach

1. formulate as gradient matching problem between the gradients of deep neural network weights that are trained on the original and our synthetic data.

# Takeaway

- i) compress a large image classification dataset into a small synthetic set, ii) train an image classification model on the synthetic set that can be further used to classify real images, iii) learn a single set of synthetic images that can be used to train different neural network architectures?

# Other references to follow

define **criterion** (e.g. diversity) **for representativeness**:
1. coreset construction (classical data selection methods, clustering problems)
2. continual learning
3. active learning


# Results (Good or Bad)

- does not rely on the presence of representative samples as the synthesized data are directly optimized for the downstream task
- outperforms the state-of-the-art methods e.g. **coreset construction**