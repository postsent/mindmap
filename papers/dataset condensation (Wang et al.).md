# What did the authors tried to accomplished?

**Main idea.** A training set synehtsis  technioque called **data condensation** that learns to condense **large** dataset into a **small** set of informative synthetic samples.

**Goal.**
1. highest generalization performance - trained on synethic comparable to the original dataset 
2. 

**Previous problems.** Relies on 
1. **heuristics** (e.g. picking cluster centers) that does not guarantee any optimal solution for the downstream task (e.g. image classification)
2. presence of **representative samples**, which is neither guaranteed.

**Motivation.** 
1. Dataset Distillation (DD)

# Key elements of the approach

1. formulate as gradient matching problem between the gradients of deep neural network weights that are trained on the original and our synthetic data.

# Takeaway

- i) compress a large image classification dataset into a small synthetic set, ii) train an image classification model on the synthetic set that can be further used to classify real images, iii) learn a single set of synthetic images that can be used to train different neural network architectures?

# Other references to follow

**first paper**
1. Dataset Distillation (DD) - 
   

define **criterion** (e.g. diversity) **for representativeness**:
1. coreset construction (classical data selection methods, clustering problems)
2. continual learning
3. active learning


# Results (Good or Bad)

- does not rely on the presence of representative samples as the synthesized data are directly optimized for the downstream task
- outperforms the state-of-the-art methods e.g. **coreset construction**