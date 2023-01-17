# Data Distillation: A Survey

- [Data Distillation: A Survey](#data-distillation-a-survey)
- [What did the authors tried to accomplished?](#what-did-the-authors-tried-to-accomplished)
- [Key elements of the approach](#key-elements-of-the-approach)
  - [2.1 Data Distillation by Meta-model Matching](#21-data-distillation-by-meta-model-matching)
    - [DD dataset distillation - Wang et al. (2018)](#dd-dataset-distillation---wang-et-al-2018)
    - [KIP - (Nguyen et al., 2021a), (Nguyen et al., 2021b)](#kip---nguyen-et-al-2021a-nguyen-et-al-2021b)
    - [RFAD (Loo et al., 2022)](#rfad-loo-et-al-2022)
    - [FRePO (Zhou et al., 2022b)](#frepo-zhou-et-al-2022b)
  - [2.2 Data Distillation by Gradient Matching](#22-data-distillation-by-gradient-matching)
  - [2.3 Data Distillation by Trajectory Matching](#23-data-distillation-by-trajectory-matching)
  - [2.4 Data Distillation by Distribution Matching](#24-data-distillation-by-distribution-matching)
- [Results (Good or Bad)](#results-good-or-bad)
- [Other references to follow](#other-references-to-follow)
- [Takeaway](#takeaway)
- [TODO](#todo)

**Keywords**:
- Data Distillation
- KD, transfer learning, model compression
- current DD framework
  - gradient matching
  - distribution matching
  - trajectory matching
  - meta-model matching
  - factorisation

**TLDR**

- explains
  - benefits of bringing a **faster model-training procedure**
  - applications
  - formal data distillation framework
  - 
- Comparison with **knowledge distillation** & **transfer learning**
- contribution

**openreview**

# What did the authors tried to accomplished?

**Main idea.**  TODO  
**Motivation.** TODO  
**Previous problems.** TODO  


# Key elements of the approach

## 2.1 Data Distillation by Meta-model Matching

In short
- the **inner-loop** trains a representative learning algorithm **on the data summary** until convergence
- the **outer-loop** subsequently optimizes the data summary for the **transferability to the original dataset** of the optimized learning algorithm 
- assumption
  - TODO

Formula

$$
\underset{\mathcal{D}_{\text {syn }}}{\arg \min } \quad \mathcal{L}_{\mathcal{D}}\left(\theta^{\mathcal{D}_{\text {syn }}}\right) \quad \text { s.t. } \quad \theta^{\mathcal{D}_{\text {syn }}} \triangleq \underset{\theta}{\arg \min } \mathcal{L}_{\mathcal{D}_{\text {syn }}}(\theta)
$$


### DD dataset distillation - Wang et al. (2018) 
- Methods
  -  **inner-loop**
     -  local optimization 
  -  **outer loop**
     -  Truncated Back-Propagation Through Time (TBPTT)
        -  unroll a **limited** number of inner-loop optimization steps while optimizing the outer-loop

- problems 
  - TBPTT 
     1. **computationally expensive** to unroll the inner-loop at each outer-loop update
     2. **bias** involved with truncated unrolling
     3. **poorly** conditioned **loss landscapes**, particularly with long unrolls

### KIP - (Nguyen et al., 2021a), (Nguyen et al., 2021b)
- Methods
  - KIP uses the NTK (Neural Tangent Kernel) of a **fully-connected neural network** (Nguyen et al., 2021a), or a **convolutional network** (Nguyen et al., 2021b) in the inner-loop for efficiency
  - solve the **inner-loop** in **closed form**
  - infinite-width correspondence
    - performing Kernelized Ridge Regression (KRR) using the NTK of a given neural network, is equivalent to training the same ∞-width neural network with L2 reconstruction loss for ∞ SGD-steps. 
    - These “∞-width” neural networks have been shown to perform reasonably compared to their **finite-width counterparts**, while also being solved in **closed-form**
- problems
  - 
  - not scalable
### RFAD (Loo et al., 2022)
- **light-weight** Empirical Neural Network Gaussian Process (NNGP) kernel
  - improvement over KIP
- a **classification loss** (e.g., NLL) instead of the **L2-reconstruction loss** for the **outer-loop**

### FRePO (Zhou et al., 2022b)
- Methods
  - decouples the **feature extractor** and a **linear classifier** in $Φ$
- Problems
  - TODO

## 2.2 Data Distillation by Gradient Matching

- Assumption
  1. **inner-loop** optimization of only **T steps**
  2. **local smoothness**: two sets of **model parameters** close to each other (given a distance metric) imply **model similarity**
  3. first-order approximation of $\theta^\mathcal{D}_t$
       - instead of exactly computing the training trajectory of optimizing θ0 on D
       - perform first-order approximation on the optimization trajectory of θ0 on the much smaller D_syn
       - i.e., approximate θD t as a single gradient-descent update on $\theta_{t-1}^{\mathcal D_{\textsf{syn}}}$ using D rather than $\theta_{t-1}^{\mathcal D}$
 
## 2.3 Data Distillation by Trajectory Matching

## 2.4 Data Distillation by Distribution Matching



# Results (Good or Bad)

(from conclusion)

# Other references to follow

- Deng & Russakovsky, 2022 - using momentum-based optimizers
- 

**More explanation**

**More papers**



# Takeaway

(what can be used in my part)

# TODO

1. summary
2. author / others explanation video / article
3. openreview
