# Data Distillation: A Survey

- [Data Distillation: A Survey](#data-distillation-a-survey)
- [What did the authors tried to accomplished?](#what-did-the-authors-tried-to-accomplished)
- [Key elements of the approach](#key-elements-of-the-approach)
  - [2.1 Data Distillation by Meta-model Matching](#21-data-distillation-by-meta-model-matching)
    - [DD dataset distillation (Wang et al., 2018) - bilevel optimisation on synethic and real data](#dd-dataset-distillation-wang-et-al-2018---bilevel-optimisation-on-synethic-and-real-data)
    - [KIP (Nguyen et al., 2021a, b) - Kernelized Ridge Regression (KRR)](#kip-nguyen-et-al-2021a-b---kernelized-ridge-regression-krr)
    - [RFAD (Loo et al., 2022) - replace with lightweight kernel](#rfad-loo-et-al-2022---replace-with-lightweight-kernel)
    - [FRePO (Zhou et al., 2022b) - decouple feature extractor & linear classifier & train alternatively](#frepo-zhou-et-al-2022b---decouple-feature-extractor--linear-classifier--train-alternatively)
  - [2.2 Data Distillation by Gradient Matching](#22-data-distillation-by-gradient-matching)
    - [DC (Zhao et al., 2021) - gradient matching objective](#dc-zhao-et-al-2021---gradient-matching-objective)
    - [DSA (Zhao & Bilen, 2021) - augmentation tailored to synethic data](#dsa-zhao--bilen-2021---augmentation-tailored-to-synethic-data)
    - [DCC (Lee et al., 2022b) - add class contrastive signals when gradient matching](#dcc-lee-et-al-2022b---add-class-contrastive-signals-when-gradient-matching)
    - [IDC - efficient synthetic-data parameterization (Kim et al., 2022) - downsample then upsample to remove spatial redundancies, match on full data over synethic (approx)](#idc---efficient-synthetic-data-parameterization-kim-et-al-2022---downsample-then-upsample-to-remove-spatial-redundancies-match-on-full-data-over-synethic-approx)
  - [2.3 Data Distillation by Trajectory Matching](#23-data-distillation-by-trajectory-matching)
    - [Trajectory Matching (Cazenavette et al., 2022)](#trajectory-matching-cazenavette-et-al-2022)
    - [TESLA Cui et al. (2022b) - scalable by re-parameterizes the parameter-matching loss of MTT & learnable soft-labels](#tesla-cui-et-al-2022b---scalable-by-re-parameterizes-the-parameter-matching-loss-of-mtt--learnable-soft-labels)
  - [2.4 Data Distillation by Distribution Matching](#24-data-distillation-by-distribution-matching)
    - [DM Zhao & Bilen (2023) - data distribution in latent space matching](#dm-zhao--bilen-2023---data-distribution-in-latent-space-matching)
    - [CAFE (Wang et al., 2022) - one encoder & consider intermediate layers](#cafe-wang-et-al-2022---one-encoder--consider-intermediate-layers)
    - [IT-GAN (Zhao & Bilen, 2022)](#it-gan-zhao--bilen-2022)
  - [Data Modalities](#data-modalities)
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


### DD dataset distillation (Wang et al., 2018) - bilevel optimisation on synethic and real data

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

### KIP (Nguyen et al., 2021a, b) - Kernelized Ridge Regression (KRR)
- Methods
  - KIP uses the NTK (Neural Tangent Kernel) of a **fully-connected neural network** (Nguyen et al., 2021a), or a **convolutional network** (Nguyen et al., 2021b) in the inner-loop for efficiency
  - solve the **inner-loop** in **closed form**
  - infinite-width correspondence
    - performing Kernelized Ridge Regression (KRR) using the NTK of a given neural network, is equivalent to training the same ∞-width neural network with L2 reconstruction loss for ∞ SGD-steps. 
    - These “∞-width” neural networks have been shown to perform reasonably compared to their **finite-width counterparts**, while also being solved in **closed-form**
- problems
  - 
  - not scalable
### RFAD (Loo et al., 2022) - replace with lightweight kernel
- **light-weight** Empirical Neural Network Gaussian Process (NNGP) kernel
  - improvement over KIP
- a **classification loss** (e.g., NLL) instead of the **L2-reconstruction loss** for the **outer-loop**

### FRePO (Zhou et al., 2022b) - decouple feature extractor & linear classifier & train alternatively
- Methods
  - decouples the **feature extractor** and a **linear classifier** in $Φ$
- Pros
  - more scalable & generalizable
- Cons
  - TODO

## 2.2 Data Distillation by Gradient Matching

In short
- Assumption
  1. **inner-loop** optimization of only **T steps**
  2. **local smoothness**: two sets of **model parameters** close to each other (given a distance metric) imply **model similarity**
  3. first-order approximation of $\theta^\mathcal{D}_t$
       - instead of exactly computing the training trajectory of optimizing θ0 on D
       - perform first-order approximation on the optimization trajectory of θ0 on the much smaller D_syn
       - i.e., approximate θD t as a single gradient-descent update on $\theta_{t-1}^{\mathcal D_{\textsf{syn}}}$ using D rather than $\theta_{t-1}^{\mathcal D}$
- pros
  - In contrast to the **meta-model matching** framework, such an approach circumvents the unrolling of the inner-loop, thereby making the overall optimization much more efficient.

Formula

$$
\underset{\mathcal{D}_{\mathrm{syn}}}{\arg \min } \underset{\substack{\theta_0 \sim \mathbf{P}_\theta \\ c \sim \mathcal{C}}}{\mathbb{E}}\left[\sum_{t=0}^T \mathbf{D}\left(\nabla_\theta \mathcal{L}_{\mathcal{D}^c}\left(\theta_t\right), \nabla_\theta \mathcal{L}_{\mathcal{D}_{\mathrm{syn}}^c}\left(\theta_t\right)\right)\right] \quad \text { s.t. } \quad \theta_{t+1} \leftarrow \theta_t-\eta \cdot \nabla_\theta \mathcal{L}_{\mathcal{D}_{\mathrm{syn}}}\left(\theta_t\right)
$$


### DC (Zhao et al., 2021) - gradient matching objective

pros
  - data summaries optimized by gradient-matching significantly outperformed heuristic data samplers, principled coreset construction techniques, TBPTT-based data distillation
  

### DSA (Zhao & Bilen, 2021) - augmentation tailored to synethic data

- improves over **DC** by 
- performing the same image-augmentations (e.g., crop, rotate, jitter, etc.) on both D and Dsyn while optimizing above formula
- Since these augmentations are **universal** and are applicable across data distillation frameworks, DSA augmentations have become a **common part** of all methods proposed henceforth

### DCC (Lee et al., 2022b) - add class contrastive signals when gradient matching

- further modifies the **gradient-matching objective** 
- to incorporate **class contrastive signals** inside each gradient-matching step 
- improve **stability** as well as **performance**.

### IDC - efficient synthetic-data parameterization (Kim et al., 2022) - downsample then upsample to remove spatial redundancies, match on full data over synethic (approx)

- extend the **gradient matching** framework by
- **multi-formation**: to synthesize a **higher amount** of data within the **same memory budget**, store the data summary (e.g., images) in a **lower resolution** to remove **spatial redundancies**, and **upsample** (using e.g., bilinear, FSRCNN (Dong et al., 2016)) to the original scale while usage;
- matching gradients of the network’s training trajectory over the **full dataset** D rather than the **data summary**
- hypothesis
  - training models on $D_{syn}$ instead of D in the inner-loop has two major drawbacks:
    - strong **coupling** of the inner- and outer-loop resulting in a **chicken-egg problem** (McLachlan & Krishnan, 2007)
    - **vanishing network gradients** due to the small size of $D_{syn}$, leading to an improper outer-loop optimization for gradient-matching based techniques


## 2.3 Data Distillation by Trajectory Matching

### Trajectory Matching (Cazenavette et al., 2022)  

- **long**-horizon **trajectory matching**
  - optimizing for similar quality models trained with N SGD steps on $D_{syn}$, compared to $M>>N$ steps on D
- it can be **pre-computed**
  - trajectory of training Φθ on D is independent of the optimization of $D_{syn}$
- inheriting the **local smoothness** assumption
  - first-order distance between parameters

### TESLA Cui et al. (2022b) - scalable by re-parameterizes the parameter-matching loss of MTT & learnable soft-labels

- using linear algebraic manipulations to make the bilevel optimization’s **memory** complexity **independent** of N
- uses **learnable soft-labels** ($Y_{syn}$) during the optimization for an increased compression efficiency.

## 2.4 Data Distillation by Distribution Matching

### DM Zhao & Bilen (2023) - data distribution in latent space matching
- TODO
- pros
  - much improved **scalability**

### CAFE (Wang et al., 2022) - one encoder & consider intermediate layers

- TODO

### IT-GAN (Zhao & Bilen, 2022)

## Data Modalities

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
