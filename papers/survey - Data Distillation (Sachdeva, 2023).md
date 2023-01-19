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
  - [2.5 Data Distillation by Factorisation](#25-data-distillation-by-factorisation)
    - [LinBa (Deng & Russakovsky, 2022) - MF with TBPTT](#linba-deng--russakovsky-2022---mf-with-tbptt)
    - [HaBa (Liu et al., 2022c) - relax hallucinator assumption of LinBa](#haba-liu-et-al-2022c---relax-hallucinator-assumption-of-linba)
    - [KFS (Lee et al. , 2022a)](#kfs-lee-et-al--2022a)
  - [Data Distillation vs. Data Compression (Matrix Factorisation)](#data-distillation-vs-data-compression-matrix-factorisation)
  - [Data Modalities](#data-modalities)
    - [Images](#images)
    - [Text](#text)
    - [Graph](#graph)
      - [GCond (Jin et al., 2022b) - node classification & Gradient Matching](#gcond-jin-et-al-2022b---node-classification--gradient-matching)
      - [(Liu et al., 2022a) (GCDM) - distribution matching](#liu-et-al-2022a-gcdm---distribution-matching)
      - [DosCond - Jin et al. (2022a) - dedicated matrix for graph & graph classifcation & single step](#doscond---jin-et-al-2022a---dedicated-matrix-for-graph--graph-classifcation--single-step)
    - [Recommender Systems](#recommender-systems)
      - [Distill-CF (Sachdeva et al., 2022a) - implicit-feedback](#distill-cf-sachdeva-et-al-2022a---implicit-feedback)
  - [5 Challenges & Future Directions](#5-challenges--future-directions)
    - [New data modalities & settings](#new-data-modalities--settings)
    - [Better scaling](#better-scaling)
    - [Improved optimization](#improved-optimization)
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
  - KIP uses the NTK (Neural Tangent Kernel) of {} in the inner-loop for efficiency
    - a **fully-connected neural network** (Nguyen et al., 2021a), or 
    - a **convolutional network** (Nguyen et al., 2021b) 
  
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

## 2.5 Data Distillation by Factorisation

- previous methods
  - maintain the synthesized data summary as a large set of **free parameters**, which are in turn optimized
  - prohibits **knowledge sharing** between synthesized data points (parameters)
    - introduce data redundancy
- factorization-based data distillation techniques
  - parameterize the data summary using two **separate** components (standard MF)
    1. **bases**: a set of **mutually independent** - **base vectors**
    2. **hallucinators**: a **mapping** from the bases’ vector space to the joint data- and label-space
- pros
  - hallucinator-bases data parameterization can be **optimized** using **any** of the **aforementioned** data optimization **frameworks**

### LinBa (Deng & Russakovsky, 2022) - MF with TBPTT

- assumption
  - the bases’ vector space (B) to be the same as the task input space (X)
  - the hallucinator to be linear and additionally conditioned on a given predictand
- methods
  - B and H are jointly optimized using the TBPTT framework
  - crucial modifications
    - using **momentum-based optimizers** instead of vanilla SGD in the inner-loop
    - **longer unrolling** (≥ 100 steps) of the inner-loop during TBPTT
  
### HaBa (Liu et al., 2022c) - relax hallucinator assumption of LinBa

- relax the linear and predictand-conditional hallucinator assumption of LinBa

### KFS (Lee et al. , 2022a)

- methods
  - maintaining a different bases’ vector space B from the data domain X , such that dim(B) < dim(X )
  - optimized using **distribution matching** framework
    - ensure **fast, single-level optimization**.
- pros
  - This parameterization allows KFS to store an even larger number of images, with a comparable storage budget to other methods

## Data Distillation vs. Data Compression (Matrix Factorisation)

- can not be compared directly
- efficiency metric
  - size of the data summary (n)  
    - Non-factorised method
    - Factorised method
      - poorer
        - need a much smaller **storage budget** to synthesize the same-sized data summaries
  - “end-to-end bytes of storage”
    - Non-factorised method
      - poorer
        - perform **no** kind of **data compression**, but focus solely on better understanding the **model-to-data relationship** through the lens of **optimization**
    - Factorised method
  
Side
- under the same **storage budget**, MF can provide more images and thus not fair to compare based on number
- Also cannot just compare based on size of storage because the non-MF techniques dont have **data compression** techniques but study the data distribution by model hidden dynamic matching.
- PCA is a form of MF.


## Data Modalities

### Images

### Text

### Graph

- applications
  - user-item interactions
  - social networks
  - autonomous driving
- **challenges**: synthesizing tiny, high-fidelity graphs
  - **data variety.** nodes in a graph can be **highly abstract**, e.g., users, products, text articles, etc. some of which could be discrete, heterogeneous, or even simply numerical IDs
  - graphs follow a variety of **intrinsic patterns** (e.g., spatial (Kipf & Welling, 2017)) which need to be **retained** in the distilled graphs; 
  - **quadratic size** of the **adjacency matrix** could be computationally prohibitive even for moderate-sized graphs

#### GCond (Jin et al., 2022b) - node classification & Gradient Matching

- inductive node-classification setting
- methods
  - model adjacency matrix as a function of node features
  - use gradient-matching framework

#### (Liu et al., 2022a) (GCDM) - distribution matching

- methods
  - shares the same framework as GCond
  - uses the distribution matching

#### DosCond - Jin et al. (2022a) - dedicated matrix for graph & graph classifcation & single step

- methods
  - extend to **graph-classification** setting (also have node classification)
  - improved on GCOND
    - maintain a **free-parameter matrix** Ω with the same size as the **adjacency matrix**, and sample each Ai,j syn entry through an independent **Bernoulli** draw on Ωi,j as the prior using the **reparameterization trick** (Maddison et al., 2017)
      -  instead of parameterizing the adjacency matrix using a similarity function on $X_{syn}$
    - gradient-matching - single-step
- pros
  - ensures **differentiability** as well as **discrete matrix** synthesis

### Recommender Systems

- challenges
  - data in the form of **abstract** and **discrete** (userID, itemID, relevance) tuples, which departs from the typical (features, label) setup;
  - the distribution of both user- and item-popularity follows a strong **power-law** which leads to **data scarcity** and **unstable optimization**; and (3) the data inherits a variety of **inherent structures**, e.g., **sequential patterns** (Kang & McAuley, 2018; Sachdeva et al., 2019), u**ser-item graph patterns** (Wu et al., 2019), **item-item co-occurrence patterns** (Steck, 2019), **missing-not-at-randomness** (Sachdeva et al., 2020; Schnabel et al., 2016), etc.

#### Distill-CF (Sachdeva et al., 2022a) - implicit-feedback

- aim
  - distills implicit-feedback recommender systems data, i.e., when the observed user-item relevance is binary (e.g., click or no-click)
- methods
  - meta-model matching framework (Nguyen et al. (2021a))
  - infinite-width autoencoders
    - suit the task of **item recommendation**
    - while also leading to **closed-form** computation of the inner-loop in the meta-model matching

## 5 Challenges & Future Directions

### New data modalities & settings

### Better scaling

### Improved optimization


# Takeaway

(what can be used in my part)

# TODO

1. summary
2. author / others explanation video / article
3. openreview
