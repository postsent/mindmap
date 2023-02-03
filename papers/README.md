<img src="https://img.shields.io/badge/Inbox-4-red" alt="Inbox"/> <img src="https://img.shields.io/badge/Read-8-green" alt="Read"/> 

<img src="https://img.shields.io/badge/Goal-50-blue" alt="Goal"/> <img src="https://img.shields.io/badge/Weekly %20Minimum-2-green" alt="Weekly Minimum"/> 

<img src="https://img.shields.io/badge/GNNs-✓-9cf" alt="GNNs"/> <img src="https://img.shields.io/badge/Dataset%20Distillation-✓-9cf" alt="Dataset Distillation"/> <img src="https://img.shields.io/badge/Classic-✓-9cf" alt="Classic"/> <img src="https://img.shields.io/badge/Interests-✓-9cf" alt="Interests"/>



TOC

- [My Reading List](#my-reading-list)
- [Data Distillation](#data-distillation)
- [Graph Distillation](#graph-distillation)
  - [Recommender System](#recommender-system)
- [Survey - data distillation](#survey---data-distillation)
- [More](#more)
- [Inbox](#inbox)
- [Common key words](#common-key-words)
# My Reading List

status

    💯 | fully read  
    ✅ | Read & Summarised (can skim experiment & supplement)  
    ❎ | Read & not fully summarised    
    ☑️ | Reading  
    🔲 | TO READ  

Note
- Paper file is named by "id-title-reference"
- Add submitted date if different to conference year
- Paper list ordered by APA reference date

| Title | Authors | Date | status | Description | Comments |


# Data Distillation

* [DD - Dataset Distillation]() ; Tongzhou Wang ; 2018 ; ✅ ; First formal DD paper, inspired by KD, Bi-level Opt ; [🔖summary](DD%20-%20dataset%20distillation%20(Wang,%202018.11).md) ;
  
* [DC - Dataset Condensation with gradient matching](); Bo Zhao ;2020, ICLR 2021 ; ✅ ; Improve DD with Gradient surrogate objective (faster, efficient) ; [🔖summary](DC%20-%20dataset%20condensation%20(Zhao,%202020.6).md) ;
  
* [DSA - Dataset Condensation with Differentiable Siamese Augmentation](); Bo Zhao ; 2021, ICML 2021 ; ❎ ; Improve DC with data augmentation techniques tailored to sythetic data, Traditional ones give poor performance gain ; [🔖summary](DSA%20-%20data%20augmentation%20(Zhao,%202021).md) ; 

* [DM - Dataset Condensation with Distribution Matching]();  Bo Zhao ; 2021, WACV 2023  ; ❎ ; Objective: embedding space dist matching; Fast   ; [🔖summary](DM%20-%20distribution%20matching%20(Zhao%20&%20Bilen,%202021.10).md) ;

* [KIP-FC - Dataset meta-learning from kernel ridge-regression](); Timothy Nguyen ; ICLR 2021 ; ☑️ ; $∞$-FC, NTK, KRR ; --- ;

* [KIP-Conv - Dataset distillation with infinitely wide convolutional networks](); Timothy Nguyen ; NeurIPS 2021 ; 🔲 ; $∞$-Conv, First to reach 80/84.8% w/ 50 synethic images, c10 ; --- ;

* [MTT - Dataset Distillation by Matching Training Trajectories](); George Cazenavette, Tongzhou Wang (2nd) ; 2022, CVPR 2022 ; 🔲 ; Large-scale datasets, long-range training dynamics, match segments of pre-recorded trajectories from models trained on real data, Prior works either computational expensive or short range (single training step) approximation ; --- ;

# Graph Distillation

* [GCOND/GC - Graph Condensation for Graph Neural Networks]() ; Wei Jin ; 2021, ICLR 2022  ; ✅ ; Adapt DD & DC to graph setting ; [🔖summary](GCOND%20(Jin,%202021.10).md) ;
  
* [DosCond/GC-One - Condensing Graphs via One-Step Gradient Matching]();  Wei Jin ; KDD 2022; ❎ ; graph level task ; --- ;

* [GCDM - Graph Condensation via Receptive Field Distribution Matching](); Mengyang Liu ; 2022; ❎ ; Adapt DM to graph setting ; --- ;

## Recommender System

* [DISTILL-CF/KIP-RecSys - Infinite Recommendation Networks: A Data-Centric Approach](); Noveen Sachdeva ; NeurIPS 2022 ; ☑️ ; $∞$-AE; --- ;



# Survey - data distillation

* [Data Distillation: A Survey](); Noveen Sachdeva, J.McAuley ; 2023 ; ✅ ; First to summarise the overal framework. formulation, definition, comparison, pros&cons, assumption, discussion ; [🔖summary](survey%20-%20Data%20Distillation%20(Sachdeva,%202023).md)  ;
  
* [Dataset Distillation: A Comprehensive Review](); Ruonan Yu ; 2023 ; ☑️ ; Very comprehensive ; --- ;


# More

Dataset Distillation paper list - https://github.com/Guang000/Awesome-Dataset-Distillation

GNN paper list - https://github.com/GRAND-Lab/Awesome-Graph-Neural-Networks

# Inbox

[An internal markdown file link - those that are skimed and skipped](misc/skim.md)

**Progress**

- [Nguyen et al. NeurIPS 2021]: distillation w.r.t. the infinite-width limit Neural Tangent Kernel.
- [Kim et al., ICML 2022]: reparametrizing distilled dataset via multi-scale patches.
- [Lee et al., ICML 2022]: careful scheduling of class-wise and class-collective objectives.
- [Such et al., ICML 2020]: training a generator that outputs good synthetic trianing data, with application in Neural Architecture Search.
- [Deng et al., 2022]: new reparametrization that improves distillation via simple backpropagation through optimization steps, with application in continual learning.
- Synthesizing Informative Training Samples with GAN - https://openreview.net/forum?id=frAv0jtUMfS

**Survey**

- ...


# Common key words

Meta-model matching
- KIP: Kernel introducing points
- KRR: Kernelized Ridge Regression
- NTK: Neural Tangent Kernel 
- $∞$-width
- closed-form

Gradient matching

Distribution matching

Trajectory matching

Factorisation

Others

- Data augmentation
- Label Distillation
