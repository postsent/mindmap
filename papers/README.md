# My Reading List

status

    💯 | fully read  
    ✅ | Read & Summarised (can skim experiment & supplement)  
    ❎ | Read & not fully summarised    
    ☑️ | Reading  
    🔲 | TO READ  
    --- | can skip / skim

Note
- Paper file is named by "id-title-reference"
- Add submitted date if different to conference year
- Paper list ordered by APA reference date

TOC

- [My Reading List](#my-reading-list)
- [Data Distillation](#data-distillation)
- [Graph Distillation](#graph-distillation)
- [Survey - data distillation](#survey---data-distillation)
- [More](#more)
- [Inbox](#inbox)

# Data Distillation

| Title | Authors | Date | status | Description | Comments |
| --- | --- | --- | --- | --- |--- |
| Dataset Distillation (DD) | Tongzhou Wang | 2018 | ❎ | First formal paper introduces the concept of dataset distillation, inspired by Knowldege distillation | [summary](DD%20-%20dataset%20distillation%20(Wang,%202018.11).md) |
| Dataset Condensation with gradient matching (DC) | Bo Zhao |2020, ICLR 2021 | ✅ | Improve DD with Gradient surrogate objective (faster, efficient) | [summary](DC%20-%20dataset%20condensation%20(Zhao,%202020.6).md) |
| Dataset Condensation with Differentiable Siamese Augmentation (DSA) | Bo Zhao | 2021, ICML 2021 | 🔲 | Improve DC with data augmentation techniques tailored to sythetic data  | Traditional ones give poor performance gain | 
| Dataset Condensation with Distribution Matching (DCDM) |  Bo Zhao | 2021, WACV 2023  | ☑️ | Distribution Matching Surrogate Objective | --- |
| Dataset Distillation by Matching Training Trajectories | George Cazenavette, Tongzhou Wang (2nd) | 2022, CVPR 2022 | 🔲 | Large-scale datasets, long-range training dynamics, match segments of pre-recorded trajectories from models trained on real data | Prior works either computational expensive or short range (single training step) approximation |

# Graph Distillation

| Title | Authors | Date | status | Description | Comments |
| --- | --- | --- | --- | --- | --- |
| Graph Condensation for Graph Neural Networks (GCOND/GC) | Wei Jin | 2021, ICLR 2022  | ✅ | Adapt DD & DC to graph setting | [summary](GCOND%20(Jin,%202021.10).md) |
| Condensing Graphs via One-Step Gradient Matching (DosCond/GC-One) |  Wei Jin | KDD 2022| ❎ | - Improve GCOND with one step gradient update <br/> - GCOND does not produce discrete graph structures and its condensation process is costly. <br/> - Approximates the overall gradient matching loss for 𝜃𝑡 with the initial matching loss at the first epoch, which they term as one-step matching loss | --- |
| Graph Condensation via Receptive Field Distribution Matching (GCDM) | Mengyang Liu | 2022| ❎ | Adapt DCDM to graph setting | --- |
| Infinite Recommendation Networks: A Data-Centric Approach | Noveen Sachdeva | NeurIPS 2022 | 🔲 | RecSys, infinite-width | --- |



# Survey - data distillation

| Title | Authors | Date | status | Description | Comments |
| --- | --- | --- | --- | --- | --- |
| A Review of Dataset Distillation for Deep Learning | Thi-Thu-Huong Le | 2022 | --- | First survey to list previous work out | --- |
| Data Distillation: A Survey | Noveen Sachdeva, J.McAuley | 2023 | ❎ | First to summarise the overal framework. formulation, definition, comparison, pros&cons, assumption, discussion | [summary](survey%20-%20Data%20Distillation%20(Sachdeva,%202023).md)  |
| A Comprehensive Survey to Dataset Distillation | S.Lei & D.Tao | 2023 | --- | Explain with framework & list out previous works | --- |


# More

Dataset Distillation paper list - https://github.com/Guang000/Awesome-Dataset-Distillation

GNN paper list - https://github.com/GRAND-Lab/Awesome-Graph-Neural-Networks

# Inbox

**Progress**

- [Nguyen et al. NeurIPS 2021]: distillation w.r.t. the infinite-width limit Neural Tangent Kernel.
- [Kim et al., ICML 2022]: reparametrizing distilled dataset via multi-scale patches.
- [Lee et al., ICML 2022]: careful scheduling of class-wise and class-collective objectives.
- [Such et al., ICML 2020]: training a generator that outputs good synthetic trianing data, with application in Neural Architecture Search.
- [Deng et al., 2022]: new reparametrization that improves distillation via simple backpropagation through optimization steps, with application in continual learning.
- Synthesizing Informative Training Samples with GAN - https://openreview.net/forum?id=frAv0jtUMfS

**Survey**

- ...