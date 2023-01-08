💯 - fully read  
✅ - Read & Summarised (can skim experiment & supplement)  
❎ - Read & not fully summarised    
☑️ - Reading  
🔲 - TO READ

Note
- Add submitted data if different to conference year

# Data Distillation

| Title | Authors | Date | status | Description | Comments |
| --- | --- | --- |--- | --- |--- |
| Dataset Distillation (DD) | Tongzhou Wang | 2018 | ☑️ | First formal paper introduces the concept of dataset distillation, inspired by Knowldege distillation | --- |
| Dataset Condensation with gradient matching (DC) | Bo Zhao |2020, ICLR 2021 | ✅ | Improve DD with Gradient surrogate objective (faster, efficient) | --- |
| Dataset Condensation with Distribution Matching |  Bo Zhao | 2021, WACV 2023  | 🔲 | Distribution Matching Surrogate Objective | --- |
| Dataset Condensation with Differentiable Siamese Augmentation (DSM) | Bo Zhao | 2021, ICML 2021 | 🔲 | Improve DC with data augmentation techniques tailored to sythetic data since the traditional ones give poor performance gain |

# Graph Distillation

| Title | Authors | Date | status | Description | Comments |
| --- | --- | --- |--- | --- | --- |
| Graph Condensation for Graph Neural Networks (GCOND) | Wei Jin | 2021, ICLR 2022  | ❎ | Adapt DD & DC to graph setting | --- |
| Condensing Graphs via One-Step Gradient Matching |  Wei Jin | KDD 2022| ❎ | - Improve GCOND with one step gradient update <br/> - GCOND does not produce discrete graph structures and its condensation process is costly. <br/> - Approximates the overall gradient matching loss for 𝜃𝑡 with the initial matching loss at the first epoch, which they term as one-step matching loss | --- |
| Graph Condensation via Receptive Field Distribution Matching (GCDM) | Mengyang Liu | 2022| ❎ | Adapt DCDM to graph setting | --- |

# GNNs

| Title | Authors | Date | status | Description | Comments |
| --- | --- |--- | --- |--- | -- | 
| Semi-Supervised Classification with Graph Convolutional Networks (GCN) | Thomas N. Kipf | 2016, ICLR 2017 | ☑️ | First popular paper adopt convolution to graph |  --- |
| Graph Attention Networks (GAT) | Petar Veličković | 2017, ICLR 2018 | ☑️ | First popular paper adopt attention mechansim to graph | --- |
| A Comprehensive Survey on Graph Neural Networks | Zonghan Wu |  2020, IEEE Trans. Neural Netw 2021  | ☑️ | --- |  --- |

# More

Dataset Distillation paper list - https://github.com/Guang000/Awesome-Dataset-Distillation

GNN paper list - https://github.com/GRAND-Lab/Awesome-Graph-Neural-Networks

# Inbox

- [Nguyen et al. NeurIPS 2021]: distillation w.r.t. the infinite-width limit Neural Tangent Kernel.
- [Kim et al., ICML 2022]: reparametrizing distilled dataset via multi-scale patches.
- [Lee et al., ICML 2022]: careful scheduling of class-wise and class-collective objectives.
- [Such et al., ICML 2020]: training a generator that outputs good synthetic trianing data, with application in Neural Architecture Search.
- [Deng et al., 2022]: new reparametrization that improves distillation via simple backpropagation through optimization steps, with application in continual learning.
- G. Cazenavette, T. Wang, A. Torralba, A. A. Efros, J. Y. Zhu. "Dataset Distillation by Matching Training Trajectories". CVPR 2022.
- G. Cazenavette, T. Wang, A. Torralba, A. A. Efros, J. Y. Zhu. "Wearable ImageNet: Synthesizing Tileable Textures via Dataset Distillation". Workshop on Computer Vision for Fashion, Art, and Design at CVPR 2022.