**Title:** GRAPH CONDENSATION FOR GRAPH NEURAL NETWORKS

# What did the authors tried to accomplished?

Aims to minimize the performance gap between GNN models trained on a synthetic, simplified graph and the original training graph.  
  

# Key elements of the approach

1. formulate the objective for graph condensation tractable for learning
2. how to parameterize the to-be-learned node features and graph structure
   - strategy of parameterizing the condensed features as free parameters and model the synthetic graph structure as a function of features

**Objective:** **Bi-level problem**  
i.e. minimise the model's loss on the original dataset and select the optimal parameters that minimise the loss on sythetic dataset

$$
\mathop{\operatorname*{min}}_{\mathcal{S}}\mathcal{L}\left(\mathrm{GNN}_{\mathcal{\theta}}(\mathbf{A},\mathbf{X}),\mathbf{Y}\right)\quad\mathrm{s.t}\quad\mathcal{\theta}_{\mathcal{S}}=\mathop{\mathrm{argmin}}_{\mathcal{\theta}}(\mathrm{GNN}_{\theta}(\mathbf{A}^{\prime},\mathbf{X}^{\prime}),\mathbf{Y}^{\prime})
$$  


**Bypass the bi-level optimization & generalisation**   
(since expensive), match the parameter gradient  

$$
D\left(\boldsymbol{\theta}_{t}^{\mathcal S},\boldsymbol{\theta}_{t}^{\mathcal T}\right)
$$
, 
$D$ is the distinace formula e.g. cos sim:  

$$
\underset{\mathcal{S}}{\operatorname*{min}\operatorname{E}_{\boldsymbol{\theta}_0}\sim P_{\boldsymbol{\theta}_0}}\left[\sum_{t=0}^{T-1}D\left(\nabla_{\boldsymbol{\theta}}\mathcal{L}\left(\operatorname{GNN}_{\boldsymbol{\theta}_t}(\mathbf{A}',\mathbf{X}'),\mathbf{Y}^{\prime}\right),\nabla_{\boldsymbol{\theta}}\mathcal{L}\left(\operatorname{GNN}_{\boldsymbol{\theta}_t}(\mathbf{A},\mathbf{X}),\mathbf{Y}\right)\right)\right]
$$

**Model graph as a function of node features**  
1\) so that the correlation between the graph and node features are also modelled, rather treating them independently. 2) avoiding jointly learning $O(N^2)$ parameters - less risk of overfitting as $N'$ gets larger.  

$$
\mathbf{A}'=g_{\Phi}(\mathbf{X}'),\quad\text{with A}'_{ij}=\mathbf{Sigmod}\left(\dfrac{\mathbf{ML}\mathbf{P}_{\Phi}([\mathbf{x}'_i;\mathbf{x}'_j])+\mathbf{M}\mathbf{L}\mathbf{P}_{\Phi}([\mathbf{x}'_j;\mathbf{x}'_i])}{2}\right)
$$  

$$
\operatorname*{min}_{\mathbf{X}^{\prime},\Phi}\operatorname{E}_{\boldsymbol{\theta}_{0}\sim P_{\theta_{0}}}\left[\sum_{t=0}^{T-1}D\left(\nabla_{\boldsymbol{\theta}}\mathcal{L}\left(\mathrm{GNN}_{\boldsymbol{\theta}},(g_{\boldsymbol{\Phi}}(\mathbf{X}^{\prime}),\mathbf{X}^{\prime}),\mathbf{Y}^{\prime}\right),\mathbf{\nabla}_{\boldsymbol{\theta}}\mathcal{L}\left(\mathrm{GNN}_{\boldsymbol{\theta}_{t}}(\mathbf{A},\mathbf{X}),\mathbf{Y}\right)\right)\right]
$$

<img src="imgs/gcond-algo.png" alt="drawing" width="600"/>


# Takeaway

- gradient matching loss as the condensation objective
- strategy of parameterizing the condensed features as free parameters and model the synthetic graph structure as a function of features, which takes advantage of the implicit relationship between structure and node features, consumes less number of parameters and offers better performance

Training

- **Alternating Optimization Schema.** Jointly optimizing X′ and Φ is often challenging as they are directly affecting each other  
- **Sparsification.** remove the entries whose values are smaller than a given threshold

Variant

- **A “Graphless” Model Variant.**

More 

- as suggested by previous works that reconstruct data from gradients (Zhu et al., 2019), **large batch size** tends to make reconstruction more difficult because more variables are involved during optimization
- sample a fixed-size set of neighbors on the original graph in each aggregation layer of GNNs and adopt a mini-batch training strategy because forward pass of GNNs involves the aggregation of enormous neighboring nodes i.e. **expensive**
- calculate the gradient matching loss for nodes from different classes separately to further **reduce memory usage and ease optimization**
- treating **A′ and X′ as independent** parameters overlooks the implicit correlations between graph structure and features
  
# Other references to follow

1. dataset distillation (Wang et al., 2018)
2. dataset condensation (Zhao et al., 2021) 
3. gradient matching scheme (Zhao et al., 2021)

baselines

1. graph coarsening method (Loukas, 2019; Huang et al., 2021b)
2. three coreset methods (Random, Herding (Welling, 2009)
3. K-Center (Farahani & Hekmatfar, 2009; Sener & Savarese, 2018))


# Results (Good or Bad)

- Approximate the original test accuracy by 95.3% on Reddit, 99.8% on Flickr and 99.0% on Citeseer, while reducing their graph size by more than 99.9%.
- consistently outperforms coarsening, coreset and dataset condensation baselines
- reliable correlation of performances between condensed dataset training and whole-dataset training