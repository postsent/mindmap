**Title:** Condensing Graphs via One-Step Gradient Matching

# What did the authors tried to accomplished?

Performs gradient matching for only one single step without training the network weights.  

Previously: GCOND. bi-level problem that is **computationally expensive** to optimize: they require multiple iterations (inner iterations) of updating neural network parameters before updating the synthetic data for multiple iterations (outer iterations) ; it does not produce **discrete graph structures** and its condensation process is **costly**;

- Improve GCOND with one step gradient update 
- GCOND does not produce discrete graph structures and its condensation process is costly. 
- Approximates the overall gradient matching loss for 𝜃𝑡 with the initial matching loss at the first epoch, which they term as one-step matching loss 

# Key elements of the approach

To produce **discrete values**, we model the graph structure as a probabilistic graph model and optimize the discrete structures in a differentiable manner

**Image setting**

$$
\begin{aligned}\min\limits_S\underset{\theta_0\sim P_{\theta_0}}{\mathbb{E}}\left[\sum\limits_{t=0}^{T-1}D\left(\nabla_\theta\ell\left(f_{\theta_t}(\mathcal{S}),\mathcal{Y}'\right),\nabla_\theta\ell\left(f_{\theta_t}(\mathcal{T}),\mathcal{Y})\right)\right)\right],\\ \text{s.t.}\theta_{t+1}=\text{opt}_\theta(\theta_t,\mathcal{S}).\end{aligned}
$$

**Graph setting**  


$$
\min_{\text{A}', \text{X}'}\underset{\theta_0\rightarrow P_{\theta_0}}{\mathbb{E}}\left[\sum_{t=0}^{T-1}D\left(\nabla_{\theta}\ell\left(f_{\theta_t}(\text{A}',\text{X}'),\mathcal{Y}'\right),\nabla_{\theta}\ell\left(f_{\theta_t}(\text{T}),\mathcal{Y}\right)\right)\right]\\ \text{s.t.}\theta_{t+1}=\operatorname{opt}_{\theta}(\theta_t,\text{S})
$$

to learn both graph structure A′ and node features X′, requires a function that outputs **binary values**.

**Model the graph structure as a probabilistic graph model with Bernoulli distribution**

$$
\operatorname*{min}_{\Omega,X^{\prime}}\underset{\theta\rightarrow P_{\theta_{0}}}{\mathbb{E}}\left[\underset{\alpha\mathrm{-Uniform}(0,1)}{\mathbb{E}}\left[\ell(\mathbf{A}^{\prime}(\Omega),\mathbf{X}^{\prime},\theta_{0})\right]\right] = 
$$

$$
\underset{\theta_{0}}{\mathbb{E}}\left[\underset{a}{\mathbb{E}}\left[\sum_{t=0}^{T-1}D\left(\nabla_{\theta}\ell\left(f_{\theta_{t}}(\mathbf{A}^{\prime}(\Omega),\mathbf{X}^{\prime}),\mathbf{Y}^{\prime}\right),\nabla_{\theta}\ell\left(f_{\theta_{t}}(\mathcal{T}),\mathbf{Y}^{\prime}\right)\right)\right]\right]
$$

$$
\text{s.t.}\theta_{t+1}=\text{opt}_{\theta}(\theta_t,\mathcal S).
$$

**One step gradient matching**

$$
\underset{\theta_{0}}{\mathbb{E}}\left[\underset{a}{\mathbb{E}}\left[D\left(\nabla_{\theta}\ell\left(f_{\theta_{t}}(\mathbf{A}^{\prime}(\Omega),\mathbf{X}^{\prime}),\mathbf{Y}^{\prime}\right),\nabla_{\theta}\ell\left(f_{\theta_{t}}(\mathcal{T}),\mathbf{Y}^{\prime}\right)\right)\right]\right]
$$

# Takeaway



# Other references to follow

# Results (Good or Bad)

- reduce the dataset size by 90% while approximating up to 98% of the original performance and our method is significantly faster than multi-step gradient matching (e.g. 15× in CIFAR10 for synthesizing 500 graphs).
- First, it significantly speeds up the condensation process while providing reasonable guidance for synthesizing condensed graphs. Second, it removes the burden of tuning hyper-parameters such as the number of outer/inner iterations of the bi-level optimization as required by DC