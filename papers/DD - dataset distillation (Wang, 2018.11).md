# Dataset Distillation

- [Dataset Distillation](#dataset-distillation)
- [What did the authors tried to accomplished?](#what-did-the-authors-tried-to-accomplished)
- [Key elements of the approach](#key-elements-of-the-approach)
  - [Optimise distilled data](#optimise-distilled-data)
  - [](#)
- [Results (Good or Bad)](#results-good-or-bad)
- [Other references to follow](#other-references-to-follow)
- [Takeaway](#takeaway)
- [openreview](#openreview)

**Keywords**:
- Dataset Distillation (DC)
- Dataset pruning, core-set construction, and instance selection
- Gradient-based hyperparameter optimization
- Understanding datasets


# What did the authors tried to accomplished?

**Main idea.**  compress the knowledge of an entire training dataset into a few **synthetic** training images & train a network to reach high performance with a **small number** of distilled images and **several** gradient descent steps.

**More** 
- adapting pre-trained models to new datasets and performing a malicious data-poisoning attack


**Previous problems.** 
- Dataset pruning, core-set construction, and instance selection. their “valuable” images have to be real  
  
**Motivation.** 
- network distillation (Hinton et al., 2015) - distill the knowledge of multiple networks into a single model
- Gradient-based hyperparameter optimization - focus more heavily on learning synthetic training data rather than tuning hyperparameters
- **Prior work**: Maclaurin et al., 2015
   
# Key elements of the approach

In short:

1. derive the **network weights** as a **differentiable function** of our **synthetic** training **data**
2. instead of **optimizing** the **network weights** for a particular training objective, we optimize the pixel values of our **distilled images**

**outlines**
1. main **optimization** algorithm for training a network with a **fixed initialization** with one gradient descent (GD) step
2. **initial weights** are **random** rather than fixed
3. linear network case - understand both the property and limitation (lower bound on the size of distilled data)
4. more than one gradient descent steps and more than one epoch (pass)
5. demonstrate how to obtain distilled images with different initialization distributions and learning objectives


## Optimise distilled data

From standard minibatch stochastic gradient descent 

$$
\begin{aligned}\theta_{t+1}&=\theta_t-\eta\nabla_{\theta_t}\ell(\textbf{x}_t,\theta_t),\end{aligned}
$$

(often takes tens of thousands or even millions of update steps to converge)

to 

$$
\theta_1=\theta_0-\tilde{\eta}\nabla_{\theta_0}\ell(\tilde{\textbf{x}},\theta_0)
$$

(derive the new **weights** $\theta\_1$ as **a function of distilled data** $\tilde{x}$) 

to 

$$
\tilde{\mathbf x}^*,\tilde\eta^*=\underset{\tilde{\mathbf x},\tilde\eta}{\operatorname{arg}\operatorname*{min}}\mathcal L(\tilde{\mathbf x},\tilde\eta;\theta_0)=\underset{\tilde{\mathbf x},\tilde\eta}{\operatorname{arg}\operatorname*{min}}\ell(\mathbf x,\theta_1)=\underset{\tilde{\mathbf x},\tilde\eta}{\operatorname{arg}\operatorname*{min}}\ell(\textbf{x},\theta_0-\tilde{\eta}\nabla_{\theta_0}\ell(\tilde{\textbf{x}},\theta_0))
$$


- **Aim**
  - learn a tiny set of synthetic distilled training data so that **a single GD step** like above using these learned synthetic data $\tilde{x}$ can greatly boost the performance on the real test set.
  - optimise $\theta_0$ same as optimise $\theta\_1$

## 

# Results (Good or Bad)

- a handful of distilled images can be used to train a model with a fixed initialization to achieve surprisingly high performance
- For networks pre-trained on other tasks, our method can find distilled images for **fast model fine-tuning**

- **E.g.** possible to compress 60, 000 MNIST training images into just 10 synthetic distilled images (one per class) and achieve close to original performance with only a few gradient descent steps

# Other references to follow


**More explanation**
- **Author**'s project site: https://www.tongzhouwang.info/dataset_distillation/

**openreview**
- ICLR2019 - https://openreview.net/forum?id=Sy4lojC9tm
- ICLR2020 - https://openreview.net/forum?id=ryxO3gBtPB

**papers**
- network distillation (Hinton et al., 2015)
- Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. "**Distilling the Knowledge** in a Neural Network", in NIPS Deep Learning Workshop 2014.
- Dougal Maclaurin, David Duvenaud, and Ryan Adams. "**Gradient-based hyperparameter optimization** through reversible learning", in ICML 2015.
- Antonio Torralba and Alexei A Efros. "Unbiased look at dataset bias", in CVPR 2011.
- Agata Lapedriza, Hamed Pirsiavash, Zoya Bylinskii, and Antonio Torralba. "Are all training examples equally valuable?", in arXiv preprint 2013.

- Dataset pruning, core-set construction, and instance selection

**More papers**

- ensemble learning (Radosavovic et al., 2018)
- model compression (Ba & Caruana, 2014; Romero et al., 2015; Howard et al., 2017)
- data-free knowledge distillation 
  - optimizes synthetic data samples, but with a different objective of matching **activation statistics** of a teacher model in knowledge distillation (Lopes et al., 2017)


# Takeaway

- extend our method to compressing **large-scale** visual datasets such as ImageNet and **other types** of data (e.g., audio and text)
- investigate other initialization strategies since **sensitive** to the **distribution of initializations**
- 

# openreview

**GD Steps and Epochs**
- Each step is associated with a different batch of distilled data. All steps are sequentially cycled over for #epochs times. We clarified this in Sec. 3.4.
