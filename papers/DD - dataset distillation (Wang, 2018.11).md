# Dataset Distillation

- [Dataset Distillation](#dataset-distillation)
- [What did the authors tried to accomplished?](#what-did-the-authors-tried-to-accomplished)
- [Key elements of the approach](#key-elements-of-the-approach)
  - [Optimise distilled data](#optimise-distilled-data)
  - [](#)
- [Results (Good or Bad)](#results-good-or-bad)
- [Other references to follow](#other-references-to-follow)
- [Takeaway](#takeaway)
- [More](#more)

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



## 

# Results (Good or Bad)

- a handful of distilled images can be used to train a model with a fixed initialization to achieve surprisingly high performance
- For networks pre-trained on other tasks, our method can find distilled images for **fast model fine-tuning**

- **E.g.** possible to compress 60, 000 MNIST training images into just 10 synthetic distilled images (one per class) and achieve close to original performance with only a few gradient descent steps

# Other references to follow

**papers**
- network distillation (Hinton et al., 2015)
- Dataset pruning, core-set construction, and instance selection
- gradient-based hyperparameter optimization 

**More papers**


- ensemble learning (Radosavovic et al., 2018)
- model compression (Ba & Caruana, 2014; Romero et al., 2015; Howard et al., 2017)
- data-free knowledge distillation 
  - optimizes synthetic data samples, but with a different objective of matching **activation statistics** of a teacher model in knowledge distillation (Lopes et al., 2017)


# Takeaway

- extend our method to compressing **large-scale** visual datasets such as ImageNet and **other types** of data (e.g., audio and text)
- investigate other initialization strategies since **sensitive** to the **distribution of initializations**
- 
# More

Template based on:
- Stanford CS230: Deep Learning | Autumn 2018 | Lecture 8 - Career Advice / Reading Research Papers

- openreview
- author's conference presentation
- youtube videos from other uni student
- reddit discussion
- twitter discussion
