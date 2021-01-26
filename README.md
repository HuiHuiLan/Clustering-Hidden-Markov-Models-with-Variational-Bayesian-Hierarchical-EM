# Clustering-Hidden-Markov-Models-with-Variational-Bayesian-Hierarchical-EM
The hidden Markov model (HMM) is a broadly applied generative model for representing time series data, and clustering HMMs attracts increased interests from machine learning researchers.  However, the number of clusters ($K$) and the number of hidden states ($S$) for cluster centers are still difficult to determine. In this paper, we propose a novel HMM-based clustering algorithm, the variational Bayesian hierarchical EM algorithm, which clusters HMMs through their densities and priors, and simultaneously learns posteriors for the novel HMM cluster centers that compactly represent the structure of each cluster. The numbers $K$ and $S$ are automatically determined  in two ways. First, we place a prior on the pair $(K,S)$ and approximate their posterior probabilities, from which the values with the maximum posterior are selected. Second, some clusters and states are pruned out implicitly when no data samples are assigned to them, thereby leading to automatic selection of the model complexity.

## Code Implementation

## method

## hyhh'
