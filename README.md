# Clustering-Hidden-Markov-Models-with-Variational-Bayesian-Hierarchical-EM
The hidden Markov model (HMM) is a broadly applied generative model for representing time series data, and clustering HMMs attracts increased interests from machine learning researchers.  However, the number of clusters (K) and the number of hidden states (S) for cluster centers are still difficult to determine. In this paper, we propose a novel HMM-based clustering algorithm, the variational Bayesian hierarchical EM algorithm, which clusters HMMs through their densities and priors, and simultaneously learns posteriors for the novel HMM cluster centers that compactly represent the structure of each cluster. The numbers K and S are automatically determined  in two ways. First, we place a prior on the pair (K,S) and approximate their posterior probabilities, from which the values with the maximum posterior are selected. Second, some clusters and states are pruned out implicitly when no data samples are assigned to them, thereby leading to automatic selection of the model complexity.

## Code Implementation
This toolbox contains the main function of VBHEM-H3M and is based on Matlab . There includes

* setup : setup the path for the toolbox.
* src: 
  * vbhem        : VBHEM algorithm.
  * hmm          : VBEM for learning HMMs.
  * compare_mtds : the comparison methods used in the paper, CCFD, VHEM, DIC, and PPK.
  * plots        : for ploting the results.
  * util         : other codes.

* demo : an example of clustering eye movement data using VBHEM.
* Synthetic_experiment : an example for clustering HMMs and comparing with other methods.


## Note
The latest version of VBHEM code will be released on the website of [VISAL lab](http://visal.cs.cityu.edu.hk/research/emhmm/ "悬停显示").

