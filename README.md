# One-Class Active Learning
_A Julia package for One-Class Active Learning._

[![Build Status](https://travis-ci.com/englhardt/OneClassActiveLearning.jl.svg?branch=master)](https://travis-ci.com/englhardt/OneClassActiveLearning.jl)
[![Coverage Status](https://coveralls.io/repos/github/englhardt/OneClassActiveLearning.jl/badge.svg?branch=master)](https://coveralls.io/github/englhardt/OneClassActiveLearning.jl?branch=master)

This package implements active learning strategies for one-class learning.
The package has been developed as part of a benchmark suite for [active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) strategies for one-class classification. For more information about this research project, see the [OCAL project](https://www.ipd.kit.edu/ocal/) website, and the companion paper.

> Holger Trittenbach, Adrian Englhardt, Klemens Böhm, "An Overview and a Benchmark of Active Learning for One-Class Classification" [arXiv:1808.04759v2](https://arxiv.org/abs/1808.04759v2), 14 May 2019

## Installation
This package requires at least Julia 1.0.
This package is not registered yet. Please use the following command to add the package with Pkg3.
```Julia
using Pkg
Pkg.add("https://github.com/englhardt/OneClassActiveLearning.jl.git")
```

The results presented in the arXiv paper base on a previous version of the package and on Julia 0.6.
To reproduce the experiment results from the arXiv paper, use the old package manager (with Pkg.clone) and checkout OneClassActiveLearning.jl at tag `v0.1.0`.

## Overview
[One-class classifiers](https://en.wikipedia.org/wiki/One-class_classification) learn to identify if objects belong to a specific class, often used for outlier detection.
The package uses the one-class classifiers implemented in [SVDD.jl](https://github.com/englhardt/SVDD.jl).
The package implements a full active-learning cycle.
This includes the setup with data loading, splitting and setting up the initial labels.
Then it includes several active learning strategies for one-class learning.
All the results are stored in `.json` files.

The scripts in `example` gives a jump start on how to use this package.
To run a minimal example, execute:
```Julia
> julia example/example.jl
```
The result is then stored in `example/example_pool_qs.json`.

The following examples are available:
* `pool_qs`: The default example with a pool based query strategy
* `query_synthesis`: Use a query synthesis strategy optimized via particle swarm optimization
Run any of the above examples by supplying the name, e.g., `pool_qs` as command line argument:
```Julia
> julia example/example.jl pool_qs
```

### Data and initial pool init_strategies

The package offers multiple data splitting strategies.
- *(Sh)* Split holdout: Model fitting and query selection on the training split, and testing on a distinct holdout sample.
- *(Sf)* Split full: Model fitting, query selection and testing on the full data set.
- *(Si)* Split inlier: Like Sf, but model fitting on labeled inliers only.
- *(Sl)* Split labels: Like Sf, but model fitting on labeled observations (inliers and outliers) only.

The package includes multiple strategies to initialize the initial pool before starting the active learning:
- *(Pu)* Pool unlabeled: All observations are unlabeled.
- *(Pp)* Pool percentage: Stratified proportion of labels for p percent of the observations.
- *(Pn)* Pool number: Stratified proportion of labels for a fixed number of observations.
- *(Pnin)* Pool number inliers: A fixed number of labeled inliers.
- *(Pa)* Pool attributes: As many labeled inliers as number of attributes.

### Active learning strategies
This is a list of the available active learning strategies:
- Data-based query strategies
  - MinimumMarginPQs and ExpectedMinimumMarginPQs [1]
  - ExpectedMaximumEntropyPQs [1]
  - MinimumLossPQs [2]
- Model-based query strategies
    - HighConfidencePQs [3]
    - DecisionBoundaryPQs
- Hybrid query strategies
    - NeighborhoodBasedPQs [4]
    - BoundaryNeighborCombination [5]
- Baselines
  - RandomPQs
  - RandomOutlierPQs

#### Extending with a new strategy

A new query strategy `NewPQs` can be implemented with the following steps:

1. First create a new type `NewPQs` and implement:
```Julia
qs_score(qs::NewPQs, data::Array{T, 2}, pools::Dict{Symbol, Vector{Int}}) where T <: Real
```
Here, `data` is a subset of the full data set which depends on the data splitting strategy.
The current labels of the active learning cycle are in `pools` (`:U` for a unlabeled, `:Lin` for labeled a inlier and `:Lout` for labeled a outlier observation).
The argument `pools` contains a dictionary mapping from the different labels to indices in `data`.
The output of the function must be a score array, where the score at index `i` belongs to observation `i` in `data`.
The framework then chooses the observation with the highest score that is still unlabeled.

2. Add a call to the constructor of `NewPQs` in `initialize_qs` (`src/QueryStrategies/qs_utils.jl`).

3. Export `NewPQs` in `src/QueryStrategies/QueryStrategies.jl`.


## Authors
We welcome contributions and bug reports.

This package is developed and maintained by [Holger Trittenbach](https://github.com/holtri/) and [Adrian Englhardt](https://github.com/englhardt).

## References
[1] A. Ghasemi, H. R. Rabiee, M. Fadaee, M. T. Manzuri, and M. H. Rohban. Active learning from positive and unlabeled data. In 2011 IEEE 11th International Conference on Data Mining Workshops, pages 244–250, Dec 2011.

[2] A. Ghasemi, M. T. Manzuri, H. R. Rabiee, M. H. Rohban, and S. Haghiri. Active one-class learning by kernel density estimation. In 2011 IEEE International Workshop on Machine Learning for Signal Processing, pages 1–6, Sept 2011.

[3] V. Barnabé-Lortie, C. Bellinger, and N. Japkowicz. Active learning for one-class classification. In 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA), pages 390–395, Dec 2015.

[4] N. Görnitz, M. Kloft, K. Rieck, and U. Brefeld. Toward supervised anomaly detection. Journal of Artificial Intelligence Research (JAIR), pages 235–262, Jan. 2013.

[5] L. Yin, H. Wang, and W. Fan. Active learning based support vector data description method for robust novelty detection. Knowledge-Based Systems, pages 40–52, Aug. 2018.
