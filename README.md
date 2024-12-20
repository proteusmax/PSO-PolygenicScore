## Metaheuristics for Polygenic Risk Scores (PRS)
*Methods for Optimizing the Effect Size (Beta) of Single Nucleotide Polymorphisms (SNP) in a PRS prediction*

> This project explores the use of Differential Evolution (DE) and Particle Swarm Optimization (PSO) for solving binary classification problems in the context of a disease. The implementation includes custom configurations and visualization tools to enhance the optimization process.

---

## Table of Contents

- [Metaheuristics for Polygenic Risk Scores (PRS)](#metaheuristics-for-polygenic-risk-scores-prs)
    - [Methods for Optimizing the Effect Size (Beta) of SNPs](#methods-for-optimizing-the-effect-size-beta-of-snps)
- [Dependencies](#dependencies)
    - [Core Libraries](#core-libraries)
    - [Additional Modules and Files](#additional-modules-and-files)
    - [Project-Specific Modules](#project-specific-modules)
    - [Configuration Parser](#configuration-parser)
    - [Installation](#installation)
- [Use](#use)
- [Initialization](#initialization)

---

## Dependencies

This project requires the following Python libraries and custom modules:

### Core Libraries

- **NumPy** (`numpy`): For numerical computations.
- **Pandas** (`pandas`): For data manipulation and analysis.
- **Pandas_plink** (`pandas_plink`): For reading and handling plink binary files.
- **Pymoo** (`pymoo`): For the metaheuristics implementation.
- **Dask** (`dask`): For parallel computing and handling large datasets.
- **ThreadPoolExecutor** (`from concurrent.futures import ThreadPoolExecutor`): For concurrent and parallel execution of tasks.
- **Random** (`random`): For generating random numbers.
- **Sklearn** (`sklearn`): For stratified splitting and ROC curves.
- **Matplotlib** (`matplotlib.pyplot`): For plotting and visualizing results.
- **Statsmodels** (`statsmodels`): For statistical modeling and analysis, including GLM and OLS.

### Additional Modules and Files

- **ABC Meta** (`abc`): Provides abstract base class functionality, used to define abstract classes and methods.

### Project-Specific Modules

The following are custom modules included in this project. Ensure these files are in the project directory:

- **GeneticFunctions** (`GeneticFunctions`): Contains utility functions used throughout the project.
- **Problems** (`Problems`): Defines the problem(s) to be solved by the optimization algorithms.
- **Population** (`Population`): Contains the Population class (from a cohort study) and customPSO and customSampling classes.

### Configuration Parser

- **ConfigParser** (`configparser`): Used to handle configuration files, making it easier to set parameters and other configurations.

### Installation

To install the main dependencies, run:
```bash
pip install numpy pandas pandas-plink pymoo dask scikit-learn matplotlib statsmodels 
```
---
## Use
```bash
python create_population.py <plink_prefix> <pheno_file> <sumstat_file> <trait_column> <config_file> <optimizer> [<iter>]"
```
- `plink_prefix`: The path prefix to the PLINK binary fileset. This is equivalent to the prefix used with PLINK's `--bfile` option.
  - For a PLINK fileset containing _data/UKBB.bed_, _data/UKBB.bim_, and _data/UKBB.fam_, the `plink_prefix` would be _data/UKBB_.
- `pheno_file`: A tab separated file with either column "IID" or "SUBJECT_ID" which correspond to the fam IDs in the PLINK's fam file and the "TRAIT" (e.g. "T2D"). For binary diseases you can use either 1 and 0 or CASE and CONTROL, respectively.
- `sumstat_file`: Path to the GWAS summary statistics file (tsv file most likely). Must include rsid, chr, pos, a1, a0, beta and p columns, check that the columns name (after changing to lower case and removing spaces) are defined in  defined in [GeneticFunctions/column_mapping.json](./GeneticFunctions/column_mapping.json).
- `trait_column`: The column name in the phenotype file (`pheno_file`) that specifies the trait to be analyzed (e.g. "T2D")
- `config_file`: Path to the configuration file (.cfg), which defines parameters for the optimization process, population settings, and other options. There are some examples in [Inputs](./inputs).
- `optimizer`: The optimizer to use for the metaheuristics-based population creation. Either PSO or DE.
- `iter` (Optional): The number of iterations to run the optimizer. Defaults to 1 if not provided.

## Initialization
A population of particles or individuals (here we are NOT referring to the study cohort population) is initialized, with each particle’s position representing a weight vector _β_ for all SNPs. The initial weight vector _β₀_ is derived from prior knowledge obtained from a European Genome-Wide Association Study (GWAS), providing a starting point for each particle.
However, to encourage exploration, each component _βᵢ_ of the weight vector has a probability _p_shrinkage_ of being shrunk and a probability _p_aug_ of being increased:

```
βᵢ = 
   β₀ᵢ * U(1, 1.15), with probability p_aug
   β₀ᵢ * U(0, 1), with probability p_shrinkage
   β₀ᵢ, with complementary probability
```

where _β₀ᵢ_ represents the weight from the GWAS study for the i-th SNP. This initialization strategy combines prior knowledge with controlled randomness, promoting a balance between exploitation of known weights and exploration of new values in the search space.

Author: [David Torres](https://github.com/proteusmax)
