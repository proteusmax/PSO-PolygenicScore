import random
import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin
from GeneticFunctions import *
import dask.array as da
from sklearn.model_selection import StratifiedShuffleSplit
from Problems import *
from PSO import *
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import differential_evolution
import configparser
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from concurrent.futures import ThreadPoolExecutor
from pymoo.algorithms.soo.nonconvex.pso import PSO


class Population:
    """
    Class for representing a population with genetic individual level data (e.g., cohort study)

    Parameters:
    - objective_function (str): name of the objective_function from Problems/problems.py
    - plink_prefix (str): Prefix path to the plink files (as you would use in 'plink --bfile')
    - pheno_path (str): Path to the phenotype TSV file (suggested columns FID \t IID \t $TRAIT \t SEX \t AGE).
    - gwas_path (str): Path to the sumstats TSV file.
    """

    def __init__(self, objective_function, plink_prefix, pheno_path, gwas_path, trait_column='TRAIT', config_file = "inputs/param_exp.cfg", optimizer=None):
        # genotype
        warnings.filterwarnings("ignore", category=FutureWarning) # Suppress FutureWarnings for read_plink_bin
        self.G = read_plink1_bin(f"{plink_prefix}.bed", f"{plink_prefix}.bim", f"{plink_prefix}.fam", verbose=False)
        self.G = self.G.fillna(0) # Remove NAs and fill them with the reference allele 
        
        self.info_snp = pd.DataFrame({'chr': self.G.chrom.values, 'pos': self.G.pos.values,
        'a0': self.G.a0.values, 'a1': self.G.a1.values, 'rsid': self.G.snp.values})
        
        # phenotype
        self.P = pd.read_csv(pheno_path, sep='\t')
        self.trait_column = trait_column

        # matched
        self.matched = snp_match(gwas_path,self.info_snp, match_min_prop=0.0005)
        threshold = 0.05 # 0.05 / 52000000
        initial_count = self.matched.shape[0]
        self.matched = self.matched[self.matched['p'] < threshold]
        final_count = self.matched.shape[0]
        removed_count = initial_count - final_count
        print(f"{removed_count} rows were removed due to the p_value threshold.")
        print(f"{final_count} SNPs passed the p_value threshold")
        self.ind_col = self.matched['_NUM_ID_']

        # train, test
        self.X = None
        self.y = None
        self.train = None
        self.test = None

        # Optimizer
        self.config_file = config_file
        if isinstance(objective_function, str):
            nvar = self.matched.shape[0]
            self.objective_function = problems.FunctionFactory.select_function(objective_function, nvar = nvar)
        else:
            raise ValueError("objective_function should be a string whose name is defined in Problems/problems.py")
        self.optimizer = optimizer

    def prepare_data(self, drop_other_phenos = True):
        """
        Sorts the Phenotype data according to the fam order (from self.G) and then prepares the phenotype 
        data by separating X (IDs, SEX, AGE) and y (TRAIT).
        """
        id_column = 'IID' if 'IID' in self.P.columns else 'SUBJECT_ID'
        self.P[id_column] = self.P[id_column].astype(str)

        self.P = self.P[self.P[id_column].isin(self.G.iid.values)]

        self.P[id_column] = pd.Categorical(self.P[id_column], categories=self.G.iid.values, ordered=True)
        self.P = self.P.sort_values(id_column).reset_index(drop=True)

        if drop_other_phenos:
            columns_to_keep = [id_column, self.trait_column]
            if 'SEX' in self.P.columns:
                columns_to_keep.append('SEX')
            if 'AGE' in self.P.columns:
                columns_to_keep.append('AGE')
            self.P = self.P[columns_to_keep]

        self.y = self.P[self.trait_column]
        self.X = self.P.drop(columns=[self.trait_column])

    def calculate_pgs(self, beta_vector, index=None):
        """Calculate Polygenic (Risk) Scores for the population"""
        temp_G = self.G.isel(sample=index, variant=self.ind_col) if index is not None else self.G.isel(variant=self.ind_col)

        if temp_G.shape[1] != len(beta_vector):
            raise ValueError(f"Incompatible dimensions for matrix-vector multiplication: "
                         f"genotype data has {temp_G.shape[1]} columns, "
                         f"but beta vector has {len(beta_vector)} elements.")
        
        data_array = temp_G.data
        result = da.dot(data_array, beta_vector.astype(np.float32)) if isinstance(data_array, da.Array) else np.dot(data_array, beta_vector.astype(np.float32))
        PGS_vector = result.compute() if isinstance(result, da.Array) else result

        if np.isnan(PGS_vector).any() or np.isinf(PGS_vector).any():
            print("Warning: PGS_vector contains NaN or inf values.")
            print("beta_vector min:", np.min(beta_vector))
            print("beta_vector max:", np.max(beta_vector))
    
        return np.asarray(PGS_vector, dtype=np.float32)

        
    def train_test_split(self, train_size=0.7, random_state=123):
        """
        Splits the data into training and testing sets, balancing for SEX, AGE, and TRAIT
        and returns the indices of the individuals in each set.

        Parameters:
        - train_size (float): Proportion of the dataset to include in the train split (default is 0.7).
        - random_state (int): Controls the shuffling applied to the data before applying the split.

        Returns:
        - tuple: train_idx, test_idx - indices for the training and testing sets.
        """
        stratify_columns = [self.y]
        if 'SEX' in self.X.columns:
            stratify_columns.append(self.X['SEX'])
        #if 'AGE' in self.X.columns:
        #    stratify_columns.append(pd.qcut(self.X['AGE'], q=4))

        stratify_labels = pd.Series(["_".join(map(str, vals)) for vals in zip(*stratify_columns)], index=self.X.index)

        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        train_idx, test_idx = next(sss.split(self.X, stratify_labels))
        
        train_idx = list(train_idx)  # Convert to list
        test_idx = list(test_idx)
        #train_idx = random.sample(train_idx, int(len(train_idx) * 0.3))
        #test_idx = random.sample(test_idx, int(len(test_idx) * 0.3))

        self.train = sorted(train_idx)
        self.test = sorted(test_idx)

    def evaluate_fitness(self, beta_vector, index=None, binary=True):
        """Calculate the loss function using the beta_vector from a given particle"""
        PGS_vector = self.calculate_pgs(beta_vector, index)
        y_filtered = self.y.iloc[index] if index is not None else self.y

        if binary:
            PGS_vector = sigmoid(PGS_vector)    
            epsilon = 1e-10
            PGS_vector = np.clip(PGS_vector, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)

        loss = self.objective_function.evaluate(y_filtered, PGS_vector)

        return np.float32(loss)

    def get_params(self):
        return "_".join([f"{key}:{value}" for key, value in self.params()])

    def load_config(self):
        """
        Load all parameters dynamically from the configuration file.
        """
        config = configparser.ConfigParser()
        config.read(self.config_file)

        self.params = {}

        if "Optimizer" in config:
            for key, value in config.items("Optimizer"):
                try:
                    # Try to infer the type (int, float, or leave as string)
                    if value.isdigit():
                        self.params[key] = int(value)
                    else:
                        self.params[key] = float(value) if "." in value else value
                except ValueError:
                    self.params[key] = value 
            print(f"Loaded optimizer parameters: {self.params}")
        else:
            print("No [Optimizer] section found in the configuration file.")
            

    def optimize_weights(self):
        """
        Use PyMoo's Differential Evolution (DE) to optimize the beta vector within the Population class.
        """
        class CustomSampling(Sampling):
            def __init__(self, population):
                """
                Custom sampling logic with probabilities for shrinkage and augmentation.

                Parameters:
                - population: Reference to the parent Population instance.
                """
                super().__init__()
                self.population = population
                # Retrieve p_shrinkage and p_aug from population's params
                self.p_shrinkage = population.params["p_shrinkage"]
                self.p_aug = population.params["p_aug"]

            def _generate_individual(self, n_var, xmin, xmax):
                """
                Generate a single individual with shrinkage and augmentation logic.

                Parameters:
                - n_var: Number of variables (dimensionality of the problem).
                - xmin: Lower bounds for the variables.
                - xmax: Upper bounds for the variables.

                Returns:
                - individual: A generated individual as a numpy array.
                """
                beta_vector_gwas = self.population.matched['beta'].values.astype(np.float32)

                # Random mask for shrinkage
                mask_shrinkage = da.random.random(n_var, chunks='auto').astype(np.float32) < self.p_shrinkage

                # Random scaling factors for shrinkage
                random_scaling_factors = da.random.random(beta_vector_gwas.shape, chunks='auto').astype(np.float32)

                # Apply shrinkage (scaled beta vector)
                scaled_beta_vector = beta_vector_gwas * random_scaling_factors

                # Random mask for augmentation
                mask_aug = da.random.random(n_var, chunks='auto').astype(np.float32) < self.p_aug

                # Augment by random values between 1 and 1.2
                random_aug_factors = da.random.uniform(1.0, 1.2, beta_vector_gwas.shape, chunks='auto').astype(np.float32)
                augmented_beta_vector = scaled_beta_vector * random_aug_factors

                # Combine scaled and augmented vectors based on masks
                individual = da.where(mask_aug, augmented_beta_vector, scaled_beta_vector)
                individual = da.where(mask_shrinkage, individual, beta_vector_gwas)

                individual = np.clip(individual.compute().astype(np.float32), xmin, xmax)

                return individual

            def _do(self, problem, n_samples, **kwargs):
                """
                Generate the initial population for the problem.

                Parameters:
                - problem: The PyMoo problem instance.
                - n_samples: Number of samples (population size).

                Returns:
                - Initial population (numpy array).
                """
                n_var = problem.n_var
                xmin = problem.xl  # Lower bounds
                xmax = problem.xu  # Upper bounds

                # Use multithreading to generate the population
                with ThreadPoolExecutor() as executor:
                    population = list(
                        executor.map(lambda _: self._generate_individual(n_var, xmin, xmax), range(n_samples))
                    )

                return np.array(population, dtype=np.float32)
        
        class CustomProblem(Problem):
            def __init__(self, population):
                # Initialize the problem with number of variables and bounds
                super().__init__(
                    n_var=population.matched.shape[0],  # Number of variables
                    n_obj=1,  # Single-objective optimization
                    n_constr=0,  # No constraints
                    xl=np.float32(-1.0),  # Lower bounds for variables
                    xu=np.float32(1.0)   # Upper bounds for variables
                )
                self.population = population  # Store reference to the Population instance

            def _evaluate(self, x, out, *args, **kwargs):
                # Evaluate fitness for each individual
                x = np.asarray(x, dtype=np.float32)
                fitness_values = [
                    self.population.evaluate_fitness(beta_vector, index=self.population.train)
                    for beta_vector in x
                ]
                out["F"] = np.array(fitness_values, dtype=np.float32)  # Store results as float32

        problem = CustomProblem(self)
        sampling = CustomSampling(self)
        
        if self.optimizer == "DE":
            algorithm = DE(
                pop_size=self.params["pop_size"],  # Population size
                sampling=sampling,  # Random initialization
                variant=self.params["variant"],  # DE strategy
                CR=self.params["cr"],  # Crossover probability
                F=self.params["f"],   # Differential mutation factor
            )
        elif self.optimizer == "PSO":
            algorithm = PSO(
                pop_size=self.params["pop_size"],  # Population size
                w=self.params["pop_size"],  # Inertia weight
                c1=self.params["c1"],  # Cognitive parameter
                c2=self.params["c2"],  # Social parameter
                sampling=sampling  # Custom sampling
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        result = minimize(
            problem,
            algorithm,
            termination=("n_gen", self.params["maxiter"]), 
            verbose=True, 
            n_jobs=-1, 
            save_history=True
        )

        self.best_solution = np.asarray(result.X, dtype=np.float32) 
        self.best_fitness = np.float32(result.F[0])  # Best fitness value
        self.history = result.history

        #print("Best solution dtype:", result.X.dtype)  # Should print float32
        #print("Best fitness dtype:", result.F.dtype)

    def return_auc(self, index=None, binary=True, adjusted=True, title='ROC Curve'):
        """Calculate the loss function using the beta_vector from a given particle"""

        PGS_vector = self.calculate_pgs(self.best_solution, index)
        y_filtered = self.y.iloc[index] if index is not None else self.y

        if adjusted:
            PGS_series = pd.Series(PGS_vector, index=self.X.index if index is None else self.X.index[index], name='PGS')
            X_filtered = self.X.iloc[index, 1:] if index is not None else self.X.iloc[:, 1:]
            X_with_PGS = pd.concat([PGS_series, X_filtered], axis=1)
            X = sm.add_constant(X_with_PGS)

        if binary:
            model = sm.GLM(y_filtered, X, family=sm.families.Binomial())
        else:
            model = sm.OLS(y_filtered, X)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")

            try:
                result = model.fit()
            except np.linalg.LinAlgError as e:
                print("Warning during model fitting:", e)
                return np.inf # Return a high loss if the model fails to fit
            
        y_hat = result.predict(X)
        epsilon = 1e-10
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon) # Clip predictions to avoid log(0)

        fpr, tpr, _ = roc_curve(y_filtered, y_hat)
        auc_score = roc_auc_score(y_filtered, y_hat)

        print(f"{title} AUC score: {auc_score:.2f}")
        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        params = self.get_params()
        fig_path = f'results/pso_{params}_{title.replace(" ", "_")}.png'
        plt.savefig(fig_path, format='png')
        print(f"Figured saved in {fig_path}")
        plt.close()

    def plot_convergence(self):
        generations = []
        fitness_values = []

        for i, entry in enumerate(self.history):
            generations.append(i + 1)
            fitness_values.append(entry.opt[0].F[0])
    
        plt.figure(figsize=(8, 5))
        plt.plot(generations, fitness_values, marker='o')
        plt.title("Convergence Plot")
        plt.xlabel("Generation")
        plt.ylabel("Binary Classification Loss")
        plt.grid(True)
        params = self.get_params()        
        fig_path = f'results/pso_{params}_convergence_plots.png'
        plt.savefig(fig_path, format='png')
        print(f"Figured saved in {fig_path}")
        plt.close()
     
    def run_experiment(self):
        if 'CASE' in self.P[self.trait_column].values and 'CONTROL' in self.P[self.trait_column].values:
            self.P[self.trait_column] = self.P[self.trait_column].replace({'CASE': 1, 'CONTROL': 0})
        print("Preparing data...")
        self.prepare_data()
        print("Data prepared successfully")
        self.train_test_split()
        print(f"Data splited into: {len(self.train)} individuals for training, {len(self.test)} individuals for testing")
        print("Reading config file...")
        self.load_config()
        print("Optimizing weights...")
        self.optimize_weights()
        print("Weights optimized successfully")
        self.return_auc(self.train, title = "ROC curve training")
        self.return_auc(self.test, title = "ROC curve testing")
        self.plot_convergence()
