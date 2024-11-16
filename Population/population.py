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


class Population:
    """
    Class for representing a population with genetic individual level data (e.g., cohort study)

    Parameters:
    - objective_function (str): name of the objective_function from Problems/problems.py
    - plink_prefix (str): Prefix path to the plink files (as you would use in 'plink --bfile')
    - pheno_path (str): Path to the phenotype TSV file (suggested columns FID \t IID \t $TRAIT \t SEX \t AGE).
    - gwas_path (str): Path to the sumstats TSV file.
    """

    def __init__(self, objective_function, plink_prefix, pheno_path, gwas_path, trait_column='TRAIT'):
        # genotype
        self.G = read_plink1_bin(f"{plink_prefix}.bed", f"{plink_prefix}.bim", f"{plink_prefix}.fam", verbose=False)
        self.G = self.G.chunk({'sample': 500, 'variant': 10000})
        self.G = self.G.fillna(0) # Remove NAs and fill them with the reference allele 
        
        # phenotype
        self.P = pd.read_csv(pheno_path, sep='\t')
        self.trait_column = trait_column

        # matched
        self.matched = snp_match(gwas_path,plink_prefix,from_file=True, match_min_prop=0.0005)
        threshold = 0.05 / 52000000
        initial_count = self.matched.shape[0]
        self.matched = self.matched[self.matched['p_value'] < threshold]
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
        if isinstance(objective_function, str):
            nvar = self.matched.shape[0]
            self.objective_function = problems.FunctionFactory.select_function(objective_function, nvar = nvar)
        else:
            raise ValueError("objective_function should be a string whose name is defined in Problems/problems.py")
        self.optimizer = None

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
        result = da.dot(data_array, beta_vector) if isinstance(data_array, da.Array) else np.dot(data_array, beta_vector)
        PGS_vector = result.compute() if isinstance(result, da.Array) else result

        if np.isnan(PGS_vector).any() or np.isinf(PGS_vector).any():
            print("Warning: PGS_vector contains NaN or inf values.")
            print("beta_vector min:", np.min(beta_vector))
            print("beta_vector max:", np.max(beta_vector))
    
        return PGS_vector
        
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
        train_idx = random.sample(train_idx, int(len(train_idx) * 0.02))
        test_idx = random.sample(test_idx, int(len(test_idx) * 0.02))

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

        return loss

    
    def optimize_weights(self, p_exploration=0.5):
        self.optimizer = PSO(self.objective_function.get_name(), population=self, beta_vector_gwas=self.matched['beta'], p_exploration=p_exploration)
        self.optimizer.run()

    def get_optimized_weights(self):
        return self.optimizer.gbest.get_x()
    
    def return_auc(self, index=None, binary=True, adjusted=True, title='ROC Curve'):
        """Calculate the loss function using the beta_vector from a given particle"""

        best_beta_vector = self.get_optimized_weights()
        PGS_vector = self.calculate_pgs(best_beta_vector, index)
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
        fig_path = f'pso_{title.replace(" ", "_")}.png'
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
        print(f"Data splited into: {self.train} individuals for training, {self.test} individuals for testing")
        print("Optimizing weights...")
        self.optimize_weights()
        print("Weights optimized successfully")
        self.return_auc(self.train, title = "ROC curve training")
        self.return_auc(self.test, title = "ROC curve testing")
