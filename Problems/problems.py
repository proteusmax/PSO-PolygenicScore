import numpy as np
from pymoo.core.problem import Problem

from abc import ABCMeta, abstractmethod
class ObjectiveFunction(metaclass=ABCMeta):
    def __init__(self, nvar):
        self.nvar = nvar
        self.xmin = np.empty(nvar)
        self.xmax = np.empty(nvar)
        self.set_xmin()
        self.set_xmax()
        self.penalty_factor = 1
        self.tolerance_factor = None
        
    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def set_xmin(self):
        pass
    
    @abstractmethod
    def set_xmax(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_nvar(self):
        return self.nvar

    def get_xmin(self):
        return self.xmin

    def get_xmin_at(self, index):
        return self.xmin[index]
    
    def get_xmax(self):
        return self.xmax
    
    def get_xmax_at(self, index):
        return self.xmax[index]


class sphere(ObjectiveFunction):        
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar):
            result = result + x[i] ** 2
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -5.0

    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 5.0
    
    def get_name(self):
        return sphere.__name__
   

class rastringin(ObjectiveFunction):     
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar):
            result = result + x[i]*x[i] - 10*np.cos(2*np.pi*x[i])
        result = result + 10*self.nvar
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -5.12
    
    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 5.12
    
    def get_name(self):
        return rastringin.__name__

class rosenbrock(ObjectiveFunction):       
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar - 1):
            result = result + 100*np.power(x[i + 1] - x[i]*x[i], 2) + np.power(1 - x[i], 2)
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -10.0
    
    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 10.0
    
    def get_name(self):
        return rosenbrock.__name__
    
class BinaryClassification(ObjectiveFunction):
    def evaluate(self, y_true, y_hat):
        loss = -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))
        return loss
    
    def set_xmin(self):
        self.xmin.fill(-1.0)
    
    def set_xmax(self):
        self.xmax.fill(1.0)
    
    def get_name(self):
        return BinaryClassification.__name__


class BinaryClassificationPymoo(Problem):
    def __init__(self, population):
        # Define the number of variables, objectives, and bounds
        super().__init__(
            n_var=population.matched.shape[0],  # Number of variables
            n_obj=1,  # Single-objective optimization
            n_constr=0,  # No constraints
            xl=-1.0,  # Lower bounds for variables
            xu=1.0   # Upper bounds for variables
        )
        self.population = population

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a 2D array where each row is an individual
        # out["F"] should store the fitness values (minimization problem)
        fitness_values = [self.population.evaluate_fitness(beta_vector, index=self.population.train) for beta_vector in x]
        out["F"] = np.array(fitness_values, dtype=np.float32)  # Store as float32

class FunctionFactory:
    function_dictionary = {
        sphere.__name__: lambda nvar: sphere(nvar),
        rastringin.__name__: lambda nvar: rastringin(nvar),
        rosenbrock.__name__: lambda nvar: rosenbrock(nvar),
        BinaryClassification.__name__: lambda nvar: BinaryClassification(nvar)
    }

    @classmethod
    def select_function(cls, function_name, nvar=None):
        if function_name in cls.function_dictionary:
            return cls.function_dictionary[function_name](nvar)
        else:
            raise ValueError(f"Objective function '{function_name}' is not defined in FunctionFactory.")