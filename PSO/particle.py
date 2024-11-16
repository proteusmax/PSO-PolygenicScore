import numpy as np
import os
from joblib import Parallel, delayed
import dask.array as da

class Particle:
    """
    Class for representing a particle from a Swarm

    Parameters:
    - objective_function (str): optimization (minimization) problem
    - population (Population instance): population with genetic individual level data (e.g., cohort study)
    """
    def __init__(self, objective_function, population, p_exploration):
        self.__obj_func_singleton = objective_function
        self.population = population
        self.nvar = objective_function.get_nvar()
        self.x = da.empty(self.nvar, chunks='auto')
        self.velocity = da.empty(self.nvar, chunks='auto')
        self.objective_value = None
        self.p_exploration = p_exploration
    
    def get_x(self):
        return self.x
    
    def set_x(self, x):
        self.x = np.array(x, copy=True)
    
    def get_x_at(self, i):
        return self.x[i]

    def set_x_at(self, value, i):
        xmin = self.__obj_func_singleton.get_xmin()[i]
        xmax = self.__obj_func_singleton.get_xmax()[i]

        value = np.nan_to_num(value, nan=0.0)
        self.x[i] = np.clip(value, xmin, xmax)

    def get_velocity(self):
        return self.velocity
    
    def set_velocity(self, velocity):
        self.velocity = np.array(velocity, copy=True)
    
    def get_velocity_at(self, i):
        return self.velocity[i]
    
    def set_velocity_at(self, value, i):
        self.velocity[i] = value
    
    def get_objective_value(self):
        return self.objective_value
    
    def set_objective_value(self, objective_value):
        self.objective_value = objective_value
    
    def evaluate_objective_function(self):
        x_computed = self.x.compute()
        loss = self.population.evaluate_fitness(beta_vector=x_computed, index=self.population.train)
        self.set_objective_value(loss)
        return self.objective_value
    
    def initialize_location(self, value=None, beta_vector_gwas=None):
        nvar = self.__obj_func_singleton.get_nvar()
        if value is None:
            self.x = np.empty(nvar)
            xmin = self.__obj_func_singleton.get_xmin()
            xmax = self.__obj_func_singleton.get_xmax()
            
            mask = da.random.random(self.nvar, chunks='auto') < self.p_exploration
            random_values = xmin + da.random.random(self.nvar, chunks='auto') * (xmax - xmin)
            self.x = da.where(mask, beta_vector_gwas, random_values)
            self.objective_value = self.evaluate_objective_function()
            
        else:
            self.x = da.full(self.nvar, value, chunks='auto')
            self.objective_value = value
    
class ParticleFactory:
    def __init__(self, obj_func_singleton, population, p_exploration):
        self.__obj_func_singleton = obj_func_singleton
        self.population = population
        self.p_exploration = p_exploration

    def create_particle(self):
        return Particle(self.__obj_func_singleton, self.population, self.p_exploration)