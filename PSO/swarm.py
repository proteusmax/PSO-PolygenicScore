import numpy as np
import os
from joblib import Parallel, delayed
from .particle import *

class Swarm:
    def __init__(self, swarm_size, particle_factory, beta_vector_gwas=None):
        self.swarm_size = swarm_size
        self.particle_factory = particle_factory
        self.beta_vector_gwas = beta_vector_gwas
        self.swarm = np.empty(swarm_size, dtype=Particle)

    def get_swarm_size(self):
        return self.swarm_size  
    
    def add_particle_at(self, index, particle):
        self.swarm[index] = particle
    
    def get_particle_at(self, index):
        return self.swarm[index]
    
    def initialize_lbest_swarm(self):  # uses all available CPU cores
        n_jobs = os.cpu_count() - 1
        def create_and_initialize_lbest_particle():
            particle = self.particle_factory.create_particle()
            particle.initialize_location(np.inf)
            return particle

        # Create lbest particles in parallel and add them to the lbest swarm
        particles = Parallel(n_jobs=n_jobs)(delayed(create_and_initialize_lbest_particle)() for _ in range(self.swarm_size))

        for i, particle in enumerate(particles):
            self.add_particle_at(i, particle)
    
    def initialize_swarm(self):  # uses all available CPU cores
        n_jobs = os.cpu_count() - 1
        def create_and_initialize_particle():
            particle = self.particle_factory.create_particle()
            particle.initialize_location(beta_vector_gwas = self.beta_vector_gwas)
            return particle

        # Create particles in parallel and add them to the swarm
        particles = Parallel(n_jobs=n_jobs)(delayed(create_and_initialize_particle)() for _ in range(self.swarm_size))

        for i, particle in enumerate(particles):
            self.add_particle_at(i, particle)