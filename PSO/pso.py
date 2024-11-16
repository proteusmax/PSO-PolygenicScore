import numpy as np
from PSO import *
import configparser
from Problems import *
from joblib import Parallel, delayed


class PSO:
    def __init__(self, objective_function, population, beta_vector_gwas, p_exploration, config_file = "inputs/param_swarm.cfg"):
        """
        p_exploration (float): 1 means taking all the beta_vector, 0 is generating random beta_vector
        """
        n_var = beta_vector_gwas.size
        self.beta_vector_gwas = beta_vector_gwas
        self.load_config(config_file)
        self.population = population
        self.objective_function = problems.FunctionFactory.select_function(objective_function, n_var)
        self.particle_factory = ParticleFactory(self.objective_function, self.population, p_exploration)      
        self.swarm = Swarm(self.swarm_size, self.particle_factory, self.beta_vector_gwas)
        self.lbest = Swarm(self.swarm_size, self.particle_factory)
        self.gbest = self.particle_factory.create_particle()
        # Initialization
        self.gbest.initialize_location(np.inf)
        self.swarm.initialize_swarm()
        self.lbest.initialize_lbest_swarm()
        self.generation_statistics = {}
        self.generation_t = 0
    
    def load_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.swarm_size = int(config['SwarmSettings']['swarm_size'])
        self.c1 = float(config['SwarmSettings']['c1'])
        self.c2 = float(config['SwarmSettings']['c2']) 
        self.w = float(config['SwarmSettings']['inertia_factor']) 
        self.Vmax = float(config['SwarmSettings']['Vmax'])
        self.max_generations = int(config['SwarmSettings']['max_generations'])

    def pass_next_generation(self):
        best_fitness =  self.gbest.get_objective_value()
        
        self.generation_t += 1

        self.generation_statistics[self.generation_t] = {
            "best_fitness": best_fitness
        }

    def get_population_statistics(self):
        """
        Returns dictionary with information about best individual across generations
        """
        return self.generation_statistics

def run(self):  #  uses all available processors
    n_jobs = os.cpu_count() - 1
    while self.generation_t < self.max_generations:
        print(f'Generation: {self.generation_t}/{self.max_generations}', end='\r')

        # Step 3a: Evaluate each particle in parallel and update positions and velocities
        def evaluate_particle(i):
            particle = self.swarm.get_particle_at(i)
            particle_lbest = self.lbest.get_particle_at(i)

            if particle.get_objective_value() is None:
                print(f"Warning: Particle {i} has None objective value!")
            if particle_lbest.get_objective_value() is None:
                print(f"Warning: Particle {i} lbest has None objective value!")

            # Set the personal best position
            if particle.get_objective_value() < particle_lbest.get_objective_value():
                particle_lbest.set_x(particle.get_x())
                particle_lbest.set_objective_value(particle.get_objective_value())
                
            # Update the gBest position
            if particle_lbest.get_objective_value() < self.gbest.get_objective_value():
                self.gbest.set_x(particle_lbest.get_x())
                self.gbest.set_objective_value(particle_lbest.get_objective_value())
                
            return particle, particle_lbest

        # Run the evaluation of particles in parallel
        results = Parallel(n_jobs=n_jobs)(delayed(evaluate_particle)(i) for i in range(self.swarm.get_swarm_size()))

        # Update the swarm and lbest after parallel processing
        for i, (particle, particle_lbest) in enumerate(results):
            self.swarm.add_particle_at(i, particle)
            self.lbest.add_particle_at(i, particle_lbest)

        # Step 3b: Update particle velocities and positions after evaluating fitness
        r1 = np.random.rand(self.objective_function.get_nvar())
        r2 = np.random.rand(self.objective_function.get_nvar())

        for i in range(self.swarm.get_swarm_size()):
            particle = self.swarm.get_particle_at(i)
            lbest = self.lbest.get_particle_at(i)

            particle_x = particle.get_x()
            lbest_x = lbest.get_x()
            gbest_x = self.gbest.get_x()
            particle_velocity = particle.get_velocity()

            cognitive_comp = np.nan_to_num(self.c1 * r1 * (lbest_x - particle_x), nan=0.0)
            social_comp = np.nan_to_num(self.c2 * r2 * (gbest_x - particle_x), nan=0.0)
            new_velocity = self.w * particle_velocity + cognitive_comp + social_comp

            new_velocity = np.clip(new_velocity, -self.Vmax, self.Vmax)
            new_position = np.nan_to_num(particle_x + new_velocity, nan=0.0)

            particle.set_velocity(new_velocity)
            particle.set_x(new_position)
            particle.evaluate_objective_function()

        self.pass_next_generation()