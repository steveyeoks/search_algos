from algorithms.genetic_algo.pygad import GA


class GeneticAlgo:
    """
    Implement Genetic Algorithm using the pygad package.

    pygad documentation: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class

    Properties are used as inputs (__init__()) for the pygad package:
        - num_generations: Number of generations / iterations
        - num_parents_mating: Number of solutions / chromosomes to be selected as parents
        - fitness_func: User-defined fitness function
        - initial population: User-defined initial population. Used to start the generation with a custom initial pop.
        - sol_per_pop: Number of solutions within a population.
        - num_genes: Number of genes in the solution / chromosome.
                     E.g. Decision variable = (2, 4). The value of gene 2 of chromosome (2, 4) is 4.
        - gene_type: Assign a single data type to all genes or to each individual gene through a list/tuple/numpy.
        - init_range_low: The initial lower value of the random range from which the gene values in the initial
                          population are selected.
        - init_range_high: The initial upper value of the random range from which the gene values in the initial
                           population are selected.
        - random_mutation_min_val: The initial lower value of the random range from which the gene values in the initial
                                   population are selected.
        - random_mutation_max_val: The initial upper value of the random range from which the gene values in the initial
                                   population are selected.
        - parent_selection_type: The parent selection type.
                                 Types include:
                                    1) 'sss' - steady-state selection
                                    2) 'rws' - roulette wheel selection
                                    3) 'sus' - stochastic universal selection
                                    4) 'rank' - rank selection
                                    5) 'random' - random selection
                                    6) 'tournament' - tournament selection
        - keep_parents: Number of parents to keep in the current population. Value cannot be less than -1
                        Types include:
                            1) '-1' (default): Keep all parents in the next population.
                            2) '0': Keep no parents in the next population.
                            3) Greater than 0: Keeps the specified number of parents in the next population.
        - crossover_type: Type of crossover operation.
                          Types include:
                              1) single_point: for single-point crossover.
                              2) two_points: for two points crossover.
                              3) uniform: for uniform crossover.
                              4) scattered: for scattered crossover.
        - mutation_type: Type of mutation operation.
                         Types include:
                            1) random: for random mutation.
                            2) swap: for swap mutation.
                            3) inversion: for inversion mutation.
                            4) scramble: for scramble mutation.
                            5) adaptive: for adaptive mutation.
        - mutation_percent_genes: Percentage of genes to mutate. An integer value of 10 means that 10% of the genes
                                  will be mutated.
    """

    def __init__(self,
                 num_generations=100,
                 num_parents_mating=4,
                 fitness_func=None,
                 initial_population=None,
                 sol_per_pop=None,
                 num_genes=2,
                 gene_type=int,
                 init_range_low=1,
                 init_range_high=20,
                 random_mutation_min_val=1,
                 random_mutation_max_val=20,
                 parent_selection_type="sss",
                 keep_parents=-1,
                 crossover_type="single_point",
                 mutation_type="random",
                 mutation_percent_genes=50,
                 acceptable_score_threshold=0.3,
                 saturation_generation_level=15):
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.fitness_func = fitness_func
        self.initial_population = initial_population
        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.gene_type = gene_type
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.random_mutation_min_val = random_mutation_min_val
        self.random_mutation_max_val = random_mutation_max_val
        self.parent_selection_type = parent_selection_type
        self.keep_parents = keep_parents
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mutation_percent_genes = mutation_percent_genes
        acceptable_score_threshold_str = "reach_" + str(1/acceptable_score_threshold)
        # saturation_generation_level_str = "saturate_" + str(saturation_generation_level)
        # stop_criteria = [acceptable_score_threshold_str, saturation_generation_level_str]
        stop_criteria = [acceptable_score_threshold_str]
        self.genetic_instance = GA(num_generations=num_generations,
                                   num_parents_mating=num_parents_mating,
                                   fitness_func=fitness_func,
                                   initial_population=initial_population,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_genes,
                                   gene_type=gene_type,
                                   init_range_low=init_range_low,
                                   init_range_high=init_range_high,
                                   random_mutation_min_val=random_mutation_min_val,
                                   random_mutation_max_val=random_mutation_max_val,
                                   parent_selection_type=parent_selection_type,
                                   keep_parents=keep_parents,
                                   crossover_type=crossover_type,
                                   mutation_type=mutation_type,
                                   mutation_percent_genes=mutation_percent_genes,
                                   stop_criteria=stop_criteria)

    def run(self):
        """
        Run Genetic Algorithm

        :return:
        """
        self.genetic_instance.run()
        best_solution, best_eval_metric, index = self.genetic_instance.best_solution()

        return best_solution, best_eval_metric
