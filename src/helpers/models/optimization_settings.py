class OptimizationSettings:
    
    def __init__(
        self,
        initial_population_size,
        population_size,
        recombinations_per_iteration,
        iterations_count
    ) -> None:
        self.initial_population_size = initial_population_size
        self.population_size = population_size
        self.recombinations_per_iteration = recombinations_per_iteration
        self.iterations_count = iterations_count