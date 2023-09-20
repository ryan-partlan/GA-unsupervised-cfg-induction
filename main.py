import random
from Trainer import *
from GenePool import *
from Parser import *
from Corpus import *
from Visualizer import *

if __name__ == "__main__":
    random.seed(42)
    ### CORPUS LOADING PARAMETERS ###
    # corpus_directory = "brown_small"
    corpus_directory = "gum/dep"
    train_examples = 300
    dev_examples = 0
    test_examples = train_examples // 10
    batch_size = 10
    max_length = 8  # inclusive
    min_length = 3  # exclusive
    reduce_ambiguity = True
    mode = "gum"
    toy = False
    ###################################
    c = Corpus(
        corpus_directory,
        train_size=train_examples,
        dev_size=dev_examples,
        test_size=test_examples,
        batch_size=batch_size,
        max_length=max_length,
        min_length=min_length,
        reduce_ambiguity=reduce_ambiguity,
        mode=mode,
    )
    print("Corpus loaded")

    ### GENE POOL LOADING PARAMETERS ###
    recursion_chance = 0.1565084487506348
    embedding_depth = 3
    fixed_pop_size = True
    sample_rules = True
    # Whether to take primary rules from the dataset such that they have at least one case of existing.
    distribution_name = (
        "equal"  # in {"more_primaries_and_secondaries", "equal", "gauss", "triangular"}
    )
    num_sent_rules = 10

    crossover_mode = "two_child"
    # crossover_mode = "one_child"
    #####################################

    # TODO: MAKE BEST PERFORMER COPY IMMUNE TO GENOPS?
    ### TRAINER LOADING PARAMETERS ###
    mutation_rate = 0.00023727741277585973
    grow_arm_rate = 0.4986264128179082  # rate at which arms will appear
    lose_arm_rate = 0.2823787086019406  # rate at which arms will be deleted.
    elite_rate = 0.31875627638955933  # Top elitism% kept, crossover employed on remaining pairs
    num_generations = 200
    fitness_threshold = 1  # Stops and retrieves objects when max proportion fully parsed surpasses or equals threshhold.
    mode = "weighted"  # in {random, weighted}, whether to weight application of each genetic operation
    fitness_propagation = "raw"  # in {none, raw, naive_fitness, rhs_fitness, undampened_rhs_naive_fitness, dampened_fitness, rhs_dampened_fitness, extra_dampened_fitness}
    epsilon = 6  # depreciation of fitness per layer of recursion
    phi = 40  # increase of epsilon per generation (multiplied in)
    stimulus = 14.0
    reform_every = 1  # After how many generations to reform (with crossover)
    print_info_every = 10
    apply_every = 1
    one_out_every = 5  # After how many generations to one out fitness of all layers (0 means don't)
    eval_every = 10
    tinker_subtrees = False
    lose_arms_comparative = True
    sampling = "bigram"

    #SEQUENCE ORDERING EXPERIMENTS
    genetic_operation_sequence = ["grow_random_arms", "lose_random_arms", "mutate"]  # exp 1
    # genetic_operation_sequence = ["lose_random_arms", "grow_random_arms", "mutate"]  # exp 2

    # genetic_operation_sequences = [["grow_random_arms"]]

    # genetic_operation_sequences = [["mutate", "lose_random_arms", "grow_random_arms"]]
    # genetic_operation_sequences = [["grow_random_arms", "lose_random_arms", "clear_unused", "mutate"]]

    # genetic_operation_sequences = [["grow_random_arms", "lose_random_arms", "mutate"]]
    # genetic_operation_sequences = [["grow_random_arms", "lose_random_arms", "mutate"]]

    # genetic_operation_sequences = [["lose_random_arms", "clear_unused", "mutate"]]

    # genetic_operation_sequences = [["clear_unused", "grow_random_arms", "lose_random_arms", "mutate"]]

    # genetic_operation_sequences = list(itertools.permutations(["mutate", "lose_random_arms", "grow_random_arms"], 3))
    #cross_rate = 0.8 #.6 best #.8 best? maybe
    cross_rate = 0.11135680159664396

    # genetic_operation_sequences = [["grow_random_arms", "mutate"]]
    ####################################
    g = GenePool(
        c,
        fixed_pop_size=fixed_pop_size,
        sample_rules=sample_rules,  # Whether to take primary rules from the dataset such that they have at least one case of existing.
        distribution_name=distribution_name,  # in {"more_primaries_and_secondaries", "equal", "gauss", "triangular"}
        embedding_depth=embedding_depth,
        num_sent_rules=num_sent_rules,
        crossover_mode=crossover_mode,
        grow_arm_rate=grow_arm_rate,
        lose_arm_rate=lose_arm_rate,
        mutation_rate=mutation_rate,
        recursion_chance=recursion_chance,
        cross_rate=cross_rate,
        elite_rate=elite_rate,
        sampling=sampling
    )
    print("Genepool loaded")

    p = Parser(c, g.unaries, fitness_propagation, phi, epsilon)
    print("Parser loaded")
    # for genetic_operation_sequence in genetic_operation_sequences:
    t = Trainer(
        p,
        g,
        genetic_operation_sequence,
        fitness_threshold=fitness_threshold,
        num_generations=num_generations,
        mode=mode,  # in {random, weighted}, whether to weight application of each genetic operation
        fitness_propagation=fitness_propagation,  # in {naive_fitness, dampened_fitness}
        epsilon=epsilon,  # depreciation of fitness per layer of recursion
        phi=phi,  # increase of epsilon per generation (multiplied in)
        stimulus=stimulus,
        reform_every=reform_every,  # Whether to reform top layer at each step
        tinker_subtrees=tinker_subtrees,  # Whether to adjust weights while parsing
        one_out_every=one_out_every,
        print_info_every=print_info_every,
        eval_every=eval_every,
        apply_every=apply_every
    )
    strt = time.time()
    max_fitnesses, max_fully_parsed_train, max_fully_parsed_test = t.training_loop()
    print(time.time() - strt)
    v = Visualizer(
        max_fitnesses[1:], max_fully_parsed_train[1:], f"{genetic_operation_sequence}"
    )
    print("fit", max(max_fitnesses))
    print("train", max(max_fully_parsed_train))
    print("test", max(max_fully_parsed_test))
    print(max_fitnesses)
    print(max_fully_parsed_train)
    print(max_fully_parsed_test)
    v.visualize()
