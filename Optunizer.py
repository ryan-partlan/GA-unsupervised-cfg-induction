# CURRENT EXPERIMENT: ONE CHILD CROSSOVER
import optuna
from Corpus import *
from Trainer import *
from GenePool import *
from Parser import *


corpus_directory = "gum/dep"
# genetic_operation_sequence = ["grow_random_arms", "lose_random_arms", "mutate"] experiment 1
genetic_operation_sequence = ["lose_random_arms", "grow_random_arms", "mutate"]
train_examples = 100
test_examples = 10
batch_size = 10
max_length = 8  # inclusive
min_length = 3  # exclusive
reduce_ambiguity = True
mode = "gum"
###################################
c = Corpus(
    corpus_directory,
    train_size=train_examples,
    test_size=test_examples,
    batch_size=batch_size,
    max_length=max_length,
    min_length=min_length,
    reduce_ambiguity=reduce_ambiguity,
    mode=mode,
)
num_generations = 100
chi = 1
fitness_threshold = 1  # Stops and retrieves objects when max proportion fully parsed surpasses or equals threshhold.
crossover_mode = "two_child"
mode = "weighted"  # in {random, weighted}, whether to weight application of each genetic operation
fitness_propagation = "dampened_fitness"
# in {naive_fitness,
# rhs_fitness,
# undampened_rhs_naive_fitness,
# dampened_fitness,
# rhs_dampened_fitness,
# extra_dampened_fitness}
distribution_name = (
    "equal"  # in {"more_primaries_and_secondaries", "equal", "gauss", "triangular"}
)
print_info_every = 0


fixed_pop_size = True
sample_rules = True
eval_every = 10
tinker_subtrees = False
embedding_depth = 3

def objective(trial):
    mutation_rate = trial.suggest_float('mutation_rate', 0.0001, 0.001)
    grow_arm_rate = trial.suggest_float('grow_arm_rate', 0.2, 0.5)
    lose_arm_rate = trial.suggest_float('lose_arm_rate', 0.1, 0.5)
    elite_rate = trial.suggest_float('elite_rate', 0.1, 0.5)
    epsilon = trial.suggest_int('epsilon', 1, 10)
    phi = trial.suggest_int('phi', 10, 50)
    stimulus = trial.suggest_int('stimulus', 1, 20)
    reform_every = trial.suggest_int('reform_every', 1, 3)
    apply_every = trial.suggest_int('apply_every', 1, 3)
    one_out_every = trial.suggest_int('one_out_every', 3, 10)
    cross_rate = trial.suggest_float('cross_rate', 0.1, 0.5)
    recursion_chance = trial.suggest_float('recursion_chance', 0.05, 0.4)

    g = GenePool(
        c,
        fixed_pop_size=fixed_pop_size,
        sample_rules=sample_rules,  # Whether to take primary rules from the dataset such that they have at least one case of existing.
        distribution_name=distribution_name,  # in {"more_primaries_and_secondaries", "equal", "gauss", "triangular"}
        embedding_depth=embedding_depth,
        num_sent_rules=10,
        crossover_mode=crossover_mode,
        grow_arm_rate=grow_arm_rate,
        lose_arm_rate=lose_arm_rate,
        mutation_rate=mutation_rate,
        recursion_chance=recursion_chance,
        cross_rate=cross_rate,
        elite_rate=elite_rate,
    )
    p = Parser(c, g.unaries, fitness_propagation, epsilon, phi)
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
        print_info_every=print_info_every,
        eval_every=eval_every,
        one_out_every=one_out_every,
        apply_every=apply_every
    )
    max_fitnesses, max_fully_parsed_train, max_fully_parsed_test = t.training_loop()
    obj = max(max_fully_parsed_train)
    print("Max fitness", max(max_fitnesses))
    print("Max train acc", max(max_fully_parsed_train))
    print("Max test acc", max(max_fully_parsed_test))
    return obj


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)