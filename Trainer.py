import numpy as np
import tqdm
from collections import defaultdict
from joblib import Parallel, delayed


class Trainer:
    def __init__(
        self,
        parser,
        genepool,
        genetic_operation_sequence,
        num_generations=100,
        fitness_propagation="naive",
        epsilon=0.01,
        phi=5,
        fitness_threshold=1,
        mode="weighted",
        batch_size=2,
        stimulus=1,
        reform_every=1,
        tinker_subtrees=True,
        print_info_every=10,
        one_out_every=0,
        eval_every=10,
        apply_every=1
  ):
        """
        :param parser:
        :param genepool:
        :param genetic_operation_sequence:
        :param mutation_rate: Rate to mutate
        :param arm_rate: Rate to grow and lose arms
        :param num_generations: Number of generations to train
        :param elitism:
        :param fitness_propagation:
        :param epsilon: How much to dampen fitness propagation fitness = fitness / parent_count + epsilon * phi
        :param phi: Multiplies epsilon
        :param mode: in {weighted, random} whether to weight application of genetic operations
        :param batch_size: Size of batches
        :param stimulus: How much to reward a complete parse
        :param reform: Whether to reform sentence level CFG at each step
        :param tinker_subtrees:
        """
        self.stimulus = stimulus
        self.tinker_subtrees = tinker_subtrees
        self.genetic_operation_sequence = genetic_operation_sequence
        self.parser = parser
        self.genepool = genepool
        self.fitness_threshold = fitness_threshold
        self.batch_size = batch_size
        self.corpus = genepool.corpus
        self.c = genepool.c
        self.unbatched_train_set = self.c.unbatched_train_set
        self.train_set = self.c.train_set
        self.dev_set = self.c.dev_set
        self.test_set = self.c.test_set
        self.num_generations = num_generations
        self.embedding_depth = genepool.embedding_depth
        self.fitness_propagation = fitness_propagation
        self.epsilon = epsilon
        self.phi = phi
        self.mode = mode
        self.stimulus = stimulus  # How much to reward a complete parse
        self.reform_every = (
            reform_every  # Whether to reform sentence layer after each generation
        )
        self.print_info_every = print_info_every
        self.one_out_every = one_out_every
        self.eval_every = eval_every
        self.apply_every = apply_every
        self.best_acc = 0

    def training_loop(self):
        """
        Runs the genetic algorithm on the population.
        :param print_gen_info: Whether to print information after each 10 generations
        :return: max fitness and max proportion of parsed examples
        """
        max_fitnesses = []
        max_fully_parsed_train = []
        # max_fully_parsed_dev = []
        max_fully_parsed_test = []
        for i in tqdm.tqdm(range(self.num_generations), desc="Generations"):
            sim_vec_generation = defaultdict(list)
            # Dictionary whose keys are the PSRs being evaluated and whose values are a list of lists of its fitness values on each batch.
            # These lists are then concatenated into one np array for processing.
            for j, batch in enumerate(self.train_set):
                self.training_step(batch, sim_vec_generation)
            for key in sim_vec_generation.keys():
                # print(key, sim_vec_generation[key])
                sim_vec_generation[key] = np.array(sim_vec_generation[key])
            # self.get_info(i)
            # start_time = time.time()
            if self.check_for_generation(i, self.eval_every):
                max_fitness = max(
                    [psr.weight for psr in self.genepool.recursive_embedding[-1]]
                )
                eval_train = self.evaluate(self.unbatched_train_set)
                # eval_dev = self.evaluate(self.dev_set)
                eval_test = self.evaluate(self.test_set)
                max_proportion_fully_parsed_train = max(eval_train)
                # max_proportion_fully_parsed_dev = max(eval_dev)
                max_proportion_fully_parsed_test = max(eval_test)
                # if max_proportion_fully_parsed_train >= self.fitness_threshold:
                #     return max_fitnesses, max_fully_parsed
                # The above is commented out so that all tests had consistent length
                max_fitnesses.append(max_fitness)
                max_fully_parsed_train.append(max_proportion_fully_parsed_train)
                max_fully_parsed_test.append(max_proportion_fully_parsed_test)
                # max_fully_parsed_dev.append(max_proportion_fully_parsed_dev)
            if self.check_for_generation(i, self.print_info_every):
                self.get_info(i)
                if self.check_for_generation(i, self.eval_every):
                    print(f"Max fitness: {max_fitness}")
                    print(f"Max proportion fully parsed train: {max_proportion_fully_parsed_train}")
            if self.check_for_generation(i, self.apply_every):
                self.apply_genetic_operations()
            # print("time to apply genops:", time.time() - start_time)
            # start_time = time.time()
            if self.check_for_generation(i, self.reform_every):
                self.genepool.reform_layer(
                    self.embedding_depth,
                    sim_vec_generation,
                    weighting=True,
                )
            if self.check_for_generation(i, self.one_out_every):
                self.one_out_fitness()
            # self.apply_genetic_operations()
            # print("time to reform:", time.time() - start_time)
            # self.apply_genetic_operations(similarity_vectors)

        return max_fitnesses, max_fully_parsed_train, max_fully_parsed_test

    def training_step(self, batch, sim_vec_generation):
        (
            sentence_fitness,
            similarity_vectors,
        ) = self.calculate_generation_fitness(batch)
        # print(len(sentence_fitness))
        # print(sentence_fitness.values())
        # print("-" * 10)
        # print([len(psr.rhs) for psr in self.genepool.recursive_embedding[-1]])
        # print(f"time_elapsed, BATCH {i}: {time.time() - start_time}")
        for key in similarity_vectors.keys():
            # Concatenate batch performances for overall performance
            sim_vec_generation[key] += similarity_vectors[key]

        # if not sim_vec_generation:
        #     sim_vec_generation = similarity_vectors
        # else:
        #     for key in similarity_vectors.keys():
        #         sim_vec_generation[key] += similarity_vectors[key]
        # print([psr.weight for psr in self.genepool.recursive_embedding[-1]])
        self.assign_phrase_fitnesses(sentence_fitness, -1)

    def check_for_generation(self, i, every):
        try:
            if i % every == 0:
                return True
        except ZeroDivisionError:
            return False
        return False

    def one_out_fitness(self):
        """
        replaces all fitness values in the sentence layer with 1
        :return: None
        """
        for psr in self.genepool.recursive_embedding[-1]:
            psr.weight = 1

    def evaluate(self, data):
        """
        Evaluates CFGs against the corpus
        :return: list where each entry is the sum of parsed examples in the dataset for each CFG
        """
        # non-parallelized version for time comparison
        # start = time.time()
        # performance = [
        #     sum(
        #         self.parser.parse_corpus(
        #             bundle_psr, self.corpus, parser_mode="eval", tinker_subtrees=False
        #         )
        #     )
        #     / len(self.corpus)
        #     for bundle_psr in self.genepool.recursive_embedding[-1]
        # ]
        # print(f"Regular time: {time.time()-start}")
        # start = time.time()
        performance = Parallel(n_jobs=8)(delayed(self.parser.parse_corpus)(
                    bundle_psr, data, parser_mode="eval", tinker_subtrees=False
                )
            for bundle_psr in self.genepool.recursive_embedding[-1]
            )
        performance = np.array(performance)
        # print("num cfgs:", len(self.genepool.recursive_embedding[-1]))
        # print("num data", len(data))
        # print("perf", len(performance), performance)
        # print("dat", len(data), data)
        performance = np.sum(performance, axis=1) / len(data)
        # print(f"Parallelized time: {time.time()-start}")
        return performance


    def assign_phrase_fitnesses(self, fitness, layer_index):
        """
        :param fitness: fitness values for the generation
        :param layer_index: layer on which to start propagation
        :return:
        """

        for psr_bundle in self.genepool.recursive_embedding[layer_index]:
            # psr_bundle.weight += fitness[psr_bundle]  # TEST
            if self.fitness_propagation == "none":
                psr_bundle.weight += fitness[psr_bundle]
            else:
                psr_bundle.prop(
                    self.fitness_propagation,
                    psr_bundle.embed_depth,
                    0,
                    fitness=fitness[psr_bundle],
                    epsilon=self.epsilon,
                    phi=self.phi,
                )

    def calculate_generation_fitness(self, batch):
        """
        :param batch: batch on which to calculate fitness
        :return: list of fitness of each CFG in the population
        """
        # start = time.time()  # Non-parallelized code for comparison
        # similarity_vectors = {
        #     bundle_psr: self.parser.parse_corpus(
        #         bundle_psr,
        #         batch,
        #         parser_mode="proportion",0
        #         tinker_subtrees=self.tinker_subtrees,
        #         stimulus=self.stimulus,
        #     )
        #     for bundle_psr in self.genepool.recursive_embedding[-1]
        # }
        # print(f"regular fitness calc time {time.time()-start}")
        # start = time.time()
        # vec = Parallel(n_jobs=8)(
        #     delayed(self.parse)(
        #         example,
        #         nonterminals,
        #         num_rules,
        #         rule_indexing,
        #         symbol_mapping,
        #         num_sentence_rules,
        #         parser_mode,
        #         weight_mapping,
        #         symb_parents,
        #         tinker_subtrees=tinker_subtrees,
        #         stimulus=stimulus,
        #     )
        #     for example in batch
        # )
        # similarity_vectors = dict(Parallel(n_jobs=8)(delayed((bundle_psr, self.parser.parse_corpus))[(bundle_psr, self.parser.parse_corpus(
        #             bundle_psr,
        #             batch,
        #             parser_mode="proportion",
        #             tinker_subtrees=self.tinker_subtrees,
        #             stimulus=self.stimulus,
        #         ))
        #         for bundle_psr in self.genepool.recursive_embedding[-1]])

        vals = Parallel(n_jobs=10)(delayed(self.parser.parse_corpus)(
            bundle_psr,
            batch,
            parser_mode="proportion",
            tinker_subtrees=self.tinker_subtrees,
            stimulus=self.stimulus,
            )
            for bundle_psr in self.genepool.recursive_embedding[-1]
        )

        similarity_vectors = dict(zip(self.genepool.recursive_embedding[-1], vals))
        # print(f"parallelize fitness calc time {time.time()-start}")

        n = len(batch)
        # fitness = (
        #     {  # The (stimulus + average length of partial parse for each bundle_psr,
        #         bundle_psr: sum(similarity_vectors[bundle_psr])
        #         / (n + len(bundle_psr.rhs))  # (n + len(bundle_psr.rhs))
        #         for bundle_psr in self.genepool.recursive_embedding[-1]
        #     }
        # )
        fitness = (
            {  # The (stimulus + average length of partial parse for each bundle_psr,
                bundle_psr: sum(similarity_vectors[bundle_psr])
                / (n + len(bundle_psr.rhs))  # (n + len(bundle_psr.rhs))
                for bundle_psr in self.genepool.recursive_embedding[-1]
            }
        )

        # print(fitness.values());
        return fitness, similarity_vectors  # TODO: return one dictionary instead

    def apply_genetic_operations(self):
        """
        :return: Applies genetic operations in desired order
        """
        max_embed_target = self.embedding_depth
        for depth in range(1, max_embed_target):
            for operation in self.genetic_operation_sequence:
                if operation == "mutate":
                    self.genepool.apply_to_tier(
                        self.genepool.mutate,
                        depth,
                        self.mode,
                    )
                elif operation == "lose_random_arms":
                    self.genepool.apply_to_tier(
                        self.genepool.lose_random_arm,
                        depth,
                        self.mode,
                    )
                elif operation == "grow_random_arms":
                    self.genepool.apply_to_tier(
                        self.genepool.grow_random_arm,
                        depth,
                        self.mode,
                    )
                elif operation == "elitism" and depth != max_embed_target:
                    self.genepool.elitism(depth)

                elif operation == "clear_unused" and depth != max_embed_target:
                    self.genepool.clear_unused_rules(depth)

    def get_info(self, i):
        """
        Prints salient information for generation i
        :param i: Generation for which to print information
        :return:
        """
        print(f"Generation {i}")
        for j, layer in enumerate(self.genepool.recursive_embedding):
            # print(f"layer {j} elements", [psr.rhs for psr in layer])
            print(f"layer {j} weights", [psr.weight for psr in layer])
            print(
                f"layer {j} average weight",
                sum([psr.weight for psr in layer]) / len(layer),
            )
            print(layer)
            print(f"layer {j} parent counts", [psr.parent_count for psr in layer])
            print(f"layer {j} embed depth", [psr.embed_depth for psr in layer])
            print(f"layer {j} keys", [psr.key for psr in layer])
            print(f"layer {j} size", len(layer))
            print(f"layer {j} rhs sizes", [len(psr.rhs) for psr in layer])
            print("-" * 30)
