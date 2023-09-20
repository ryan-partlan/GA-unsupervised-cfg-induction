import numpy as np
import itertools
import math
import tqdm
from Distribution import *
from Initializer import *
from scipy.special import softmax

random.seed(42)


class GenePool:
    def __init__(
        self,
        corpus,
        fixed_pop_size=True,
        distribution_name="gauss",
        embedding_depth=2,
        num_sent_rules=100,
        sample_rules=True,
        cross_rate=0,
        mutation_rate=0.01,
        grow_arm_rate=0.3,
        lose_arm_rate=0.3,
        crossover_mode="two_child",
        recursion_chance=0.1,
        elite_rate=0.2,
        lose_arms_comparative=True,
        sampling="bigram"
    ):
        """
        :param corpus:
        :param distribution_name:
        :param embedding_depth:
        :param population_size:
        """
        self.cross_index = 0
        self.elite_rate = elite_rate
        self.mutation_rate = mutation_rate
        self.grow_arm_rate = grow_arm_rate
        self.lose_arm_rate = lose_arm_rate
        self.crossover_mode = crossover_mode
        self.recursion_chance = recursion_chance
        self.corpus = corpus.corpus
        self.c = corpus
        self.cross_rate = cross_rate
        self.sample_rules = sample_rules
        self.fixed_pop_size = fixed_pop_size
        self.embedding_depth = embedding_depth
        self.num_sent_rules = num_sent_rules
        self.sampling = sampling
        self.initializer = Initializer(self.corpus, self.embedding_depth, self.num_sent_rules, self.sampling)
        self.cur_key = self.initializer.cur_key
        self.unaries = self.initializer.unaries
        self.pop_size = 30
        self.distribution = Distribution(
            distribution_name, self.embedding_depth, sample_rules=self.sample_rules
        ).dist
        if self.sampling == "random":
            self.recursive_embedding = [self.unaries] + [
                [] for _ in range(self.embedding_depth)
            ]
            self.initialize_population()
        else:
            self.recursive_embedding = self.initializer.final_embedding
        # num_rules_per_level = [0] + [
        #     int(frac * self.population_size) for frac in self.distribution
        # ]
        self.lose_arms_comparative = lose_arms_comparative
        # i = 1
        # while i < len(num_rules_per_level):
        #     num_rules = num_rules_per_level[i]
        #     self.recursive_embedding[i] = self.recursive_embedding[i][:num_rules]
        #     i += 1

        # if self.sample_rules:
        #     num_primaries = num_rules_per_level[1]
        #     self.recursive_embedding = self.initializer.recursive_embedding
        #     self.recursive_embedding[1] = sorted(
        #         self.initializer.recursive_embedding[1], key=lambda x: (-1)*x.weight
        #     )[:num_primaries]
        #     self.cur_key = self.initializer.cur_key
        # self.recursive_embedding stores candidate solutions organized by layer. self.recursive_embedding stores candidate solutions,
        # self.recursive_embedding[0] stores unaries, self.recursive_embedding[1] stores primaries etc.
        # self.initialize_population()

    def initialize_population(self):
        """
        :param distribution: a list summing to 1, each entry is fraction of total rules at that embedding level
        :return:
        """
        # num_rules_per_level = [
        #     int(frac * self.pop_size) for frac in self.distribution
        # ]
        num_rules_per_level = [10, 10, 10]
        i = 1
        self.recursive_embedding[0] = self.unaries
        while i <= self.embedding_depth:
            num = num_rules_per_level[i - 1]
            for pop in range(num):
                self.generate_random_rule(i, weighting=True)
            i += 1

    def reform_layer(self, layer_index, similarity_vectors, weighting=True):
        crossover_mode = self.crossover_mode
        elitism = self.elite_rate
        self.recursive_embedding[layer_index] = sorted(
            self.recursive_embedding[layer_index],
            key=lambda x: -1 * x.weight,
        )  # Sorted such that the best rule is entry 0
        # This code copies the best rule at this generation, shown to decrease genetic diversity.
        # best_rule = copy.deepcopy(self.recursive_embedding[layer_index][0])
        # best_rule.parent_count = 0
        # best_rule.weight = 1

        # new_rule_key = self.generate_key(best_rule.embed_depth)
        # old_key = best_rule.key
        #
        # best_rule.adapt_key(new_rule_key, {old_key})
        # best_rule.prop("gain_parent", best_rule.embed_depth, 0)
        # best_rule.parent_count = 0

        layer = self.recursive_embedding[layer_index]
        layer_size = len(layer)
        elite_indices = math.ceil(layer_size * elitism)
        for psr_bundle in layer[elite_indices:]:
            psr_bundle.prop("lose_parent", psr_bundle.embed_depth, 0)
            # psr_bundle.prop("lose_parent", 0, psr_bundle.embed_depth)
        for psr_bundle in layer[:elite_indices]:
            psr_bundle.weight = 1
        # print("before", len(self.recursive_embedding[layer_index]))
        self.recursive_embedding[layer_index] = layer[:elite_indices]
        # print("after", len(self.recursive_embedding[layer_index]))
        # pairs = list(set(itertools.combinations(self.recursive_embedding[layer_index], 2)))
        pairs = list(itertools.combinations(self.recursive_embedding[layer_index], 2))

        # random.shuffle(pairs)
        # print(pairs)
        if crossover_mode == "single_child":
            for rule1, rule2 in pairs:
                baby = self.agglutinate(rule1, rule2, similarity_vectors)
                if baby is not None:
                    self.recursive_embedding[layer_index].append(baby)
                if len(self.recursive_embedding[layer_index]) >= (
                    0.75 * layer_size
                ):  # break out if producing too many crossover children
                    break

        elif crossover_mode == "two_child":
            for rule1, rule2 in pairs:
                # print(rule1.key, "cross", rule2.key)
                if self.similarity_threshold(rule1, rule2, similarity_vectors):
                    baby1, baby2 = self.crossover(rule1, rule2)
                    self.recursive_embedding[layer_index].append(baby1)
                    self.recursive_embedding[layer_index].append(baby2)
                    # print(baby1.key, baby2.key)
                if len(self.recursive_embedding[layer_index]) >= 0.75 * len(layer):
                    # break out if producing too many crossover children,
                    # this keeps genetic diversity in the system
                    break
        num_arms_layer = [len(psr.rhs) for psr in self.recursive_embedding[layer_index]]
        average_num_arms = math.ceil(sum(num_arms_layer)/len(num_arms_layer))
        for i in range(0, layer_size - len(self.recursive_embedding[layer_index])):
            rule = self.generate_random_rule(layer_index, weighting=weighting)
            j = 0
            while j < average_num_arms:  # add arms to new rules for the average size of existing rhs (test)
                self.grow_random_arm(rule, weighting=True)
                j += 1
        # self.recursive_embedding[layer_index].append(best_rule)

        # function, embed_tier, rate, mode="random"
        # self.apply_to_tier(self.grow_random_arm, layer_index, 0.3, mode="weighted")
        # self.apply_to_tier(self.lose_random_arm, layer_index, 0.5, mode="weighted")

    def similarity_threshold(self, rule1, rule2, similarity_vectors):
        sim_vec1 = similarity_vectors[rule1]
        sim_vec2 = similarity_vectors[rule2]
        cos_sim_denominator = np.linalg.norm(sim_vec1) * np.linalg.norm(sim_vec2)
        if cos_sim_denominator != 0:
            cos_sim = (sim_vec1 @ sim_vec2.T) / cos_sim_denominator
            if cos_sim < self.cross_rate:
                return True
        return False

    def crossover(self, rule1, rule2):
        """
        :param rule1: PSRBundle parent 1
        :param rule2: PSRBundle parent 2
        :return:
        """
        # Return two new rules by copying
        # The best weighted :k of each to each other

        new_rule_tier = rule1.embed_depth
        max_cutoff_point = min(len(rule1.rhs), len(rule2.rhs))
        cutoff_point = random.randint(0, max_cutoff_point)
        # TODO: Implement different ways of distributing PSRs to children (more randomness)
        old_keys = {rule1.key, rule2.key}
        new_key1 = f"CROSS{self.cross_index}"
        new_key2 = f"CROSS{self.cross_index + 1}"
        self.cross_index += 2
        list_rules = [psr for psr in rule1.rhs + rule2.rhs]
        seed_pair1 = list_rules[0]
        seed_pair2 = list_rules[cutoff_point]
        baby_rule1 = PSRBundle(
            new_key1,
            seed_pair1.left,
            seed_pair1.right,
            new_rule_tier,
            1,
        )
        baby_rule1.adapt_key(
            new_key1, old_keys
        )
        baby_rule2 = PSRBundle(
            new_key1,
            seed_pair2.left,
            seed_pair2.right,
            new_rule_tier,
            1,
        )
        baby_rule2.adapt_key(
            new_key2, old_keys
        )
        for rule in list_rules[1:cutoff_point]:
            baby_rule1.add_psr(rule)
        for rule in list_rules[cutoff_point + 1:]:
            baby_rule2.add_psr(rule)
        # print(baby_rule1)
        # print(baby_rule2)
        # baby_rule1.adapt_key(
        #     new_key1, old_keys
        # )
        # baby_rule2.adapt_key(
        #     new_key2, old_keys
        # )
        # print("_" * 20)
        # print(rule1)
        # print(rule2)
        # print(baby_rule1)
        # print(baby_rule2)
        return baby_rule1, baby_rule2

    def agglutinate(self, rule1, rule2, similarity_vectors):
        """
        :param rule1: PSRBundle parent 1
        :param rule2: PSRBundle parent 2
        :return:
        """
        # Take rule1 and rule2 comparison, only apply if selection is sufficiently different.
        # Then, if sufficiently different, return two new rules by copying
        # The best weighted :k of each to each other
        cross_rate = self.cross_rate
        new_rule_tier = max(rule1.embed_depth, rule2.embed_depth)
        sim_vec1 = similarity_vectors[rule1]
        sim_vec2 = similarity_vectors[rule2]
        cos_sim_denominator = np.linalg.norm(sim_vec1) * np.linalg.norm(sim_vec2)
        # print(cos_sim_denominator)
        if cos_sim_denominator != 0:
            cos_sim = (sim_vec1 @ sim_vec2.T) / cos_sim_denominator
            if cos_sim < cross_rate:
                list_rules_psr1 = [(psr.left, psr.right) for psr in rule1.rhs]
                list_rules_psr2 = [(psr.left, psr.right) for psr in rule2.rhs]
                list_rules = list_rules_psr1 + list_rules_psr2
                seed_pair1 = list_rules[0]
                baby_rule = PSRBundle(
                    "baby", seed_pair1[0], seed_pair1[1], new_rule_tier, 1
                )
                for left, right in list_rules[1:]:
                    baby_rule.add_rhs(left, right, 1)
                baby_rule.adapt_key(f"CROSS{self.cross_index}", {rule1.key, rule2.key})
                self.cross_index += 1
                # print("left:")
                # for psr in rule1.rhs:
                #     print(psr)
                # print("___________________")
                # print("right:")
                # for psr in rule2.rhs:
                #     print(psr)
                # print("___________________")
                # for psr in baby_rule.rhs:
                #     print(psr)
                return baby_rule
        return None

    def one_out_weights(self):
        """
        :return:
        """
        for layer in self.recursive_embedding:
            for psr in layer:
                psr.weight = 1

    def clear_unused_rules(self, embed_level):
        old_length = len(self.recursive_embedding[embed_level])
        # print("before clear: ", self.recursive_embedding[embed_level])
        deleted_rules = [psr for psr in self.recursive_embedding[embed_level] if psr.parent_count == 0]
        for rule in deleted_rules:
            rule.prop("lose_parent", embed_level, 0)
        self.recursive_embedding[embed_level] = [psr for psr in self.recursive_embedding[embed_level] if psr.parent_count != 0]
        new_length = len(self.recursive_embedding[embed_level])
        num_rules_deleted = old_length - new_length
        # print(embed_level)
        average_rhs_len = sum([len(psr.rhs) for psr in self.recursive_embedding[embed_level]]) / len(self.recursive_embedding[embed_level])
        for i in range(num_rules_deleted):
            rule = self.generate_random_rule(embed_level)
            self.recursive_embedding[embed_level].append(rule)
            for j in range(int(average_rhs_len)):
                self.grow_random_arm(rule)
        # print("after clear: ", self.recursive_embedding[embed_level])

    def generate_random_rule(self, embed_level, weighting=True):
        """
        :param recursion_level: string in {primary, secondary, sentence}
        :return:
        """
        if weighting:
            # print(self.recursive_embedding)
            level1, level2 = random.choices(self.recursive_embedding[:embed_level-1], k=2)
            left_weights = [psr.weight for psr in level1]
            right_weights = [psr.weight for psr in level2]
            rule1 = random.choices(level1, weights=left_weights, k=1)[0]
            rule2 = random.choices(level2, weights=right_weights, k=1)[0]
        else:
            level1, level2 = random.choices(self.recursive_embedding[:embed_level], k=2)
            rule1 = random.choice(level1)
            rule2 = random.choice(level2)
        key = self.generate_key(embed_level)
        # new_weight = sum([rule1.weight, rule2.weight])
        new_rule = PSRBundle(key, rule1, rule2, embed_level, 1)
        self.recursive_embedding[embed_level].append(new_rule)
        return new_rule

    def generate_unique_random_rules(self, k):
        # TODO: Make sure newly generated set doesn't contain repeats
        pass

    def generate_key(self, embed_level):
        """
        :param embed_level: Level of the rule for which the key is generated
        :return: New key ex. A1, D2, etc.
        """
        self.cur_key[embed_level] += 1
        index = self.cur_key[embed_level]
        alpha = list(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )  # Technically limits embedding to 52
        return f"{alpha[embed_level-1]}{index}"

    def mutate(self, psr_bundle):
        success = random.random() < self.mutation_rate
        if success:
            target_side = random.choice(["left", "right"])
            new_rule_tier = self.recursive_embedding[
                random.randint(0, psr_bundle.embed_depth - 1)
            ]
            new_rule_choice = random.choice(
                new_rule_tier
            )  # This is done randomly, could be done according to weight of entire layer, problem is normalization.
            target_psr_index = random.randint(0, len(psr_bundle.rhs) - 1)
            cur_psr = psr_bundle.rhs.pop(target_psr_index)
            psr_bundle.used_rhs[cur_psr] = None
            # def propagate(self, recursion_limit, recursion_count, mode, fitness=0, epsilon=0.01, phi=5):
            cur_psr.propagate(psr_bundle.embed_depth, 0, "lose_parent")
            if target_side == "left":
                left = new_rule_choice
                right = cur_psr.right
                new_weight = sum([left.weight, right.weight])
                psr_bundle.add_rhs(left, right, new_weight)
            if target_side == "right":
                left = cur_psr.left
                right = new_rule_choice
                new_weight = sum([left.weight, right.weight])
                psr_bundle.add_rhs(left, right, new_weight)
            return psr_bundle

    def grow_random_arm(self, psr_bundle, weighting=True):
        success = random.random() < self.grow_arm_rate
        if success:
            embed_level = psr_bundle.embed_depth
            if self.fixed_pop_size:
                level1, level2 = random.choices(
                    self.recursive_embedding[:embed_level], k=2
                )
                left_weights = [psr.weight for psr in level1]
                right_weights = [psr.weight for psr in level2]
                left_recurse = random.random() < self.recursion_chance
                right_recurse = random.random() < self.recursion_chance

                if left_recurse:
                    left = psr_bundle
                    psr_bundle.recur = True
                else:
                    left = random.choices(level1, weights=left_weights, k=1)[0]
                if right_recurse:
                    psr_bundle.recur = True
                    right = psr_bundle
                else:
                    right = random.choices(level2, weights=right_weights, k=1)[0]
                # new_weight = (left.weight + right.weight) / 2
                psr_bundle.add_rhs(left, right, 1)

            else:
                left = self.generate_random_rule(embed_level, weighting=weighting)
                right = self.generate_random_rule(embed_level, weighting=weighting)
                psr_bundle.add_rhs(left, right, 1)
            return psr_bundle

    def softmax_layers(self):
        """
        Creates list of lookup dictionaries, one for each layer of the embedding.
        :return:
        """
        softmaxxed = [softmax([np.log(psr.weight) for psr in layer]) for layer in self.recursive_embedding]
        # print(softmaxxed)
        # print(len(softmaxxed))
        softmax_layers = [dict(zip(self.recursive_embedding[i], softmaxxed[i])) for i in range(len(self.recursive_embedding))]
        return {k.key: v for d in softmax_layers for k, v in d.items()}

    def lose_random_arm(self, psr_bundle, soft):
        success = random.random() < self.lose_arm_rate
        if success and len(psr_bundle.rhs) > 1:
            if self.lose_arms_comparative:
                # print([psr.key for psr in self.recursive_embedding[-1]])
                # print([SymbolicPSR(psr) for psr in psr_bundle.rhs])
                # try:
                weights = [soft[psr.left.key] + soft[psr.right.key] for psr in psr_bundle.rhs]
                # except:
                #     print(soft)
                #     print(psr_bundle.key)
                #     print(psr_bundle)
                #     print([psr.key for psr in self.recursive_embedding[-1]])
                #     print([(ps.left.parent_count, ps.right.parent_count) for ps in psr_bundle.rhs])
                #     print([SymbolicPSR(psr) for psr in psr_bundle.rhs])
                #     print([[SymbolicPSR(psr) for psr in bundle.rhs] for bundle in self.recursive_embedding[-2]])
                #     print("parent count", psr_bundle.parent_count)

                # print(weights)
                max_weight = max(
                    weights
                )  # Ordering reversed so least weighted is more likely to be deleted.
                reversed_weights = [(-1 * weight) + max_weight + 1 for weight in weights]
                deletion_target = random.choices(
                    psr_bundle.rhs, weights=reversed_weights, k=1
                )[0]
                deletion_target.propagate(psr_bundle.embed_depth, 0, "lose_parent")
                psr_bundle.rhs.remove(deletion_target)
                psr_bundle.used_rhs[deletion_target] = None
            else:
                weights = [psr.left.weight + psr.right.weight for psr in psr_bundle.rhs]
                max_weight = max(
                    weights
                )  # Ordering reversed so least weighted is more likely to be deleted.
                reversed_weights = [(-1 * weight) + max_weight + 1 for weight in weights]
                deletion_target = random.choices(
                    psr_bundle.rhs, weights=reversed_weights, k=1
                )[0]
                deletion_target.propagate(psr_bundle.embed_depth, 0, "lose_parent")
                psr_bundle.rhs.remove(deletion_target)
                psr_bundle.used_rhs[deletion_target] = None

    # Deprecated elitism
    # def elitism(self, layer_index):
    #     """
    #     Eliminates PSRs with 0 parent count.
    #     :param tier: index of tier
    #     :return:
    #     """
    #     layer_length = len(self.recursive_embedding[layer_index])
    #     for psr in self.recursive_embedding[layer_index]:
    #         if psr.parent_count == 0:
    #             self.recursive_embedding[layer_index].remove(psr)
    #     cur_length = len(self.recursive_embedding[layer_index])
    #     copy_targets = random.choices(
    #         self.recursive_embedding[layer_index], k=layer_length - cur_length
    #     )
    #     for copy_target in copy_targets:
    #         new_copy = copy.copy(copy_target)
    #         new_copy.parent_count = 0
    #         new_key = self.generate_key(layer_index)
    #         new_copy.change_key(new_key)
    #         self.recursive_embedding[layer_index].append(new_copy)

    def apply_to_tier(self, function, embed_tier, rate, mode="weighted"):
        # if mode == "random":
        #     for psr in self.recursive_embedding[embed_tier]:
        #         rand = random.random()
        #         if rand < rate:
        #             function(psr)
        # if mode == "weighted":
        #     num_targets = random.randint(
        #         1, len(self.recursive_embedding[embed_tier]) - 1
        #     )
        #     pos_weights = [len(psr.rhs) for psr in self.recursive_embedding[embed_tier]]
        #     if function == self.lose_random_arm:
        #         weights = pos_weights
        #     elif function == self.grow_random_arm or function == self.mutate:
        #         max_weight = max(pos_weights)
        #         weights = [
        #             (-1 * len(psr.rhs) + (max_weight + 1)) * psr.weight
        #             for psr in self.recursive_embedding[embed_tier]
        #         ]
        #     else:
        #         weights = [psr.weight for psr in self.recursive_embedding[embed_tier]]
        #         # raw_weights = [psr.weight for psr in self.recursive_embedding[embed_tier]]
        #         # max_weight = max(raw_weights)
        #         # weights = [(-1 * weight) + max_weight + 1 for weight in raw_weights]
        #     # print(weights)
        #     weights = softmax(np.log(np.array(weights)))
        #     # print(weights)
        #     # print(f"embed tier: {embed_tier}")
        #     # print(function.__name__)
        #     target_psrs = np.random.choice(
        #         self.recursive_embedding[embed_tier],
        #         num_targets,
        #         p=weights,
        #         replace=False,
        #     )
        #     # print("genetic operation:", function)
        #     print([psr.key for psr in target_psrs])
        #     # print(self.recursive_embedding[embed_tier])
        #     for psr in target_psrs:
        #         function(psr)
        if function == self.lose_random_arm:
            soft = self.softmax_layers()
            for psr in self.recursive_embedding[embed_tier]:
                self.lose_random_arm(psr, soft)
        else:
            for psr in self.recursive_embedding[embed_tier]:
                function(psr)
