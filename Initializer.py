from PSr.Unary import *
from PSr.PSRBundle import *
from collections import Counter
import nltk


class Initializer:
    def __init__(self, corpus, embed_depth, num_sentence, sampling):
        self.corpus = corpus
        self.embed_depth = embed_depth
        self.init_embedding = [[] for _ in range(self.embed_depth + 1)]
        set_embedding = [set() for _ in range(self.embed_depth)]
        self.rule_keys = {}
        self.ngram_list = [[] for _ in range(self.embed_depth + 1)]
        self.cur_key = [0 for _ in range(self.embed_depth + 1)]
        self.unaries, self.unary_map = self.derive_unary_rules()
        self.init_embedding[0] = self.unaries
        self.sampling = sampling
        if self.sampling != "random":
            i = 0
            while i <= self.embed_depth:
                self.initialize_ngram_layer(i)
                i += 1
            i = 0
            while i <= self.embed_depth:
                self.init_embedding[i] = sorted(
                    self.init_embedding[i], key=lambda psr: (-1) * psr.weight
                )
                i += 1
            set_embedding[-1] = self.init_embedding[-1][:num_sentence]
            i = 0
            while i < embed_depth - 1:
                for bund in set_embedding[-1 - i]:
                    psr = bund.rhs[0]
                    set_embedding[-1 - i - 1].add(psr.left)
                    set_embedding[-1 - i - 1].add(psr.right)
                i += 1
            self.final_embedding = [self.unaries] + [list(layer) for layer in set_embedding]
        # print(self.final_embedding)
        # print([len(layer) for layer in self.recursive_embedding])
        # print([len(layer) for layer in self.recursive_embedding])
        # print([el.weight for el in self.recursive_embedding[-1]])

    def derive_unary_rules(self):
        """
        :param dataset: Preprocessed list for which each POS tag is element 0, token is element 1
        :return: tuple(mapping between POS tag and their unary rule, set of unary rules, list of all used pos tags)
        """
        unaries = {}
        for example in self.corpus:
            for pair in example:
                pos_tag, token = pair
                if pos_tag in unaries.keys():
                    unaries[pos_tag].add_token(token)
                else:
                    unaries[pos_tag] = Unary({token}, pos_tag)
        # return [unary.key for unary in unaries.values()]
        return list(unaries.values()), unaries
        # return unaries, set(unaries.values()), list(unaries.keys())

    def assign_probability_layer(self, n):
        total_rules = 0
        rule_count = Counter()
        all_rules = set()
        for layer in self.ngram_list[n]:
            for rule in layer:
                rule_count[rule] += 1
                total_rules += 1
                all_rules.add(rule)
        for rule in all_rules:
            rule.weight += rule_count[rule] / total_rules
            # print(rule, rule.weight)

    def initialize_ngram_layer(self, n):
        layer = []
        if n == 0:
            for example in self.corpus:
                example_unaries = []
                for pos, _ in example:
                    corresponding_unary = self.unary_map[pos]
                    example_unaries.append(corresponding_unary)
                layer.append(example_unaries)

        else:
            existing_rules = set()
            rule_mapping = {}
            for example in self.ngram_list[n - 1]:
                example_naries = []
                if self.sampling == "bigram":
                    grams = nltk.bigrams(example)
                if self.sampling == "adjacent":
                    if len(example) == 2:
                        grams = [(example[0], example[1])]
                    else:
                        grams = [(example[i], example[i+1]) for i in range(len(example)-1) if i % 2 == 0]
                for gram in grams:
                    if gram not in existing_rules:
                        existing_rules.add(gram)
                        new_key = self.generate_key(n)
                        new_weight = gram[0].weight + gram[1].weight
                        rule = PSRBundle(new_key, gram[0], gram[1], n, new_weight)
                        self.rule_keys[new_key] = rule
                        self.init_embedding[n].append(rule)
                        example_naries.append(rule)
                        rule_mapping[gram] = rule
                    else:
                        example_naries.append(rule_mapping[gram])
                layer.append(example_naries)
        self.ngram_list[n] = layer
        self.assign_probability_layer(n)
        # print([[x.weight for x in layer] for layer in self.recursive_embedding])

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
