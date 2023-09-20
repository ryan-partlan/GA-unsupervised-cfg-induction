import numpy as np
from PSr.ProcessedGrammar import *
from joblib import Parallel, delayed

# np.set_printoptions(threshold=np.inf)
import time


class Parser:
    def __init__(self, corpus, unaries, fitness_propagation, epsilon, phi):
        self.corpus = corpus.corpus
        self.unaries = unaries
        self.fitness_propagation = fitness_propagation
        self.epsilon = epsilon
        self.phi = phi
        self.grammar = None

    def parse_corpus(
        self, psr_bundle, batch, parser_mode="train", tinker_subtrees=False, stimulus=1
    ):
        """
        This function just caches a bunch of information about the CFG, so we don't have to unroll at every step.
        :param psr_bundle: PSRBundle representing CFG
        :param corpus: a list of tuples (pos, word)
        :return: binary array, 1 for parsed success, 0 for failure. (For stochastic, returns weight of max weight parse)
        """
        grammar = ProcessedGrammar(psr_bundle, self.unaries)

        nonterminals = grammar.unrolled_grammar
        # nonunaries = grammar.nonunaries
        num_rules = grammar.num_rules
        rule_indexing = grammar.rule_indexing
        symbol_mapping = grammar.symbol_mapping
        symb_parents = grammar.symb_parents
        # print(num_rules)
        num_sentence_rules = grammar.num_sentence_rules
        weight_mapping = grammar.weight_mapping

        # start = time.time()
        # vec = [
        #     self.parse(
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
        # ]
        # print(f"Time for regular parse: {time.time()-start}")
        # start = time.time()
        vec = Parallel(n_jobs=-1)(
            delayed(self.parse)(
                example,
                nonterminals,
                num_rules,
                rule_indexing,
                symbol_mapping,
                num_sentence_rules,
                parser_mode,
                weight_mapping,
                symb_parents,
                tinker_subtrees=tinker_subtrees,
                stimulus=stimulus,
            )
            for example in batch
        )
        # print(f"Grammar size {num_rules}")
        # print(f"Time for parallel parse: {time.time()-start}")
        return vec

    def parse(
        self,
        example,
        nonterminals,
        num_rules,
        rule_indexing,
        symbol_mapping,
        num_sentence_rules,
        parser_mode,
        weight_mapping,
        symb_parents,
        tinker_subtrees=True,
        stimulus=1,
    ):
        """
        :param example: The example to be parsed
        :param nonterminals: list of symbolicCFGs that are not Unary
        :param num_rules: number of total rules
        :param rule_indexing: dictionary {SymbolicPSR : index}, rules with the same key are co-indexed
        :param symbol_mapping: dictionary {SymbolicPSR : list[rules]}
        :param num_sentence_rules:
        :return:
        """
        n = len(example)
        num_rules = max(rule_indexing.values()) + 1
        if n == 0 or n == 1:
            return 0
        unaries = self.unaries
        parse_trellis = np.full((n, n, num_rules), 0.0)
        # print(nonterminals)
        # print(symbol_mapping)
        # trellis[n, n, r] keeping track of failure/success of grammar to parse example
        # back = [[[] for _ in range(num_rules)] for _ in range(n)]
        # back[n, n, r] list of lists of backpointer triples
        maximal_parse_length = 0
        # print(symb_parents.keys())
        for s, (pos, word) in enumerate(example):
            for unary in unaries:
                if word in unary.terminal_set:
                    if parser_mode == "weight":
                        unary_add = unary.weight
                    if parser_mode == "proportion" or parser_mode == "eval":
                        unary_add = 1
                    parse_trellis[0, s, rule_indexing[unary]] = unary_add
        for length in range(1, n + 1):
            for start in range(0, n - length + 1):
                for partition in range(1, length):
                    for nonterminal in nonterminals:
                        a = rule_indexing[nonterminal]
                        left = nonterminal.left
                        right = nonterminal.right
                        for rule1 in symbol_mapping[left]:
                            b = rule_indexing[rule1]
                            for rule2 in symbol_mapping[right]:
                                c = rule_indexing[rule2]
                                if parser_mode == "weight":
                                    add = nonterminal.weight
                                if parser_mode == "proportion" or parser_mode == "eval":
                                    add = 1
                                if (
                                    parse_trellis[partition - 1, start, b]
                                    and parse_trellis[
                                        length - partition - 1, start + partition, c
                                    ]
                                ):
                                    parse_trellis[length - 1, start, a] = add
                                    # print("Found subtree")
                                    # print(nonterminal)
                                    # print(f"of length: {length + start}")
                                    # print("partition", partition)
                                    # print("length", length)
                                    # print("start", start)
                                    if tinker_subtrees:
                                        bund = symb_parents[nonterminal]
                                        f = length / (n + len(bund.rhs))
                                        # if self.fitness_propagation == "none":
                                        #     symbol_mapping[nonterminal].weight += f
                                        # symb_parents[nonterminal].prop(
                                        #     self.fitness_propagation,
                                        #     symb_parents[nonterminal].embed_depth,
                                        #     0,
                                        #     fitness=f,
                                        #     epsilon=self.epsilon,
                                        #     phi=self.phi,
                                        # )
                                        # bund.prop(
                                        #     self.fitness_propagation,
                                        #     symb_parents[nonterminal].embed_depth,
                                        #     0,
                                        #     fitness=f,
                                        #     epsilon=self.epsilon,
                                        #     phi=self.phi,
                                        # )
                                        bund.prop(
                                            'none',
                                            symb_parents[nonterminal].embed_depth,
                                            0,
                                            fitness=f,
                                            epsilon=self.epsilon,
                                            phi=self.phi,
                                        )

                                    # highest_symb_rule = symb_parents[nonterminal]
                                    maximal_parse_length = max(
                                        maximal_parse_length, length
                                    )
        # print(maximal_parse_length)
        sentence_detected = parse_trellis[n - 1, 0, :num_sentence_rules].any()
        # if parser_mode == "eval" and (maximal_parse_length / n) == 1 and sentence_detected:
        #     print("parse trellis shape", parse_trellis.shape)
        #     print("seq length", n)
        #     print("max parse", maximal_parse_length)
        #     print("num sentence rules", num_sentence_rules)
        #     print("nonzero indices", np.transpose(np.nonzero(parse_trellis)))
        #     print("_"*10)
        # TODO: IF MAX PARSE LENGTH IS 1 BUT SENTENCE NOT DETECTED, TRANSFER TO SENTENCE RULE
        if sentence_detected:
            if parser_mode == "weight":
                # return max(parse_trellis[n - 1, 0, :num_sentence_rules])
                return max(parse_trellis[n - 1, 0, :num_sentence_rules])
            if parser_mode == "proportion":
                return stimulus
            if parser_mode == "eval":
                return 1.0
        else:
            if parser_mode == "eval":
                return 0
            if parser_mode == "proportion" or parser_mode == "weight":
                return maximal_parse_length / n
