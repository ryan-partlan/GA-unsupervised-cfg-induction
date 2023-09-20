from PSr.SymbolicPSR import *
from PSr.Unary import *
import time

class ProcessedGrammar:

    def __init__(self, psr_bundle, unaries):
        # start_time = time.time()
        self.processed_keys = set()
        self.psr_bundle = psr_bundle
        self.unaries = unaries
        # self.unrolled_grammar, self.sentence_rules, self.num_sentence_rules, self.symb_parents = self.unroll_grammar()
        self.unrolled_grammar, self.sentence_rules, self.num_sentence_rules, self.symb_parents = self.unroll_grammar()
        self.all_rules = self.unrolled_grammar + self.unaries
        self.num_rules = len(self.all_rules)
        self.symbol_mapping, self.rule_indexing = self.derive_mappings
        self.weight_mapping = {}
        # if self.psr_bundle.recur:
        #     print(self.psr_bundle)
        #     print(len(self.all_rules))
        #     print(self.symbol_mapping)
        # print(self.num_rules)
        # print("recursive: ", psr_bundle.recur)
        # for rule in psr_bundle.rhs:
        #     print(rule)
        # print(f"time spent processing grammar: {time.time()-start_time}")

    def unroll_grammar(self):
        embed_depth = self.psr_bundle.embed_depth
        unrolled_grammar, symb_parents = self.unroll(self.psr_bundle, embed_depth, i=0)
        # unrolled_grammar = self.unroll(self.psr_bundle, embed_depth, i=0)
        unrolled_grammar = self.eliminate_dupes(unrolled_grammar)
        sentence_rules = []
        for psr in self.psr_bundle.rhs:
            symb_psr = SymbolicPSR(psr)
            sentence_rules.append(symb_psr)
            symb_parents[symb_psr] = self.psr_bundle
        unrolled_grammar = list(unrolled_grammar)
        full_unrolled_grammar = (
            sentence_rules + unrolled_grammar
        )  # For parsing, all sentence rules must be ordered at the beginning
        # full_unrolled_grammar = (
        #     unrolled_grammar + sentence_rules #try reverse
        # )  # For parsing, all sentence rules must be ordered at the beginning
        return full_unrolled_grammar, sentence_rules, len(sentence_rules), symb_parents

    def unroll(self, psr_bundle, recursion_limit, i=0):
        """
        :param psr_bundle: A PSRBundle to be converted to CNF
        :param i: current level of embedding
        :param recursion_limit: maximum embedding, recursion should not go further than this
        :return: list of nonterminal rules
        """
        symb_parent = {}
        if i > recursion_limit:
            return set(), {}
        if isinstance(psr_bundle, Unary):
            return set(), {}
        unrolled = set()
        for psr in psr_bundle.rhs:
            if i > 0:
                symbolic_psr = SymbolicPSR(psr)
                unrolled.add(symbolic_psr)
                #symb_parent[symbolic_psr] = psr_bundle
                symb_parent.update({symbolic_psr: psr_bundle})
            if psr.left.key != psr.key:
                side1_unrolled, symb_parents1 = self.unroll(psr.left, recursion_limit, i=i+1)
                unrolled |= side1_unrolled
                symb_parent.update(symb_parents1)
            if psr.right.key != psr.key:
                side2_unrolled, symb_parents2 = self.unroll(psr.right, recursion_limit, i=i+1)
                unrolled |= side2_unrolled
                symb_parent.update(symb_parents2)
            # symb_parent.update(symb_parents1)
            # symb_parent.update(symb_parents2)
            # side1_unrolled, symb_parents2 = self.unroll(psr.left, recursion_limit, i=i+1)
            # side2_unrolled, symb_parents2 = self.unroll(psr.right, recursion_limit, i=i+1)
            # # side1_unrolled = self.unroll(psr.left, recursion_limit, i=i+1)
            # # side2_unrolled = self.unroll(psr.right, recursion_limit, i=i+1)
            # unrolled |= side1_unrolled
            # unrolled |= side2_unrolled
            # # unrolled |= self.unroll(psr.left, recursion_limit, i=i+1)
            # # unrolled |= self.unroll(psr.right, recursion_limit, i=i+1)
        return unrolled, symb_parent

    @property
    def derive_mappings(self):
        """
        Produces two essential mappings, the first taking symbol to psr, the second taking each rule to a unique index.
        :return: (symbol_mapping, rule_indexing)
        """
        symbol_mapping = {}
        rule_indexing = {}
        i = 0
        first_use_of_key_dict = {}
        for rule in self.all_rules:
            if rule.key not in first_use_of_key_dict.keys():
                first_use_of_key_dict[rule.key] = rule
                rule_indexing[rule] = i
                i += 1
            elif rule.key in first_use_of_key_dict.keys():
                rule_indexing[rule] = rule_indexing[first_use_of_key_dict[rule.key]]

            if rule.key in symbol_mapping.keys():
                symbol_mapping[rule.key].add(rule)
            else:
                symbol_mapping[rule.key] = {rule}
        return symbol_mapping, rule_indexing

    def eliminate_dupes(self, unrolled_grammar):
        # Here's a dumb hack to get rid of duplicate reps with the same ID.
        string_set = set()
        no_dupes = set()
        for symb_psr in unrolled_grammar:
            code = symb_psr.key + symb_psr.left + symb_psr.right
            if code not in string_set:
                no_dupes.add(symb_psr)
                string_set.add(code)
        return no_dupes