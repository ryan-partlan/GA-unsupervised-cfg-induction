from PSr.PSR import *
from PSr.SymbolicPSR import *


class PSRBundle:
    """
    This class stores coindexed PSRs, giving functionality for tuning weights
    """
    def __init__(self, key, left, right, embed_depth, weight):
        self.key = key  # Unique, ex. A1, A2, A3, etc.
        self.embed_depth = embed_depth
        self.rhs = []  # List of possible right-hand sides that this PSR may resolve to
        self.used_rhs = {}  # dictionary psr: encoding for tracking repeats
        self.weight = weight
        self.recur = False
        self.parent_count = 0  # How many times this rule occurs in another rule
        # self.active_depth = max(left.active_depth, right.active_depth) + 1
        self.fitness = 1
        self.weights = {}  # One weight per PSR option
        self.add_rhs(left, right, weight)

    def generate_phrase(self):
        # TODO: generate an in-grammar example
        pass

    def change_key(self, new_key):
        self.key = new_key
        for psr in self.rhs:
            psr.change_key(new_key)

    def add_rhs(self, left, right, weight, parent_prop=True):
        """
        :param left: LHS of the rule
        :param right: RHS of the rule
        :param weight: Probability of this particular RHS
        :return: None
        """
        encoding = left.key + right.key
        if encoding not in self.used_rhs.values():  # Prevents repeats.
            psr = PSR(self.key, left, right)
            self.used_rhs[psr] = encoding
            self.rhs.append(psr)
            if parent_prop:
                psr.propagate(self.embed_depth, 0, "gain_parent")
            self.weights[psr] = weight
        # highest_active_depth = max([left.active_depth, right.active_depth])
        # if highest_active_depth > self.active_depth:
        #     self.active_depth = highest_active_depth + 1

    def adapt_key(self, new_key, old_keys):
        """
        On crossover, changes recursive PSRs to match the correct key
        :param new_key:
        :param old_keys: set of keys to replace, (set of strings)
        :return:
        """
        # print("_________________________")
        # print(f"{self.key}")
        # print("old keys:", old_keys)
        # print(self.rhs)
        # print("to ", new_key)
        self.key = new_key
        check_key = lambda bundle: bundle.key in old_keys
        for psr in self.rhs:
            psr.key = new_key
            # if psr.left.key in old_keys:
            changed = False
            if check_key(psr.left):
                # psr.left.prop("lose_parent", 0, self.embed_depth)
                psr.left = self
                changed = True
            if check_key(psr.right):
                # psr.right.prop("lose_parent", 0, self.embed_depth)
                psr.right = self
                changed = True
            encoding = psr.left.key+psr.right.key
            if encoding in self.used_rhs.values() and changed == True:
                psr.propagate(self.embed_depth, 0, "lose_parent")
                self.rhs.remove(psr)
            self.used_rhs[psr] = encoding
            # if changed:
            #     psr.propagate(self.embed_depth, 0, "lose_parent")
            #     encoding = psr.left.key + psr.right.key
            #     # print(encoding)
            #     if encoding in self.used_rhs.values():
            #         psr.propagate(self.embed_depth, 0, "lose_parent")
            #         self.rhs.remove(psr)
            #     else:
            #         self.used_rhs[psr] = encoding
        # print(self.key)
        # print(self.rhs)

    def add_psr(self, psr):
        """
        This function makes sure that recursion is adapted properly
        Say we're adding A -> A B to psr C, we translate to C -> C B
        :param psr:
        :return:
        """
        if psr.left.key == psr.key:
            l = self
        else:
            l = psr.left
        if psr.right.key == psr.key:
            r = self
        else:
            r = psr.right
        self.add_rhs(l, r, 1)

    def prop(self, mode, recursion_limit, recursion_count, fitness=0, epsilon=0.01, phi=5):
        if mode == "none":
            self.weight += fitness
            return
        if mode == "raw":
            self.weight += fitness
        if mode == "naive_fitness":
            self.weight += fitness / max(self.parent_count, 1)
        if mode == "rhs_fitness":
            self.weight += fitness / len(self.rhs)
        if mode == "undampened_rhs_naive_fitness":
            self.weight += fitness / len(self.rhs) + self.parent_count
        if mode == "dampened_fitness":
            self.weight += fitness / (self.parent_count + epsilon)
            epsilon = epsilon * phi
            # if self.weight < 0:
            #     print(self)
            #     print(self.weight)
            #     print("parents", self.parent_count)
            #     print("fitness", fitness)
            #     print("new_eps", epsilon)
            #     print("old_eps", epsilon/phi)\
        if mode == "rhs_dampened_fitness":
            self.weight += fitness / (len(self.rhs) + epsilon)
            epsilon = epsilon * phi
        if mode == "extra_dampened_fitness":
            self.weight += fitness / (self.parent_count + len(self.rhs) + epsilon)
            epsilon = epsilon * phi
        if mode == "gain_parent":
            self.parent_count += 1
        if mode == "lose_parent":
            self.parent_count = max(0, self.parent_count - 1)
            # self.parent_count -= 1
        for psr in self.rhs:
            psr.propagate(recursion_limit, recursion_count, mode, fitness=fitness, epsilon=epsilon, phi=phi)



    def __repr__(self):
        return f"{self.key} {[SymbolicPSR(psr) for psr in self.rhs]}"
