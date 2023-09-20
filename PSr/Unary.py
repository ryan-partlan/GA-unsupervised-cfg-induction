import math

class Unary:
    """
    Represents a Unary rule, for use in Chomsky Normalization
    """
    def __init__(self, terminal, pos_tag):
        """
        :param terminal: A set of tokens that can be tagged with pos_tag
        :param pos_tag: taken from the data, represents class of tokens for patterning.
        """
        self.terminal_set = terminal
        self.key = pos_tag
        self.fitness = 1
        self.embedding_depth = 0
        self.active_depth = 0
        self.weight = 1
        self.parent_count = 0
        self.rhs = []
        self.embed_depth = 0

    def add_token(self, token):
        """
        Adds a token to the set of terminals falling under this POS tag.
        :param token: token to be added
        :return: None
        """
        self.terminal_set.add(token)

    def prop(self, mode, recursion_limit, recursion_count, fitness, epsilon, phi):
        """
        Acts as the last point of propagation down the tree of PSRBundles and PSRs
        :param mode: what internal value to manipulate, in {"naive_fitness", "dampened_fitness", "gain_parent", "lose_parent"}
        :param recursion_limit: Present for notational convenience. See prop() documentation in PSRBundle.
        :param recursion_count: Present for notational convenience. See prop() documentation in PSRBundle.
        :param fitness:
        :param epsilon:
        :param phi: Present for notational convenience. See prop() documentation in PSRBundle.
        :return: None.
        """
        if mode == "naive_fitness":
            self.weight += fitness / self.parent_count
        if mode == "dampened_fitness":
            self.weight += fitness / (self.parent_count + epsilon)
        if mode == "gain_parent":
            self.parent_count += 1
        if mode == "lose_parent":
            # self.parent_count -= 1  # max(self.parent_count - 1, 0)
            self.parent_count = max(self.parent_count - 1, 0)

    def __repr__(self):
        """
        :return: Key is just POS tag that this represents.
        """
        return self.key
