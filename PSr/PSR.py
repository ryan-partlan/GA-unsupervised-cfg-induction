
class PSR:
    def __init__(self, key, left, right):
        self.left = left
        self.right = right
        self.key = key

    def propagate(self, recursion_limit, recursion_count, mode, fitness=0, epsilon=0.01, phi=5):
        # TODO: refactor to take a function rather than mode, fitness, epsilon, phi
        if recursion_count < recursion_limit:
            if self.left.key != self.key:
                self.left.prop(mode, recursion_limit, recursion_count + 1, fitness=fitness, epsilon=epsilon, phi=phi)
            if self.right.key != self.key:
                self.right.prop(mode, recursion_limit, recursion_count + 1, fitness=fitness, epsilon=epsilon, phi=phi)

    def change_key(self, new_key):
        self.key = new_key

    def __repr__(self):
        return f"({self.key} -> {str(self.left)} {str(self.right)})"