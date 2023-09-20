class SymbolicPSR:
    """
    This class gives a simple string representation of a PSR for the sake of processing.
    """
    def __init__(self, psr):
        self.key = psr.key
        self.left = psr.left.key
        self.right = psr.right.key
        self.weight = psr.left.weight + psr.right.weight

    def __repr__(self):
        """
        :return: Easily interpretable result, no specifying options.
        """
        return f"{self.key} -> {self.left} {self.right}"
