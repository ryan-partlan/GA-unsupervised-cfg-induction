import random as random

random.seed(42)


def normalize(sample):
    s = sum(sample)
    return [val / s for val in sample]


class Distribution:
    def __init__(
        self,
        name,
        embedding_depth,
        percent_primaries=0.4,
        percent_secondaries=0.4,
        sample_rules=True,
    ):
        self.name = name
        self.sample_rules = sample_rules
        self.embedding_depth = embedding_depth
        self.percent_primaries = percent_primaries
        self.percent_secondaries = percent_secondaries
        self.dist = self.make_distribution()

    def make_distribution(self):
        """
        :return: list of probabilities taken from specified distribution. Length determined by embedding_depth.
        """
        if self.name == "equal":
            val = 1 / self.embedding_depth
            return [val for _ in range(self.embedding_depth)]
        if self.name == "gauss":
            sample = normalize(
                [
                    abs(random.gauss(mu=0.0, sigma=1.0))
                    for _ in range(self.embedding_depth)
                ]
            )
            return sample
        if self.name == "triangular":
            sample = [abs(random.triangular(1, 2)) for _ in range(self.embedding_depth)]
            return normalize(sample)
        if self.name == "more_primaries_and_secondaries":
            interval = 1 - (self.percent_primaries + self.percent_secondaries)
            val = interval / (self.embedding_depth - 2)
            middle_layers = [val for _ in range(self.embedding_depth - 3)]
            return (
                [self.percent_primaries]
                + middle_layers
                + [self.percent_secondaries]
                + [val]
            )
        else:
            return f"{self.name} sample not available"

    def __call__(self):
        return self.dist
