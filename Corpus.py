from PSr.Unary import *
import conllu
import os
import re
from torch.utils.data import DataLoader

class Corpus:
    def __init__(
        self,
        directory,
        train_size=2,
        dev_size=2,
        test_size=2,
        max_length=4,
        min_length=3,
        mode="toy",
        batch_size=1,
        reduce_ambiguity=True,
    ):
        self.directory = directory
        if mode == "toy":
            self.corpus = [
                [("ART", "the"), ("NN", "dog"), ("V", "runs")],
                [("ART", "the"), ("NN", "cat"), ("Vpast", "ran")],
                [("ART", "a"), ("NN", "capybara"), ("V", "eats"), ("NN", "food")],
                [("V", "swim"), ("P", "to"), ("NN", "mama")],
                [("ART", "the"), ("NN", "man"), ("Vpast", "wept")],
                [
                    ("ART", "the"),
                    ("NN", "man"),
                    ("Vpast", "wept"),
                    ("ART", "the"),
                    ("NN", "man"),
                    ("Vpast", "wept"),
                ],
            ]
        if mode == "gum":
            self.corpus = self.process_gum()

        if mode == "brown":
            self.reduce_ambiguity = reduce_ambiguity
            self.corpus = self.process(directory)

        self.max_length = max_length
        self.min_length = min_length
        self.train_size = train_size
        self.dev_size = dev_size
        self.test_size = test_size
        self.corpus = self.filter(self.corpus)
        self.unbatched_train_set = self.corpus[:self.train_size]
        self.train_set = self.batchify(self.unbatched_train_set, batch_size)
        self.dev_set = self.corpus[self.train_size:self.train_size + self.dev_size]
        self.test_set = self.corpus[self.train_size + self.dev_size:]

    def filter(self, corpus):
        """
        Filters out for desired example length
        :param corpus:
        :return:
        """
        filtered_corpus = []
        total_examples = self.train_size + self.dev_size + self.test_size
        added_examples = 0
        for example in corpus:
            if len(example) <= self.max_length and len(example) > self.min_length:
                filtered_corpus.append(example)
                added_examples += 1
            if added_examples >= total_examples:
                return filtered_corpus
        return filtered_corpus

    def process(self, directory):
        """
        :param directory: directory from which to take the corpus
        :return: list of all examples in the corpus, each example being a list of tuples (POS, TOKEN)
        """
        corpus = []
        train_path = directory + "/train"
        for filename in os.listdir(train_path):
            with open(os.path.join(train_path, filename), "r") as file:
                data = file.read()
                no_hard_returns = re.sub(
                    "(\n)*", "", data
                )  # Get rid of new line characters
                split_on_periods = re.split("\./\.|;", no_hard_returns)
                # split on ./. marker or the ;
                split_on_tabs = [
                    re.sub("(\t)*", "", example.lower()).split(" ")
                    for example in split_on_periods
                    if len(example) > 1
                ]
                processed_example = [
                    [
                        tuple(token.split("/", maxsplit=2))[::-1]
                        for token in example
                        if len(token.split("/", maxsplit=2)) == 2
                    ]
                    for example in split_on_tabs
                ]  # Brown corpus has POS tag second, hence the reverse.
                if self.reduce_ambiguity:
                    for example in processed_example:
                        reduced_example = []
                        for pos, tok in example:
                            eliminate_ambiguity = pos.split("-", maxsplit=1)[0].strip(
                                "$"
                            )
                            new_pos = eliminate_ambiguity.split("+", maxsplit=1)[
                                0
                            ].strip("$")
                            if new_pos not in {".", "", ",", ":", "`", "''", "*", "``"}:
                                reduced_example.append(tuple([new_pos, tok]))
                        corpus.append(reduced_example)
                else:
                    corpus.append(processed_example)
        return corpus

    def process_gum(self):
        corpus = []
        for filename in os.listdir(self.directory):
            if filename.split("_")[1] != "reddit":
                # reddit data does not have tokens due to licensing.
                with open(
                    os.path.join(self.directory, filename), mode="r", encoding="utf8"
                ) as file:
                    annotations = file.read()
                    sentences = conllu.parse(annotations)
                    for sentence in sentences:
                        example = [
                            (token["upos"], token["form"])
                            for token in sentence
                            if token["upos"]
                            not in {"PUNCT", "PROPN", "X", "_", "NUM", "SYM"}
                        ]
                        corpus.append(example)
        return corpus

    def batchify(self, dataset, batch_size):
        """
        :return:
        """
        l = len(dataset)
        batched = []
        for ndx in range(0, l, batch_size):
            batched.append(dataset[ndx : min(ndx + batch_size, l)])
        return batched
