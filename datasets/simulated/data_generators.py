from abc import ABC, abstractmethod
from typing import Literal, Any
import numpy as np
from lark import Lark, Tree, Token
from utils import skellam


class SimulatedDataGenerator(ABC):
    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Generates a sample of size n from the distribution.

        Args:
            n (int): Number of samples

        Returns:
            tuple[np.ndarray, np.ndarray]: (W integer matrix, Z float matrix)
        """
        w = self.sample_w(n)
        z = self.sample_z_given_w(w)
        return w, z

    def compositionality(
        self, z: np.ndarray, w: np.ndarray, per_sample: bool = False
    ) -> np.ndarray:
        """Returns the compositionality of a representation K(Z)/K(Z|W) in nats.

        Args:
            z (np.ndarray): (n, d) float matrix of representations Z.
            w (np.ndarray): (n, k) integer matrix of sentences W.
            per_sample (bool): Whether to use per-sample terms or sums.

        Returns:
            float: The compositionality of a representation.
        """
        return self.k_z(z, w, per_sample=per_sample) / self.k_z_given_w(
            z, w, per_sample=per_sample
        )

    def k_z(self, z: np.ndarray, w: np.ndarray, per_sample: bool = False) -> float:
        """Returns the expected Kolmogorov complexity of a representation in nats.

        Args:
            z (np.ndarray): (n, d) float matrix of representations Z.
            w (np.ndarray): (n, k) integer matrix of sentences W.
            per_sample (bool): Whether to use per-sample terms or sums.

        Returns:
            float: Either the expected per-sample or the total Kolmogorov complexity in nats.
        """
        return (
            self.k_language
            + self.k_w_given_language(w, per_sample=per_sample)
            + self.k_decoder
            + self.k_z_given_w_and_decoder(z, w, per_sample=per_sample)
        )

    def k_z_given_w(
        self, z: np.ndarray, w: np.ndarray, per_sample: bool = False
    ) -> float:
        """Returns the Kolmogorov complexity of a representation given W in nats.

        Args:
            z (np.ndarray): (n, d) float matrix of representations Z.
            w (np.ndarray): (n, k) integer matrix of sentences W.
            per_sample (bool): Whether to use per-sample terms or sums.

        Returns:
            float: Either the expected per-sample or the total Kolmogorov complexity in nats.
        """
        return self.k_decoder + self.k_z_given_w_and_decoder(
            z, w, per_sample=per_sample
        )

    def k_w_given_language(self, w: np.ndarray, per_sample: bool = False) -> float:
        """Returns the Kolmogorov complexity of a given W given p(w) in nats.

        Args:
            w (np.ndarray): (n, k) integer matrix of sentences W.
            per_sample (bool): Whether to use per-sample terms or sums.

        Returns:
            float: Either the expected per-sample or the total Kolmogorov complexity in nats.
        """
        if per_sample:
            return -self.logp_w(w).mean()
        else:
            return -self.logp_w(w).sum()

    def k_z_given_w_and_decoder(
        self, z: np.ndarray, w: np.ndarray, per_sample: bool = False
    ) -> float:
        """Returns the Kolmogorov complexity of a given Z given W and p(z | w) in nats.

        Args:
            z (np.ndarray): (n, d) float matrix of representations Z.
            w (np.ndarray): (n, k) integer matrix of sentences W.
            per_sample (bool): Whether to use per-sample terms or sums.

        Returns:
            float: Either the expected per-sample or the total Kolmogorov complexity in nats.
        """
        if per_sample:
            return -self.logp_z_given_w_and_decoder(z, w).mean()
        else:
            return -self.logp_z_given_w_and_decoder(z, w).sum()

    @property
    @abstractmethod
    def k_language(self) -> float:
        """The Kolmogorov complexity of the language function that describes p(w) in nats."""
        pass

    @property
    @abstractmethod
    def k_decoder(self) -> float:
        """The Kolmogorov complexity of the decoding function that describes p(z | w) in nats."""
        pass

    @abstractmethod
    def sample_w(self, n: int) -> np.ndarray:
        """Generates a sample of size (n, k) from the distribution over sentences.

        Args:
            n (int): Number of samples

        Returns:
            tuple[np.ndarray, np.ndarray]: (n, k) integer matrix of sentences W.
        """
        pass

    @abstractmethod
    def sample_z_given_w(self, w: np.ndarray) -> np.ndarray:
        """Generates a sample of size (n, d) from the distribution over representations given sentences.

        Args:
            w (np.ndarray): (n, k) integer matrix of sentences W.

        Returns:
            np.ndarray: (n, d) float matrix of representations Z.
        """
        pass

    @abstractmethod
    def logp_w(self, w: np.ndarray) -> np.ndarray:
        """Returns the log probability of a given W.

        Args:
            w (np.ndarray): Integer matrix of size (n, k)

        Returns:
            np.ndarray: (n,) vector of sentence log probabilities.
        """
        pass

    @abstractmethod
    def logp_z_given_w_and_decoder(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Returns the log probability of a given Z given W.

        Args:
            z (np.ndarray): Float matrix of size (n, d)
            w (np.ndarray): Integer matrix of size (n, k)

        Returns:
            np.ndarray: (n,) vector of representation log probabilities.
        """
        pass


##############################################################
########## Subclasses for different data generators ##########
##############################################################


class LookupTableDataGenerator(SimulatedDataGenerator):
    AggregationType = Literal["concatenation", "addition", "multiplication"]

    def __init__(
        self,
        z_dim: int,
        num_words: int,
        vocab_size: int,
        disentanglement: int,
        precision: float = 0.01,
        noise_ratio: float = 0.1,
        aggregation: AggregationType = "concatenation",
        random_seed: int = 0,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.num_words = num_words
        self.vocab_size = vocab_size
        self.disentanglement = disentanglement
        self.precision = precision
        self.noise_ratio = noise_ratio
        self.aggregation = aggregation
        self.random_state = np.random.RandomState(random_seed)

        self.num_positions = num_words // disentanglement
        self.num_entries = vocab_size**disentanglement
        self.component_dim = (
            z_dim // self.num_positions if aggregation == "concatenation" else z_dim
        )

        self.lookup_table = skellam.approx_gaussian_sample(
            mean=0.0,
            std=1.0,
            shape=(self.num_entries, self.component_dim),
            precision=self.precision,
            random_state=self.random_state,
        ).astype(np.float32)

    def decode_w(self, w: np.ndarray) -> np.ndarray:
        z = []
        for wi in w:
            zi = []
            wi = np.split(wi, self.num_positions)
            for wi_pos in wi:
                entry = (
                    (wi_pos * (self.vocab_size ** np.arange(self.disentanglement)))
                    .sum()
                    .astype(int)
                )
                zi.append(self.lookup_table[entry])
            if self.aggregation == "concatenation":
                zi = np.concatenate(zi)
            elif self.aggregation == "addition":
                zi = np.sum(zi, axis=0)
            elif self.aggregation == "multiplication":
                zi = np.prod(zi, axis=0)
            z.append(zi)
        z = np.stack(z)
        return z

    @property
    def k_language(self) -> float:
        # Have to specify the bits for two integer numbers that define the uniform distribution, and the rest of p(w) has (small) constant complexity.
        return np.log(self.vocab_size) + np.log(self.num_words)

    @property
    def k_decoder(self) -> float:
        # Have to specify the bits for all the numbers in the lookup table,  and the rest of p(z | w) has (small) constant complexity.
        logp = skellam.approx_gaussian_logpmf(
            x=self.lookup_table, mean=0.0, std=1.0, precision=self.precision
        ).sum()
        return -logp

    def sample_w(self, n: int) -> np.ndarray:
        return self.random_state.randint(0, self.vocab_size, size=(n, self.num_words))

    def sample_z_given_w(self, w: np.ndarray) -> np.ndarray:
        z = self.decode_w(w)
        if self.noise_ratio > 0:
            noise = skellam.approx_gaussian_sample(
                mean=0.0,
                std=self.noise_ratio,
                shape=z.shape,
                precision=self.precision,
                random_state=self.random_state,
            ).astype(np.float32)
            z += noise
        return z

    def logp_w(self, w: np.ndarray) -> np.ndarray:
        return (
            np.ones(w.shape[0], dtype=np.float32)
            * np.log(1 / self.vocab_size)
            * self.num_words
        )

    def logp_z_given_w_and_decoder(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        z_mean = self.decode_w(w)
        noise = z - z_mean
        logp = skellam.approx_gaussian_logpmf(
            x=noise, mean=0.0, std=self.noise_ratio, precision=self.precision
        ).sum(axis=1)
        return logp


class SyntacticDataGenerator(SimulatedDataGenerator):
    CompositionType = Literal["linear", "tensorproduct"]

    def __init__(
        self,
        z_dim: int,
        num_words: int,
        vocab_size: int,
        vocab_terms: dict[int, str],
        grammar: dict[tuple[str, str], str],
        roots: list[str],
        language: dict[str, list[str]] | None = None,
        precision: float = 0.01,
        noise_ratio: float = 0.1,
        composition: CompositionType = "linear",
        random_seed: int = 0,
    ) -> None:
        # Note: we are generating sentences uniformly at random, so the grammar must
        # permit all possible sentences. If this does not happen, the parser will
        # raise an error when trying to produce Z.
        super().__init__()

        # All words have an associated terminal POS
        assert all([word in vocab_terms for word in range(vocab_size)])

        # The grammar is formatted correctly, with non-terminals
        # in lowercase and terminals in uppercase
        assert all([term.upper() == term for term in vocab_terms.values()])
        assert all([nonterm.lower() == nonterm for nonterm in grammar.values()])

        # Valid language sequence generation model
        if language is not None:
            assert "START" in language
            for term in vocab_terms.values():
                assert term in language
            for i, js in language.items():
                if i != "START":
                    assert i in vocab_terms.values()
                for j in js:
                    assert j in vocab_terms.values()

        self.z_dim = z_dim
        self.num_words = num_words
        self.vocab_size = vocab_size
        self.roots = roots
        self.precision = precision
        self.noise_ratio = noise_ratio
        self.composition = composition
        self.random_state = np.random.RandomState(random_seed)

        # Invert the dictionaries to represent it as a
        # generative grammar (more convenient inside the class)
        self.vocab_terms = {}
        for word, term in vocab_terms.items():
            if term not in self.vocab_terms:
                self.vocab_terms[term] = []
            self.vocab_terms[term].append(word)
        self.grammar = {}
        for ij, k in grammar.items():
            if k not in self.grammar:
                self.grammar[k] = []
            self.grammar[k].append(ij)

        # Grammar parser
        self.parser = self.make_parser()

        # Sequence generation model
        if language is None:
            terms = list(self.vocab_terms.keys())
            self.language = {"START": terms, **{term: terms for term in terms}}
        else:
            self.language = language
        self.transition_matrix = np.zeros((vocab_size + 1, vocab_size))
        for i, js in self.language.items():
            term_i_words = self.vocab_terms[i] if i != "START" else [vocab_size]
            for j in js:
                term_j_words = self.vocab_terms[j]
                self.transition_matrix[
                    np.array(term_i_words).reshape(-1, 1),
                    np.array(term_j_words).reshape(1, -1),
                ] = 1
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)

        # Make role embeddings
        if composition == "tensorproduct":
            parts_of_speech = []
            for rule_options in self.grammar.values():
                for input_i, input_j in rule_options:
                    parts_of_speech += [input_i, input_j]
            parts_of_speech = set(parts_of_speech)
            self.role_embeddings = {
                pos: skellam.approx_gaussian_sample(
                    mean=0.0,
                    std=1.0,
                    shape=(self.z_dim,),
                    precision=self.precision,
                    random_state=self.random_state,
                ).astype(np.float32)
                for pos in parts_of_speech
            }

        self.word_embeddings = skellam.approx_gaussian_sample(
            mean=0.0,
            std=1.0,
            shape=(vocab_size, z_dim),
            precision=precision,
            random_state=self.random_state,
        ).astype(np.float32)
        self.rules = {(i, j): self.make_rule(i, j) for i, j in grammar}

    def make_parser(self) -> Lark:
        start = "start: " + " | ".join(self.roots)

        rules = {
            k: " | ".join([f'{i} " " {j}' for i, j in ij])
            for k, ij in self.grammar.items()
        }
        rules = "\n".join([f"{k}: {ij}" for k, ij in rules.items()])

        terminals = {
            term: [f'"{str(word)}"' for word in words]
            for term, words in self.vocab_terms.items()
        }
        terminals = {term: " | ".join(words) for term, words in terminals.items()}
        terminals = "\n".join([f"{term}: {words}" for term, words in terminals.items()])

        grammar = "\n".join([start, rules, terminals])
        parser = Lark(grammar, start="start", parser="earley")

        return parser

    def make_rule(self, i: str, j: str) -> Any:
        if self.composition == "linear":
            w = skellam.approx_gaussian_sample(
                mean=0.0,
                std=1.0,
                shape=(2 * self.z_dim, self.z_dim),
                precision=self.precision,
                random_state=self.random_state,
            ).astype(np.float32)
            return w
        elif self.composition == "tensorproduct":
            role_i, role_j = self.role_embeddings[i], self.role_embeddings[j]
            return role_i, role_j
        else:
            raise ValueError("Composition type not recognized")

    def parse_w(self, w: np.ndarray) -> list[Tree]:
        w_parses = []
        for wi in w:
            wi = " ".join([str(word) for word in wi])
            parse = self.parser.parse(wi).children[0]
            w_parses.append(parse)
        return w_parses

    def apply_rule(self, x: np.ndarray, y: np.ndarray, rule_params: Any):
        if self.composition == "linear":
            return np.concatenate([x, y]) @ rule_params
        elif self.composition == "tensorproduct":
            # See https://arxiv.org/pdf/2106.01317#page=3.46
            role_x, role_y = rule_params
            tpr_i, tpr_j = (x * role_x + x), (y * role_y + y)
            return tpr_i + tpr_j  # The sum here is arbitrary, the paper doesn't specify
        else:
            raise ValueError("Composition type not recognized")

    def decode_w(self, w: np.ndarray) -> np.ndarray:
        def decode_parse(parse: Tree | Token):
            if isinstance(parse, Token):
                word = int(parse.value)
                return self.word_embeddings[word]
            else:
                x, y = parse.children
                x_pos = x.data.value if isinstance(x, Tree) else x.type
                y_pos = y.data.value if isinstance(y, Tree) else y.type
                x_z, y_z = decode_parse(x), decode_parse(y)
                rule_params = self.rules[(x_pos, y_pos)]
                return self.apply_rule(x_z, y_z, rule_params)

        z = []
        w_parses = self.parse_w(w)
        for parse in w_parses:
            z.append(decode_parse(parse))
        return np.stack(z)

    @property
    def k_language(self) -> float:
        # Have to specify:
        # 1) The terminal POS for each vocabulary item, which is a vocab_size'd list of integers that each take log(num_terms) bits to encode
        # 2) The permissible terminal POS's that follow each other, which is a (num_terms + 1) x num_terms matrix of binary values (the +1 is for the start token)
        vocab_bits = self.vocab_size * np.log(len(self.vocab_terms))
        transition_bits = (
            (len(self.vocab_terms) + 1) * len(self.vocab_terms) * np.log(2)
        )
        return vocab_bits + transition_bits

    @property
    def k_decoder(self) -> float:
        # Have to specify the rule semantics and the parsing algorithm.
        # The parsing algorithm grows in complexity as a function of the grammar size.
        # The rule semantics grow in complexity as a function of the vocabulary size
        # and the grammar size. We can therefore compute the complexity
        # by counting the number of bits needed to specify the rule semantics.
        logp_emb = skellam.approx_gaussian_logpmf(
            x=self.word_embeddings, mean=0.0, std=1.0, precision=self.precision
        ).sum()
        if self.composition == "linear":
            logp_rules = sum(
                [
                    skellam.approx_gaussian_logpmf(
                        x=r, mean=0.0, std=1.0, precision=self.precision
                    ).sum()
                    for r in self.rules.values()
                ]
            )
        elif self.composition == "tensorproduct":
            logp_rules = sum(
                [
                    skellam.approx_gaussian_logpmf(
                        x=role, mean=0.0, std=1.0, precision=self.precision
                    ).sum()
                    for role in self.role_embeddings.values()
                ]
            )
        else:
            raise ValueError("Composition type not recognized")
        return -(logp_emb + logp_rules)

    def sample_w(self, n: int) -> np.ndarray:
        w = np.zeros((n, self.num_words + 1), dtype=int)
        w[:, 0] = self.vocab_size  # Add start token
        for i in range(self.num_words):
            trans = self.transition_matrix[w[:, i]]
            w[:, i + 1] = np.array(
                [
                    self.random_state.choice(self.vocab_size, p=trans[i])
                    for i in range(n)
                ]
            )
        w = w[:, 1:]  # Remove start token
        return w

    def sample_z_given_w(self, w: np.ndarray) -> np.ndarray:
        z = self.decode_w(w)
        if self.noise_ratio > 0:
            noise = skellam.approx_gaussian_sample(
                mean=0.0,
                std=self.noise_ratio,
                shape=z.shape,
                precision=self.precision,
                random_state=self.random_state,
            ).astype(np.float32)
            z += noise
        return z

    def logp_w(self, w: np.ndarray) -> np.ndarray:
        return (
            np.ones(w.shape[0], dtype=np.float32)
            * np.log(1 / self.vocab_size)
            * self.num_words
        )

    def logp_z_given_w_and_decoder(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        z_mean = self.decode_w(w)
        noise = z - z_mean
        logp = skellam.approx_gaussian_logpmf(
            x=noise, mean=0.0, std=self.noise_ratio, precision=self.precision
        ).sum(axis=1)
        return logp
