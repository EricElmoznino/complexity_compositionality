from abc import ABC, abstractmethod
from typing import Literal, Any
import numpy as np
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
        return np.random.randint(0, self.vocab_size, size=(n, self.num_words))

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
    CompositionType = Literal["linear"]

    def __init__(
        self,
        z_dim: int,
        num_words: int,
        vocab_size: int,
        num_terminal_pos: int,
        grammar: dict[tuple[int, int], int],
        precision: float = 0.01,
        noise_ratio: float = 0.1,
        composition: CompositionType = "linear",
        random_seed: int = 0,
    ) -> None:
        super().__init__()

        assert np.log2(num_words) % 1 == 0
        assert all(
            [
                (i, j) in grammar
                for i in range(num_terminal_pos)
                for j in range(num_terminal_pos)
            ]
        )
        assert all([k >= num_terminal_pos for k in grammar.values()])

        self.z_dim = z_dim
        self.num_words = num_words
        self.vocab_size = vocab_size
        self.num_terminal_pos = num_terminal_pos
        self.precision = precision
        self.noise_ratio = noise_ratio
        self.composition = composition
        self.random_state = np.random.RandomState(random_seed)

        self.word_embeddings = skellam.approx_gaussian_sample(
            mean=0.0,
            std=1.0,
            shape=(vocab_size, z_dim),
            precision=precision,
            random_state=self.random_state,
        ).astype(np.float32)
        self.word_pos = {i: i % num_terminal_pos for i in range(vocab_size)}
        self.grammar = grammar
        self.rules = {ij: self.sample_rule() for ij in self.grammar}

    def sample_rule(self) -> Any:
        if self.composition == "linear":
            w = skellam.approx_gaussian_sample(
                mean=0.0,
                std=1.0,
                shape=(2 * self.z_dim, self.z_dim),
                precision=self.precision,
                random_state=self.random_state,
            ).astype(np.float32)
            return w
        else:
            raise ValueError("Composition type not recognized")

    def apply_rule(self, x: np.ndarray, y: np.ndarray, rule_params: Any):
        if self.composition == "linear":
            return np.concatenate([x, y]) @ rule_params
        else:
            raise ValueError("Composition type not recognized")

    def decode_w(self, w: np.ndarray) -> np.ndarray:
        z = []
        for wi in w:
            pos = [self.word_pos[word] for word in wi]
            zi = [self.word_embeddings[word] for word in wi]
            while len(zi) > 1:
                pos_next, z_next = [], []
                for i in range(0, len(zi), 2):
                    rule_params = self.rules[(pos[i], pos[i + 1])]
                    out_pos = self.grammar[(pos[i], pos[i + 1])]
                    out_z = self.apply_rule(zi[i], zi[i + 1], rule_params)
                    pos_next.append(out_pos), z_next.append(out_z)
                pos, zi = pos_next, z_next
            z.append(zi[0])
        return np.stack(z)

    @property
    def k_language(self) -> float:
        # Have to specify the bits for two integer numbers that define the uniform distribution, and the rest of p(w) has (small) constant complexity.
        return np.log(self.vocab_size) + np.log(self.num_words)

    @property
    def k_decoder(self) -> float:
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
        else:
            raise ValueError("Composition type not recognized")
        return -(logp_emb + logp_rules)

    def sample_w(self, n: int) -> np.ndarray:
        return np.random.randint(0, self.vocab_size, size=(n, self.num_words))

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
