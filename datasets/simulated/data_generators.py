from abc import ABC, abstractmethod
from typing import Literal
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
        num_words: int,
        vocab_size: int,
        z_dim: int,
        disentanglement: int,
        precision: float = 0.01,
        noise_ratio: float = 0.1,
        aggregation: AggregationType = "concatenation",
        random_seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_words = num_words
        self.vocab_size = vocab_size
        self.z_dim = z_dim
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


# class LookupTableHierarchicalNonLinearDataGenerator(LookupTableDataGenerator):
#     def __init__(
#         self,
#         k: int,
#         d: int,
#         vocab_size: int,
#         nonlinearity=np.sin,
#         granularity: float = 0.01,
#         noise_scale: float = 0.001,
#         random_seed: int = 0,
#     ):
#         """Generates data in the following way
#         - Generate a lookup table of size (vocab_size, int(k)) such that we have a representation of dimension int(k) for each word
#         - Generate a sentence W by:
#             - sampling k words uniformly at random from the vocabulary.
#             - taking the corresponding representations from the lookup table and adding them
#             - adding noise sampled from a uniform distribution on [-noise_level, noise_level]

#         Args:
#             k (int): Number of words in a sentence.
#             d (int): Dimensionality of the final representation -> same as dimensionality of k in this case
#             vocab_size (int): Size of the vocabulary.
#             granularity (float): Granularity of the lookup table.
#             random_seed (int): Random seed for reproducibility
#         """
#         super().__init__(k, d, vocab_size, granularity, noise_scale, random_seed)
#         self.nonlinearity = nonlinearity

#     @property
#     def k_decoder(self) -> float:
#         # Have to specify the bits for all the numbers in the lookup table,
#         # And then the complexity of the decoding function is the number of layers in the hierarchy (log2(k))
#         # multiplied by the number of operations at each layer (~2 in this case)

#         return (
#             self.lookup_table.size * np.log(2 * self.max_int + 1) + np.log2(self.k) * 2
#         )

#     def get_word_dim(self) -> int:
#         return self.d

#     def one_layer_hierarchy(self, curr_rep: np.ndarray) -> np.ndarray:
#         batch_size, n_words, word_dim = curr_rep.shape
#         curr_rep = curr_rep.reshape(batch_size, 2, int(n_words / 2), word_dim)
#         curr_rep = self.nonlinearity(curr_rep.sum(axis=1))
#         return curr_rep

#     def decode_w_perfectly(self, w: np.ndarray) -> np.ndarray:
#         """Computes and returns the perfect (noiseless) decoding of z given w
#         Args:
#             w (np.ndarray): (n, k) integer matrix of sentences W.

#         Returns:
#             np.ndarray: (n, d) float matrix of representations z.
#         """
#         reps = [self.lookup_table[w[:, i]] for i in range(self.k)]
#         reps = np.array(reps).transpose(1, 0, 2)
#         while reps.shape[1] > 1:
#             reps = self.one_layer_hierarchy(reps)
#         return reps.squeeze()


# data_gen = LookupTableHierarchicalNonLinearDataGenerator(k=32, d=256, vocab_size=10)
# w, z = data_gen.sample(10)
# print(data_gen.k_z(z, w, per_sample=True))
# print(data_gen.compositionality(z, w))
