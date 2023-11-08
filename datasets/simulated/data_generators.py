from abc import ABC, abstractmethod
import numpy as np
import scipy
from scipy.stats import skellam


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

    def compositionality(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Returns the compositionality of a representation with our first formula

        Args:
            z (np.ndarray): (n, d) float matrix of representations Z.
            w (np.ndarray): (n, k) integer matrix of sentences W.

        Returns:
            float: The expected per-sample compositionality of a representation.
        """
        return self.k_w_given_language(w, per_sample=True) / self.k_decoder

    def k_z(self, z: np.ndarray, w: np.ndarray, per_sample: bool = False) -> float:
        """Returns the expected Kolmogorov complexity of a representation in nats.

        Args:
            z (np.ndarray): (n, d) float matrix of representations Z.
            w (np.ndarray): (n, k) integer matrix of sentences W.
            per_sample (bool): Whether to return the per-sample or total complexity.

        Returns:
            float: Either the expected per-sample or the total Kolmogorov complexity in nats.
        """
        return (
            self.k_language
            + self.k_w_given_language(w, per_sample=per_sample)
            + self.k_decoder
            + self.k_z_given_w_and_decoder(z, w, per_sample=per_sample)
        )

    def k_w_given_language(self, w: np.ndarray, per_sample: bool = False) -> float:
        """Returns the Kolmogorov complexity of a given W given p(w) in nats.

        Args:
            w (np.ndarray): (n, k) integer matrix of sentences W.

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

        Returns:
            float: Either the expected per-sample or the total Kolmogorov complexity in nats.
        """
        if per_sample:
            return -self.logp_z_given_w(z, w).mean()
        else:
            return -self.logp_z_given_w(z, w).sum()

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
    def logp_z_given_w(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
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


class UniformDataGenerator(SimulatedDataGenerator):
    def __init__(
        self,
        k: int,
        d: int,
        vocab_size: int,
        granularity: float = 0.01,
        noise_scale: float = 0.001,
        random_seed: int = 0,
    ):
        """Generates data in the following way
        - Generate a lookup table of size (vocab_size, int(d/k)) such that we have a representation of dimension int(d/k) for each word
        - Generate a sentence W by:
            - sampling k words uniformly at random from the vocabulary.
            - taking the corresponding representations from the lookup table and concatenating them
            - adding noise sampled from a uniform distribution on [-noise_level, noise_level]

        Args:
            k (int): Number of words in a sentence.
            d (int): Dimensionality of the final representation
            vocab_size (int): Size of the vocabulary.
            granularity (float): Granularity of the lookup table.
            random_seed (int): Random seed for reproducibility
        """
        self.k = k
        self.d = d
        self.vocab_size = vocab_size
        self.max_int = 1 / granularity

        np.random.seed(random_seed)

        self.lookup_table = (
            np.random.randint(2 * self.max_int + 1, size=(vocab_size, int(d / k)))
            / self.max_int
            - 1
        )
        self.noise_scale = noise_scale

    @property
    def k_language(self) -> float:
        # Have to specify the bits for two integer numbers,
        # and the rest of p(w) has (small) constant complexity.
        return np.log(self.vocab_size) + np.log(self.k)

    @property
    def k_decoder(self) -> float:
        # Have to specify the bits for all the numbers in the lookup table,
        # and the rest of p(z | w) has (small) constant complexity.
        return self.lookup_table.size * np.log(2 * self.max_int + 1)

    def sample_w(self, n: int) -> np.ndarray:
        return np.random.randint(0, self.vocab_size, size=(n, self.k))

    def sample_z_given_w(self, w: np.ndarray) -> np.ndarray:
        pure_samples = self.decode_w_perfectly(w)
        noise = (
            skellam.rvs(mu1=0.5, mu2=0.5, size=pure_samples.shape) * self.noise_scale
        )

        return pure_samples + noise

    def decode_w_perfectly(self, w: np.ndarray) -> np.ndarray:
        """Computes and returns the perfect (noiseless) decoding of z given w
        Args:
            w (np.ndarray): (n, k) integer matrix of sentences W.

        Returns:
            np.ndarray: (n, d) float matrix of representations z.
        """

        return np.concatenate(
            [self.lookup_table[w[:, i]] for i in range(self.k)], axis=1
        )

    def logp_w(self, w: np.ndarray) -> np.ndarray:
        return np.log(1 / self.vocab_size) * self.k

    def logp_z_given_w(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        batch_size, _ = z.shape
        means = self.decode_w_perfectly(w)

        int_noise = ((z - means) / self.noise_scale).astype(int)
        return np.array(
            [skellam.logpmf(int_noise[i], mu1=0.5, mu2=0.5) for i in range(batch_size)]
        )
