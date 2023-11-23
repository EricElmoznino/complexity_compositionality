import matplotlib.pyplot as plt
from data_generators import UniformDataGenerator
test_data_generator = UniformDataGenerator(k=2, d=2, vocab_size=2, noise_scale=0.1)
test_w, test_z = test_data_generator.sample(100)
plt.scatter(test_z[:, 0], test_z[:, 1], marker="o", s=1)
plt.show()
plt.savefig("test.png")
