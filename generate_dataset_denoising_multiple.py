from fktools import *

from walk import walk_circuit_simpler
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh, FakeBrisbane, FakeFez, FakeTorino

from tqdm import tqdm

simulator_backend = AerSimulator()
print(f"using {simulator_backend.name} for simulator backend")

noisy_backends = [FakeMarrakesh(), FakeBrisbane(), FakeFez(), FakeTorino()]
print("\nusing")
for backend in noisy_backends:
    print(backend.backend_name)
print("for noisy backends")
while True:

    # randomizing input
    nodes_power = 3
    num_steps = np.random.randint(0, 24)

    start_bits = ""
    for i in range(nodes_power): start_bits += np.random.choice(['0', '1'])

    coin_phase = np.random.rand() * 2 * np.pi

    noisy_backend = np.random.choice(noisy_backends)

    # running simulations
    simulator_wcs = walk_circuit_simpler(nodes_power=nodes_power, num_steps=num_steps, start_bits=start_bits, coin_phase=coin_phase)
    noisy_wcs =     walk_circuit_simpler(nodes_power=nodes_power, num_steps=num_steps, start_bits=start_bits, coin_phase=coin_phase)
    
    simulator_wcs.build()
    simulator_probs = simulator_wcs.run(simulator=simulator_backend)
    
    noisy_wcs.build()
    noisy_probs = noisy_wcs.run(simulator=noisy_backend)

    # adding to dataset
    X = np.load("dataset_denoising_multiple_X.npz")['arr_0']
    y = np.load("dataset_denoising_multiple_y.npz")['arr_0']

    # X = np.array([noisy_probs])
    # y = np.array([simulator_probs])

    X = np.vstack((X, noisy_probs))
    y = np.vstack((y, simulator_probs))

    np.savez_compressed("dataset_denoising_multiple_X", X)
    np.savez_compressed("dataset_denoising_multiple_y", y)

    print(X.shape, y.shape)