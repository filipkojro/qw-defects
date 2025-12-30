from fktools import *

from walk import walk_circuit_simpler
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from tqdm import tqdm

simulator_backend = AerSimulator()
print(f"using {simulator_backend.name} for simulator backend")

noisy_backend = FakeMarrakesh()
print(f"using {noisy_backend.backend_name} for noisy backend")

while True:

    # randomizing input
    nodes_power = 3
    num_steps = np.random.randint(0, 24)

    start_bits = ""
    for i in range(nodes_power): start_bits += np.random.choice(['0', '1'])

    coin_phase = np.random.rand() * 2 * np.pi

    # running simulations
    simulator_wcs = walk_circuit_simpler(nodes_power=nodes_power, num_steps=num_steps, start_bits=start_bits, coin_phase=coin_phase)
    noisy_wcs =     walk_circuit_simpler(nodes_power=nodes_power, num_steps=num_steps, start_bits=start_bits, coin_phase=coin_phase)
    
    simulator_wcs.build()
    simulator_probs = simulator_wcs.run(simulator=simulator_backend)
    
    noisy_wcs.build()
    noisy_probs = noisy_wcs.run(simulator=noisy_backend)

    # adding to dataset
    X = np.load("dataset_X.npz")['arr_0']
    y = np.load("dataset_y.npz")['arr_0']

    X = np.vstack((X, noisy_probs))
    y = np.vstack((y, simulator_probs))

    np.savez_compressed("dataset_X", X)
    np.savez_compressed("dataset_y", y)

    print(X.shape, y.shape)