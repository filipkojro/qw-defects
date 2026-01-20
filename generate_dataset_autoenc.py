from fktools import *

from walk import walk_circuit_simpler
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from tqdm import tqdm

simulator_backend = AerSimulator()
print(f"using {simulator_backend.name} for simulator backend")

# noisy_backend = FakeMarrakesh()
# print(f"\nusing {noisy_backend.backend_name} for noisy backend")

while True:

    # randomizing input
    nodes_power = 3
    num_steps = np.random.randint(0, 24)

    start_bits = ""
    for i in range(nodes_power): start_bits += np.random.choice(['0', '1'])

    coin_phase = np.random.rand() * 2 * np.pi

    # running simulations
    simulator_wcs = walk_circuit_simpler(nodes_power=nodes_power, num_steps=num_steps, start_bits=start_bits, coin_phase=coin_phase)
    # noisy_wcs =     walk_circuit_simpler(nodes_power=nodes_power, num_steps=num_steps, start_bits=start_bits, coin_phase=coin_phase)
    
    simulator_wcs.build()
    simulator_probs = simulator_wcs.run(simulator=simulator_backend)
    
    # noisy_wcs.build()
    # noisy_probs = noisy_wcs.run(simulator=noisy_backend)

    params = np.array([num_steps, int(start_bits, 2), coin_phase], dtype=object)

    # adding to dataset
    X = np.load(f"dataset_autoenc_{nodes_power}_X.npz")['arr_0']
    y = np.load(f"dataset_autoenc_{nodes_power}_y.npz", allow_pickle=True)['arr_0']

    # X = np.array([simulator_probs])
    # y = np.array([params])

    # X = np.vstack((X, noisy_probs))
    X = np.vstack((X, simulator_probs))
    y = np.vstack((y, params))

    np.savez_compressed(f"dataset_autoenc_{nodes_power}_X", X)
    np.savez_compressed(f"dataset_autoenc_{nodes_power}_y", y)

    print(X.shape, y.shape)
