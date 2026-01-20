trying to denoise quantum walk results :)

## important files

- [generate_dataset_denoising_multiple.py](./generate_dataset_denoising_multiple.py) - generator for data used in denoising

- [model_tests.ipynb](./model_tests.ipynb) - (in proggress) tests of ML models

- [generate_dataset_autoenc.py](./generate_dataset_autoenc.py) - generator for data used in autoencoder

- [autoencoder_test.ipynb](./autoencoder_test.ipynb) - (in progress) interesting structure in the encoding of 8 node walk

- [walk.py](./walk.py) - quantum walk circuit builder and runner

- [dist_metric.py](./dist_metric.py) - metric for model accuracy

- [optuna_dm.py](./optuna_dm.py) - optimizing hyperparameters

## requirements
- `python=3.11.14`
- packages from `autorequirements.txt` (automatically generated)

## notes

- [DistributionOverlap](./dist_metric.py) metric for 10000 random distributions each witch 8 nodes gives ≈60% "accuracy" / overlap

## useful links
### quantum walk
- [github with simulations of quantum walk](https://github.com/NMcDowall17/quantum_walk/tree/main)

- [qiskit.circuit.library.CDKMRippleCarryAdder](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CDKMRippleCarryAdder)

- [Modular adder qiskit](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.ModularAdderGate)

- [nice quantum walk description](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.044143)