# AQCEL
AQCEL (Advancing Quantum Circuits by ICEPP and LBNL) is a Python module for optimizing quantum circuits.


## Circuit Optimization
AQCEL aims to remove redundant controlled operations from controlled gates, depending on the initial states of the circuit using a quantum computer.

The AQCEL-optimized circuit can produce equivalent final states with a much smaller number of gates. Moreover, it efficiently produces an approximated quantum circuit with high fidelity by truncating low-amplitude computational basis states below certain thresholds. Our technique is useful for a wide variety of quantum algorithms, opening up new possibilities to further simplify quantum circuits to be more effective for real devices.


## Getting Started
See a [tutorial](https://github.com/UTokyo-ICEPP/aqcel/blob/main/tutorial_aqcel.ipynb).

This code is adapted to the following environment.

| qiskit | qiskit-ibm-runtime | qiskit-aer | mthree |
| ------ | ------------------ | ---------- | -------|
| 2.3.0  | 0.45.0             | 0.17.2     | 3.0.0  |


## Papers
The state label manager and redundant CX-pair removal are described in [arXiv:2509.04761](https://doi.org/10.48550/arXiv.2509.04761)

The core part of AQCEL is summarized in this paper: [Quantum](https://doi.org/10.22331/q-2022-09-08-798). ([arXiv : 2209.02322](https://doi.org/10.48550/arXiv.2209.02322))

The proceeding was published in [EPJ Web of Conferences](https://doi.org/10.1051/epjconf/202125103023). ([arXiv : 2102.10008](https://doi.org/10.48550/arXiv.2102.10008))