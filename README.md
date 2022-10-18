# AQCEL
AQCEL (Advancin Quantum Circuits by ICEPP and LBNL) is a python module for optimizing quantum circuits.


## Circuit Optimization
AQCEL aims to remove redundant controlled operations from controlled gates, depending on initial states of the circuit using a quantum computer.

The AQCEL-optimized circuit can produce equivalent final states with much smaller number of gates. Moreover, it efficiently produces a approximated quantum circuit with high fidelity by truncating low-amplitude computational basis states below certain thresholds.Our technique is useful for a wide variety of quantum algorithms, opening up new possibilities to further simplify quantum circuits to be more effective for real devices.


## Getting Started
See a [turorial](https://github.com/UTokyo-ICEPP/aqcel/blob/main/tutorial_aqcel.ipynb).

You must install qiskit=='0.32.1' for using this module. We will try update our codes for the latest version of Qiskit.

AQCEL optimizes quantum circuits by using a quantum computer for polynomial computational resources, however we support demo ideal test using a classical simulation.

※ Some quantum gates cannot be supported in this version. We are trying to fix it.


## Papers
The paper was published in [Quantum](https://doi.org/10.22331/q-2022-09-08-798). ([arXiv : 2209.02322](https://doi.org/10.48550/arXiv.2209.02322))

The proceeding was published in [EPJ Web of Conferences](https://doi.org/10.1051/epjconf/202125103023). ([arXiv : 2102.10008](https://doi.org/10.48550/arXiv.2102.10008))