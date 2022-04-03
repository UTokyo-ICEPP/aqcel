# AQCEL
AQCEL (Advancin Quantum Circuits by ICEPP and LBNL) is a python module for optimizing quantum circuits.

## Circuit Optimization
There is a case where a quantum circuit has been desigined with complete generality in mind, but for a certain initial state the circuit only reaches a select set of intermediate states. In other words, such a circuit has many redundant controlled operations which AQCEL can eliminate automatically. This optimation technique resides in the identification of zero-amplitude computational basis states and determine whether the entire gate or qubit controls can be removed.

## Getting Started
See a [turorial](https://github.com/UTokyo-ICEPP/aqcel/blob/main/tutorial_aqcel.ipynb). We support following types of gates : X, Y, Z, H, RX, RY, RZ, U1, U2, U3, SX, SXdg, T, Tdg, C(X,Y,Z,H,RX,RY,RZ,U1,U2,U3,SX), TOFFOLI, MCU.
AQCEL optimizes quantum circuits by using a quantum computer for polynomial computational resources, however we support demo ideal test using a classical simulation.

※ Some quantum gates cannot be supported in this version. We are trying to fix this bug.

## Paper
You can see details of AQCEL in [arXiv : 2102.10008](https://arxiv.org/abs/2102.10008).