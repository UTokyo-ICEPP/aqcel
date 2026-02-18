import copy
from qiskit import *
from qiskit.compiler import transpile
from qiskit.circuit.library.standard_gates import XGate
from .. import aqcel
from ..util.transpiler import transpiler
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import CircuitInstruction
from qiskit.transpiler.passes import RemoveBarriers
import mthree

class remove_controlled_operations():    
    #INFO=True
    
    def __init__(self, qc):
        self.qc = qc
        self.skipDict = {}
        self.preGatePos = {}
        self.midGatePos = {}
        self.midGates = {}
        self.preValue = {}
        self.preGate  = {}
        self.bellStateManager = list()
        self.INFO=False

#        self.skipDict = {}

    def updateDict_control_qubits_bell(self, cq0, cq1, fSame):
        if self.INFO: print(f'updateDict_control_qubits_bell: {cq0}, {cq1}, fSame = {fSame}')
        fIsList = False
        for i in range(len(self.bellStateManager)):
            if cq0 in self.bellStateManager[i] and cq1 in self.bellStateManager[i]:
                fIsList = True
                if self.INFO: print(f'updateDict_control_qubits_bell: This case never happend')
            elif cq0 in self.bellStateManager[i]:
                # self.bellStateManager[i][cq0]==fSame indicates True:True or False:False -> cq1 = True
                fIsList = True
                self.bellStateManager[i][cq1] = True if (self.bellStateManager[i][cq0]==fSame) else False
            elif cq1 in self.bellStateManager[i]:
                fIsList = True
                self.bellStateManager[i][cq0] = True if (self.bellStateManager[i][cq1]==fSame) else False
        if fIsList==False:
            self.bellStateManager.append({cq0: True, cq1: True} if fSame==True else {cq0: True, cq1: False})
        if self.INFO: print(f'bellStateManager: {self.bellStateManager}')
            
    def updateDict_control_qubit_status(self, gate, bitstrings):
        if len(bitstrings) == 1:
            if '11' in bitstrings:
                self.skipDict[gate[1][0]] = 1
                self.skipDict[gate[1][1]] = 1
                if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = (1, 1)')
            elif '00' in bitstrings:
                self.skipDict[gate[1][0]] = 0
                self.skipDict[gate[1][1]] = 0
                if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = (0, 0)')
            elif '01' in bitstrings:
                self.skipDict[gate[1][0]] = 1
                self.skipDict[gate[1][1]] = 0
                if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = (1, 0)')
            elif '10' in bitstrings:
                self.skipDict[gate[1][0]] = 0
                self.skipDict[gate[1][1]] = 1
                if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = (0, 1)')
                
        elif len(bitstrings) == 2:
            if '11' in bitstrings and '00' in bitstrings:
                if (gate[1][0] in self.skipDict and gate[1][1] in self.skipDict) and (self.skipDict[gate[1][0]] == 2 and self.skipDict[gate[1][1]] == 2):
                    # Merge two bell group
                    for i in range(len(self.bellStateManager)):
                        if gate[1][0] in self.bellStateManager[i]:
                            for j in range(len(self.bellStateManager)):
                                if gate[1][1] in self.bellStateManager[j]:
                                    if i == j:
                                        if self.INFO: print(f' --- updateDict_control_qubit_status: This case should never happened')
                                    else:
                                        fSame = True if (self.bellStateManager[i][gate[1][0]]==self.bellStateManager[j][gate[1][1]]) else False
                                        for k in self.bellStateManager[j].keys():
                                            self.bellStateManager[i][k] = self.bellStateManager[j][k] if fSame==True else not self.bellStateManager[j][k]
                                        self.bellStateManager.pop(j)
                                        break
                            break
                else:
                    self.skipDict[gate[1][0]] = 2
                    self.skipDict[gate[1][1]] = 2
                    self.updateDict_control_qubits_bell(gate[1][0], gate[1][1], True)
                    if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = 00, 11')
            elif '11' in bitstrings and '01' in bitstrings:
                self.skipDict[gate[1][0]] = 1
                self.skipDict[gate[1][1]] = -1
                if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = (1, -1)')
            elif '11' in bitstrings and '10' in bitstrings:
                self.skipDict[gate[1][0]] = -1
                self.skipDict[gate[1][1]] = 1
                if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = (-1, 1)')
            elif '10' in bitstrings and '00' in bitstrings:
                self.skipDict[gate[1][0]] = 0
                self.skipDict[gate[1][1]] = -1
                if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = (0, -1)')
            elif '10' in bitstrings and '01' in bitstrings:
                if (gate[1][0] in self.skipDict and gate[1][1] in self.skipDict) and (self.skipDict[gate[1][0]] == 2 and self.skipDict[gate[1][1]] == 2):
                    # Merge two bell group
                    for i in range(len(self.bellStateManager)):
                        if gate[1][0] in self.bellStateManager[i]:
                            for j in range(len(self.bellStateManager)):
                                if gate[1][1] in self.bellStateManager[j]:
                                    if i == j:
                                        if self.INFO: print(f' --- updateDict_control_qubit_status: This case should never happened')
                                    else:
                                        fSame = True if (self.bellStateManager[i][gate[1][0]]==self.bellStateManager[j][gate[1][1]]) else False
                                        for k in self.bellStateManager[j].keys():
                                            self.bellStateManager[i][k] = not self.bellStateManager[j][k] if fSame==True else self.bellStateManager[j][k]
                                        self.bellStateManager.pop(j)
                                        break
                            break
                else:
                    self.skipDict[gate[1][0]] = 2
                    self.skipDict[gate[1][1]] = 2
                    self.updateDict_control_qubits_bell(gate[1][0], gate[1][1], False)
                    if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = 10, 01')
            elif '01' in bitstrings and '00' in bitstrings:
                self.skipDict[gate[1][0]] = -1
                self.skipDict[gate[1][1]] = 0
                if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = (-1, 0)')
        else:
            if gate[1][0] in self.skipDict and self.skipDict[gate[1][0]] == 2:
                pass
            else:
                self.skipDict[gate[1][0]] = -1

            if gate[1][1] in self.skipDict and self.skipDict[gate[1][1]] == 2:
                pass
            else:
                self.skipDict[gate[1][1]] = -1
            if self.INFO: print(f' ---  updateDict_control_qubit_status: ({gate[1][0]}, {gate[1][1]}) = ({self.skipDict[gate[1][0]]}, {self.skipDict[gate[1][1]]})')    

    
    def CXLadders_Removal(self, new_qc, gate):
        # remove CX-ladders
        fDoRemoval = True
        GatePos = len(new_qc) - 1
        c_qubit = new_qc.data[-1][1][0]
        t_qubit = new_qc.data[-1][1][-1]
        if self.INFO: print(f'start CXLadders_Removal, GatePos = {GatePos}, last gate = {new_qc.data[-1][1]}')
    
        if len(new_qc.data[-1][1]) > 2:
            fDoRemoval = False
            
        if (GatePos - self.preGatePos[gate[1][-1]]) > 100:
            fDoRemoval = False
            
        for i in range(self.preGatePos[gate[1][-1]]+1, GatePos):
            if new_qc.data[i][0].name=='barrier':
                continue
            
            if new_qc.data[i][1][-1]==c_qubit or new_qc.data[i][1][-1]==t_qubit:
                fDoRemoval = False
                if self.INFO: print(f'CXLadders_Removal: This case should never happened')
    
        if fDoRemoval:
            for i in range(self.preGatePos[gate[1][-1]]+1, GatePos):
                if new_qc.data[i][0].name=='barrier':
                    continue
                
                if t_qubit in new_qc.data[i][1]:
                    if c_qubit in new_qc.data[i][1]: # c_qubit and t_qubit are control qubits, we can remove t_qubit
                        new_qc.data.insert(i, CircuitInstruction(XGate(label=None).control(1), [c_qubit, new_qc.data[i][1][-1]], []))
                        if self.INFO: print(f' CXLadders_Removal (Add gate) : {new_qc.data[i]}')
                        if self.INFO: print(f' CXLadders_Removal (Remove gate) : {new_qc.data[i+1]}')
                        new_qc.data.pop(i+1)
                    else:
                        new_qc.data.insert(i, CircuitInstruction(new_qc.data[i][0], [c_qubit if x==t_qubit else x for x in new_qc.data[i][1]], []))
                        if self.INFO: print(f' CXLadders_Removal (Add gate) : {new_qc.data[i]}')
                        if self.INFO: print(f' CXLadders_Removal (Remove gate) : {new_qc.data[i+1]}')
                        new_qc.data.pop(i+1)
            
            if self.INFO: print(f' CXLadders_Removal (Remove gate) : {new_qc.data[GatePos]}')
            if self.INFO: print(f' CXLadders_Removal (Remove gate) : {new_qc.data[self.preGatePos[gate[1][-1]]]}')
            new_qc.data.pop(GatePos)
            new_qc.data.pop(self.preGatePos[gate[1][-1]])
            for i in self.preGatePos.keys():
                if self.preGatePos[i] > self.preGatePos[gate[1][-1]]:
                    self.preGatePos[i] -= 1
                    if self.INFO: print(f' CXLadders_Removal (Reset preGatePos) : {i}')

        else:
            pass    
#            self.skipDict.pop(gate[1][-1], None)
        if self.INFO: print(f' after removal gate pos = {len(new_qc) - 1}')
        self.preValue.pop(gate[1][-1], None)
        self.preGate.pop(gate[1][-1], None)
        self.preGatePos.pop(gate[1][-1], None)
        # temporary
        #self.skipDict[gate[1][-1]] = 0

    
    def updateDict_cx(self, new_qc, gate):
        if self.INFO: print(f'updateDict_cx, skipDict= {self.skipDict[gate[1][-1]]}')
        if self.skipDict[gate[1][-1]] == -1:
            if gate[1][-1] in self.preGate and self.preGate[gate[1][-1]] == gate:
                GatePos = len(new_qc) - 1
                if gate[1][-1] in self.preGate:
                    for i in range(self.preGatePos[gate[1][-1]]+1, GatePos):
                        if new_qc.data[i][0].name=='barrier':
                            continue
                        if new_qc.data[i][1][-1] in gate[1]:
                            if self.INFO: print(f' --- target = {new_qc.data[i][1]}, in {gate[1]}, i = {i}')
                            self.preValue.pop(gate[1][-1], None)
                            self.preGate.pop(gate[1][-1], None)
                            self.preGatePos.pop(gate[1][-1], None)                    
            
            if gate[1][-1] in self.preGate and self.preGate[gate[1][-1]] == gate:
                self.skipDict[gate[1][-1]] = self.preValue[gate[1][-1]]
                if self.INFO: print(f' --- set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
                if self.preValue[gate[1][-1]] == 2: # We do not implement proper action at this moment
                    self.skipDict.pop(gate[1][-1], None)
                    if self.INFO: print(f'remove {gate[1][-1]} from dict 1')
                if self.preValue[gate[1][-1]] == 0:
                    self.CXLadders_Removal(new_qc, gate)
                self.preValue.pop(gate[1][-1], None)
                self.preGate.pop(gate[1][-1], None)
                self.preGatePos.pop(gate[1][-1], None)
            else:
                self.preGate[gate[1][-1]] = gate
                self.preGatePos[gate[1][-1]] = len(new_qc.data) - 1
                self.preValue[gate[1][-1]] = self.skipDict[gate[1][-1]]
                self.skipDict[gate[1][-1]] = -1
                if self.INFO: print(f' --- set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
        else:
            if self.INFO: print(f' --- hoge 1 self.preGate = {gate}, Preself.skipDict= {self.skipDict[gate[1][-1]]}')
            self.preGate[gate[1][-1]] = gate
            self.preGatePos[gate[1][-1]] = len(new_qc.data) - 1
            self.preValue[gate[1][-1]] = self.skipDict[gate[1][-1]]
            self.skipDict[gate[1][-1]] = -1
            if self.INFO: print(f' --- hoge 1 set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')

        
    def updateDict_bell(self, new_qc, gate, pos_ctrl_qubit):
        if gate[1][-1] in self.skipDict:
            if self.skipDict[gate[1][-1]] == -1:
                if gate[1][-1] in self.preGate and self.preGate[gate[1][-1]] == gate:
                    GatePos = len(new_qc) - 1
                    if gate[1][-1] in self.preGate:
                        for i in range(self.preGatePos[gate[1][-1]]+1, GatePos):
                            if new_qc.data[i][0].name=='barrier':
                                continue                            
                            if new_qc.data[i][1][-1] in gate[1]:
                                if self.INFO: print(f' --- target = {new_qc.data[i][1]}, in {gate[1]}, i = {i}')
                                self.preValue.pop(gate[1][-1], None)
                                self.preGate.pop(gate[1][-1], None)
                                self.preGatePos.pop(gate[1][-1], None)
                                break
                if gate[1][-1] in self.preGate and self.preGate[gate[1][-1]] == gate:
                    if self.preValue[gate[1][-1]] == 2: # different bell-state group
                        # We do not implement proper action at this moment
                        # self.skipDict[gate[1][-1]] = self.preValue[gate[1][-1]]
                        self.skipDict.pop(gate[1][-1], None)
                        self.preValue.pop(gate[1][-1], None)
                        self.preGate.pop(gate[1][-1], None)
                        self.preGatePos.pop(gate[1][-1], None)
                        if self.INFO: print(f'remove {gate[1][-1]} from dict 2')
                        # add gate[1][-1] to the bell dictionary
                    else: # self.preValue[gate[1][-1]] == -1
                        self.skipDict[gate[1][-1]] = self.preValue[gate[1][-1]]
                        self.preValue.pop(gate[1][-1], None)
                        self.preGate.pop(gate[1][-1], None)
                        self.preGatePos.pop(gate[1][-1], None)
                        if self.INFO: print(f'set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
                else:
                    self.preValue[gate[1][-1]] = -1
                    self.preGate[gate[1][-1]] = gate
                    self.preGatePos[gate[1][-1]] = len(new_qc.data) - 1
            elif self.skipDict[gate[1][-1]] == 2:
                if gate[1][-1] in self.preGate and self.preGate[gate[1][-1]] == gate:
                    GatePos = len(new_qc) - 1
                    if gate[1][-1] in self.preGate:
                        for i in range(self.preGatePos[gate[1][-1]]+1, GatePos):
                            if new_qc.data[i][0].name=='barrier':
                                continue                            
                            if new_qc.data[i][1][-1] in gate[1]:
                                if self.INFO: print(f' --- target = {new_qc.data[i][1]}, in {gate[1]}, i = {i}')
                                self.preValue.pop(gate[1][-1], None)
                                self.preGate.pop(gate[1][-1], None)
                                self.preGatePos.pop(gate[1][-1], None)
                                break
                if gate[1][-1] in self.preGate and self.preGate[gate[1][-1]] == gate:
                    if self.preValue[gate[1][-1]] == 0:
                        self.skipDict[gate[1][-1]] = self.preValue[gate[1][-1]]
                        if self.INFO: print(f'set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
                        for i in range(len(self.bellStateManager)):
                            if gate[1][-1] in self.bellStateManager[i]:
                                self.bellStateManager[i].pop(gate[1][-1])
                        self.CXLadders_Removal(new_qc, gate)
                    elif self.preValue[gate[1][-1]] == 1:
                        self.skipDict[gate[1][-1]] = self.preValue[gate[1][-1]]
                        self.preValue.pop(gate[1][-1], None)
                        self.preGate.pop(gate[1][-1], None)
                        self.preGatePos.pop(gate[1][-1], None)
                        if self.INFO: print(f'set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
                        for i in range(len(self.bellStateManager)):
                            if gate[1][-1] in self.bellStateManager[i]:
                                self.bellStateManager[i].pop(gate[1][-1])                                    
                    else:
                        self.skipDict[gate[1][-1]] = self.preValue[gate[1][-1]]
                        self.preValue.pop(gate[1][-1], None)
                        self.preGate.pop(gate[1][-1], None)
                        self.preGatePos.pop(gate[1][-1], None)
                        if self.INFO: print(f'judge: This case never happend')
                else:
                    fSameBell = False
                    for i in range(len(self.bellStateManager)):
                        if (gate[1][pos_ctrl_qubit] in self.bellStateManager[i]) and (gate[1][-1] in self.bellStateManager[i]):
                            self.skipDict[gate[1][-1]] = 0 if (self.bellStateManager[i][gate[1][pos_ctrl_qubit]] == self.bellStateManager[i][gate[1][-1]]) else 1
                            self.bellStateManager[i].pop(gate[1][-1])
                            if self.INFO: print(f'set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
                            fSameBell = True
                            self.preValue.pop(gate[1][-1], None)
                            self.preGate.pop(gate[1][-1], None)
                            self.preGatePos.pop(gate[1][-1], None)
                            break
                    if fSameBell==False:
                        # Since original state was different bell state group, it becomes -1
                        self.skipDict[gate[1][-1]] = -1
                        self.preValue.pop(gate[1][-1], None)
                        self.preGate.pop(gate[1][-1], None)
                        self.preGatePos.pop(gate[1][-1], None)
                        if self.INFO: print(f'set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
            else: # {00, 11}{0} -> {000, 111} or {00, 11}{1} -> {001, 110}
                self.preGate[gate[1][-1]] = gate
                self.preGatePos[gate[1][-1]] = len(new_qc.data) - 1
                self.preValue[gate[1][-1]] = self.skipDict[gate[1][-1]]
                self.updateDict_control_qubits_bell(gate[1][pos_ctrl_qubit], gate[1][-1], True if self.skipDict[gate[1][-1]]==0 else False)
                self.skipDict[gate[1][-1]] = 2
                if self.INFO: print(f' --- set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')                        
        else:
            pass
        
    def run_cc(self):
        
        bistrings_dict = self.measure_bitstrings_cc() # Dictionary of bitstrings on controlled qubits
        
        new_qc = QuantumCircuit(*self.qc.qregs, *self.qc.cregs)
        
        for index, gate in enumerate(self.qc):
            
            if indeox in bistrings_dict: # Check 'gate' is a controlled gate or not
                bitstrings = bistrings_dict[index]
                
                new_qc = self.judge(new_qc, gate, bitstrings)
                
            else:
                new_qc.append(gate[0],gate[1],gate[2])
                
        return new_qc
    
    
    # Measure a list of bitstrings on controlled qubits by a statevector simulator
    def measure_bitstrings_cc(self):
        
        new_qc = QuantumCircuit(*self.qc.qregs, *self.qc.cregs)
        info_list = []

        for index, gate in enumerate(self.qc):
            if (len(gate[1]) > 1) and (gate[0].name != 'barrier'): # Check 'gate' is a controlled gate or not
                label = f'{index}'
                ctrl_qubits = self.ctrl_qubits_indices(gate)
                info_list.append([label,ctrl_qubits])

                new_qc.save_statevector(label)
                new_qc.append(gate[0],gate[1],gate[2])

            else:
                new_qc.append(gate[0],gate[1],gate[2])

        # Transpile for simulator
        simulator = Aer.get_backend('aer_simulator')
        transpiled_qc = transpile(new_qc, simulator)

        # Run and get saved data
        result = simulator.run(transpiled_qc).result()
        data = result.data()

        bitstrings_dict = {}

        for label, ctrl_qubits in info_list:

            array = self.data_to_array(label, data)
            probabilities = self.array_to_probabilities(array)
            bitstrings = self.basis_to_bitstring(probabilities, ctrl_qubits)
            bitstrings = self.get_unique_list(bitstrings)
            bitstrings_dict[int(label)] = bitstrings

        return bitstrings_dict


    def ctrl_qubits_indices(self, gate):

        ctrl_qubits = []

        qregs = gate[1]
        #num_ctrl_qubits = gate[0].num_ctrl_qubits
        num_ctrl_qubits = len(gate[1]) - 1
        q_controls = qregs[:num_ctrl_qubits]

        for qubit in q_controls:
            index = self.qc.qubits.index(qubit)
            ctrl_qubits.append(index)

        return ctrl_qubits


    @staticmethod
    def data_to_array(label, data):

        return data[label]


    @staticmethod
    def array_to_probabilities(array, threshold=1e-10):

        statevector = Statevector(array)
        probabilities = statevector.probabilities_dict()

        for key in list(probabilities.keys()):
            if probabilities[key] < threshold: # Cut off a floating point numerical error
                probabilities.pop(key)

        return probabilities

    
    # Obtain bitstrings on controlled qubits
    @staticmethod
    def basis_to_bitstring(probabilities, ctrl_qubits):

        bitstring_list = []

        for key in probabilities.keys():
            bitstring = ''

            for qubit in reversed(ctrl_qubits): # "reversed" needs for the order of bitstrings
                bitstring = bitstring + key[-qubit-1]

            bitstring_list.append(bitstring)

        return bitstring_list
    
    
    @staticmethod
    def get_unique_list(seq):
        seen = []
        return [x for x in seq if x not in seen and not seen.append(x)]
    
    
    #    def judge(new_qc, gate, bitstrings, self.skipDict={}, self.preValue={}, self.preGate={}, fUpdate=True):
#    @staticmethod
    def judge(self, new_qc, gate, bitstrings, fUpdate=True):
        target_qubit = gate[1][-1]
        
        """
        gate[0] : gate class
        gate[1] : qregs
        gate[2] : cregs
        """

        # Toffoli gates
        if gate[0].name in ['ccx', 'rccx']:
            # States of control qubits has been measured and update the status of control qubits
            if fUpdate:
                self.updateDict_control_qubit_status(gate, bitstrings)
                
            if '11' in bitstrings:
                if len(bitstrings) == 1:
                    new_qc.append(XGate(label=None), [target_qubit], gate[2])
                    if target_qubit in self.skipDict:
                        if self.skipDict[target_qubit] == -1: pass
                        elif self.skipDict[target_qubit] == 0: self.skipDict[target_qubit] = 1
                        elif self.skipDict[target_qubit] == 1: self.skipDict[target_qubit] = 0
                        else:
                            for i in range(len(self.bellStateManager)):
                                if target_qubit in self.bellStateManager[i]:
                                    self.bellStateManager[i][target_qubit] = not self.bellStateManager[i][target_qubit]
                        if self.INFO: print(f'set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
                elif len(bitstrings) == 2 and '00' in bitstrings: # {00, 11}
                    new_qc.append(XGate(label=None).control(1), [gate[1][1], target_qubit], gate[2])
                    self.updateDict_bell(new_qc, gate, 1)
#                elif len(bitstrings) == 2 and '10' not in bitstrings:
                elif '10' not in bitstrings:
                    new_qc.append(XGate(label=None).control(1), [gate[1][1], target_qubit], gate[2])
                    if gate[1][1] in self.skipDict and self.skipDict[gate[1][1]] == 2:
                        # if self.skipDict[gate[1][1]] == 2, same treatment as {00, 11} case
                        self.updateDict_bell(new_qc, gate, 1)
                    elif target_qubit in self.skipDict:
                        self.updateDict_cx(new_qc, gate)
                    else:
                        self.skipDict[target_qubit] = -1
                        if self.INFO: print(f' --- set {target_qubit} = {self.skipDict[target_qubit]}')
#                elif len(bitstrings) == 2 and '01' not in bitstrings:
                elif '01' not in bitstrings:
                    new_qc.append(XGate(label=None).control(1), [gate[1][0], target_qubit], gate[2])
                    if gate[1][0] in self.skipDict and self.skipDict[gate[1][0]] == 2:
                        # if self.skipDict[gate[1][0]] == 2, same treatment as {00, 11} case
                        self.updateDict_bell(new_qc, gate, 0)
                    elif target_qubit in self.skipDict:
                        self.updateDict_cx(new_qc, gate)
                    else:
                        self.skipDict[target_qubit] = -1
                        if self.INFO: print(f' --- set {target_qubit} = {self.skipDict[target_qubit]}')

                else: #11と00の時だけようにもう一つ作る？(Qubit connectionを一つ減らせる)
                    new_qc.append(gate[0],gate[1],gate[2]) # CCX
                    if target_qubit in self.skipDict:
                        self.updateDict_cx(new_qc, gate)
                    else:
                        if self.INFO: print(f'skipDict = {self.skipDict}')
                        if self.INFO: print(f'preValue = {self.preValue}')
                        self.skipDict[target_qubit] = -1
                        if self.INFO: print(f' --- set14 {target_qubit} = {self.skipDict[target_qubit]}')
            elif len(bitstrings) == 2 and '01' in bitstrings and '10' in bitstrings:
                pass
            else:
                pass # Delete CCX

        # Two-qubit gates
        else:
            if fUpdate:
                if len(bitstrings) == 1:
                    self.skipDict[gate[1][0]] = 0 if '0' in bitstrings else 1
                    if self.INFO: print(f' --- set {gate[1][0]} = {self.skipDict[gate[1][0]]}')
                else:
                    pass
                    #self.skipDict[gate[1][0]] = -1
                    #print(f' --- set {gate[1][0]} = {self.skipDict[gate[1][0]]}')

            if '1' in bitstrings:
                if '0' not in bitstrings:
                    new_qc.append(gate.operation.base_gate, [target_qubit], gate[2])
                    if gate[0].name == 'cx':
                        if target_qubit in self.skipDict:
                            if self.skipDict[target_qubit] == -1: pass
                            elif self.skipDict[target_qubit] == 0: self.skipDict[target_qubit] = 1
                            elif self.skipDict[target_qubit] == 1: self.skipDict[target_qubit] = 0
                            else:
                                for i in range(len(self.bellStateManager)):
                                    if target_qubit in self.bellStateManager[i]:
                                        self.bellStateManager[i][target_qubit] = not self.bellStateManager[i][target_qubit]
                            if self.INFO: print(f'set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
                    else:
                        self.skipDict.pop(target_qubit, None)
                        self.preValue.pop(target_qubit, None)
                        self.preGate.pop(target_qubit, None)
                        self.preGatePos.pop(target_qubit, None)
                        if self.INFO: print(f' --- remove {target_qubit} from dictionary 1')

                else: # 1 and 0
                    new_qc.append(gate[0],gate[1],gate[2])
                    if gate[0].name == 'cx':
                        if gate[1][0] in self.skipDict and self.skipDict[gate[1][0]] == 2:
                            self.updateDict_bell(new_qc, gate, 0)
                        elif target_qubit in self.skipDict:
                            self.updateDict_cx(new_qc, gate)
                        else:
                            pass
                            #self.skipDict[target_qubit] = -1
                            #print(f' --- set {target_qubit} = {self.skipDict[target_qubit]}')
                    else:
                        self.skipDict.pop(target_qubit, None)
                        self.preValue.pop(target_qubit, None)
                        self.preGate.pop(target_qubit, None)
                        self.preGatePos.pop(target_qubit, None)
                        if self.INFO: print(f' --- remove {target_qubit} from dictionary 2')
            else:
                pass # Delete CU
            
        return new_qc
    
    
#    def run_qc(self, shots, backend, backend_tket, threshold_type, zne, fSkip=False):
    def run_qc(self, session, sampler, shots, backend, threshold_type, fSkip=True):
        new_qc = QuantumCircuit(*self.qc.qregs)

#        mit2 = mthree.M3Mitigation()
#        mit2.cals_from_file('my_cals.json')
#        print(f'mthree : {mit2.single_qubit_cals}')
        
        rccx_qregs, rccx_insteads = [], [] # List of rccxs of a former part of s decomposed mcu gate

#        self.skipDict = {}
        num_meas = 0
        bitstrings = {}
        if self.INFO: print(f'Info = {self.INFO}')
        for ireg in self.qc.qregs:
            for qubit in ireg:
                self.skipDict[qubit] = 0
                if self.INFO: print(f' set {qubit} = 0')
                self.midGates[qubit] = list()

        for gate in self.qc:
            if self.INFO: print(f'  skipDict  = {self.skipDict}')
            if self.INFO: print(f'  preGate   = {self.preGate}')
            if self.INFO: print(f'  preGatePos= {self.preGatePos}')
            if self.INFO: print(f'  preValue  = {self.preValue}')
            if self.INFO: print(f'  bellState = {self.bellStateManager}')
            

            if self.INFO: print(gate)
#            for i in range(len(new_qc)):
#                print(f' Pos = {i}, gate = {new_qc.data[i]}')

            # Measure a list of bitstrings on controlled qubits by a quantum computer
            if (len(gate[1]) > 1) and (gate[0].name != 'barrier'): # Check 'gate' is a controlled gate or not

                if (gate[0].name == 'rccx') and (gate[1] in rccx_qregs):
                    alter_gate = rccx_insteads[-1]
                    if alter_gate == None:
                        pass
                    else:
                        new_qc.append(alter_gate[0],alter_gate[1],alter_gate[2])
                        if len(alter_gate[1])==1: # X gate
                            if gate[1][-1] in self.skipDict:
                                if self.skipDict[gate[1][-1]] == -1: pass
                                elif self.skipDict[gate[1][-1]] == 0: self.skipDict[gate[1][-1]] = 1
                                elif self.skipDict[gate[1][-1]] == 1: self.skipDict[gate[1][-1]] = 0
                                else:
                                    for i in range(len(self.bellStateManager)):
                                        if gate[1][-1] in self.bellStateManager[i]:
                                            self.bellStateManager[i][gate[1][-1]] = not self.bellStateManager[i][gate[1][-1]]
                                if self.INFO: print(f'set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
#                                    self.midGates[gate[1][-1]] = list()
                                    # CX ゲートの場合同じ CX が二回出てきたときの扱いが難しい
                        elif len(alter_gate[1])==2: # CX gate
                            if self.skipDict[gate[1][0]]==2 or self.skipDict[gate[1][1]]==2:
                                self.updateDict_bell(new_qc, gate, 0 if self.skipDict[gate[1][0]]==2 else 1)
                            elif gate[1][-1] in self.skipDict:
                                self.updateDict_cx(new_qc, gate)
                                        #                                        for midGate in self.midGates[]:
                                        #                                            if len(midGate[1])==2:
                                        #                                                if midGate[1][0] == gate[1][-1]:
                            else:
                                self.skipDict[gate[1][-1]] = -1
                                if self.INFO: print(f' --- 4set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
                        elif len(alter_gate[1])==3: # CCX gate
                            if gate[1][-1] in self.skipDict:
                                self.updateDict_cx(new_qc, gate)
                            else:
                                self.skipDict[gate[1][-1]] = -1
                                if self.INFO: print(f' --- 5set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')

#                                    print(f' --- 3set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')                                    
#                                    self.preGate[gate[1][-1]] = gate
#                                    self.preValue[gate[1][-1]] = self.skipDict[gate[1][-1]]
#                                    self.skipDict[gate[1][-1]] = -1
#                                    print(f' --- 3set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')


#                            self.skipDict.pop(gate[1][-1], None)
#                            print(f' --- remove {gate[1][-1]} from dictionary')

                    rccx_qregs.pop(-1)
                    rccx_insteads.pop(-1)

                else: # gate is not rccx
                    num_ctrl_qubits = len(gate[1]) - 1

                    fInDict = False
                    if fSkip:
                        fInDict=True
                        ctrl_qubits = gate[1][:num_ctrl_qubits]
                        for ctrl_qubit in ctrl_qubits:
                            if ctrl_qubit not in self.skipDict:
                                fInDict=False

                    if fInDict and num_ctrl_qubits == 2:
                        if (self.skipDict[gate[1][0]] == -1 or self.skipDict[gate[1][0]] == 2) and (self.skipDict[gate[1][1]] == -1 or self.skipDict[gate[1][1]] == 2):
                            fInDict = False
                            if self.skipDict[gate[1][0]] == 2 and self.skipDict[gate[1][1]] == 2:
                                fSameBell = False
                                for i in range(len(self.bellStateManager)):
                                    if (gate[1][0] in self.bellStateManager[i] and gate[1][1] in self.bellStateManager[i]):
                                        fSameBell = True
                                if fSameBell == True:
                                    fInDict = True
                    
                    if fInDict:
                        bitstrings.clear()
                        if self.INFO: print(f'All control qubits in dictionary, ctrl = {num_ctrl_qubits}')
                        if num_ctrl_qubits == 2:
                            if self.INFO: print(f'value1 = {self.skipDict[gate[1][0]]}, value2 = {self.skipDict[gate[1][1]]}')
                            list_bit0, list_bit1 = [], []

                            if self.skipDict[gate[1][0]] == 2 and self.skipDict[gate[1][1]] == 2:
                                fSameBell = False
                                for i in range(len(self.bellStateManager)):
                                    if (gate[1][0] in self.bellStateManager[i] and gate[1][1] in self.bellStateManager[i]):
                                        fSameBell = True
                                        if self.bellStateManager[i][gate[1][0]] == self.bellStateManager[i][gate[1][1]]:
                                            bitstrings['00'] = 100
                                            bitstrings['11'] = 100
                                        else:
                                            bitstrings['01'] = 100
                                            bitstrings['10'] = 100

                                if fSameBell==False:
                                    bitstrings['00'] = 100
                                    bitstrings['01'] = 100
                                    bitstrings['10'] = 100
                                    bitstrings['11'] = 100
                            else:
                                if self.skipDict[gate[1][0]] == -1 or self.skipDict[gate[1][0]] == 2:
                                    list_bit0.append(0)
                                    list_bit0.append(1)
                                else:
                                    list_bit0.append(self.skipDict[gate[1][0]])
                                if self.skipDict[gate[1][1]] == -1 or self.skipDict[gate[1][1]] == 2:
                                    list_bit1.append(0)
                                    list_bit1.append(1)
                                else:
                                    list_bit1.append(self.skipDict[gate[1][1]])
                                for bit0 in list_bit0:
                                    for bit1 in list_bit1:
                                        bitstrings[str(bit1)+str(bit0)] = 100
                                        if self.INFO: print(f'    --- bit1 = {bit1}, bit2 = {bit0}')

                        elif num_ctrl_qubits == 1:
                            if self.skipDict[gate[1][0]] == -1 or self.skipDict[gate[1][0]] == 2:
                                bitstrings['0'] = 100
                                bitstrings['1'] = 100
                            else:
                                bitstrings[str(self.skipDict[gate[1][0]])] = 100
                        copy_qc = copy.deepcopy(new_qc)
                        if self.INFO: print(f' bitstrings = {bitstrings}')
                        new_qc = self.judge(new_qc, gate, bitstrings, fUpdate=False)
                    else:
                        c = ClassicalRegister(num_ctrl_qubits, 'c')
                        new_qc.add_register(c)
                        new_qc.barrier()

                        for n in range(num_ctrl_qubits):
                            new_qc.measure(gate[1][n],c[n])

                        if self.INFO: print(f'num_ctrl_qubits={num_ctrl_qubits}')
#                        print(new_qc.draw(fold=200, vertical_compression='high'))

                        threshold  = threshold_type[1]
                        num_meas += 1
                        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
                        isa_circ = pm.run(RemoveBarriers()(new_qc))
#                        isa_circ = pm.run(new_qc)
#                        print(isa_circ.draw(fold=200, idle_wires=False))
                        #                        job = sampler.run([(isa_circ,)])
                        job = sampler.run([(isa_circ)], shots=shots)
                        print(f">>> Job ID: {job.job_id()}")
                        print(f">>> Job Status: {job.status()}")
                        result = job.result()
                        if self.INFO: print(result)

                        raw_counts = result[0].data.c.get_counts()
                        #bitstrings = None

                        if backend.name=='aer_simulator':
                            bitstrings = {i: raw_counts[i]/shots for i in raw_counts.keys()}
                            print(f'Prob : {bitstrings}')
                        else:
                            mit = mthree.M3Mitigation(backend)
                            mapping = mthree.utils.final_measurement_mapping(isa_circ)
                            mit.cals_from_system(mapping, len(mapping)*shots, runtime_mode=session)
                            bitstrings = mit.apply_correction(raw_counts, mapping)
                            print(f'Raw count: {raw_counts}')
                            print(f'Prob (M3): {bitstrings}')

                        if self.INFO: print(result)

                        for key in list(bitstrings.keys()):
                            if bitstrings[key] < threshold: # Cut-off bitstrings below a threshold
                                bitstrings.pop(key)
                                
                        print('Threshold :',threshold)
                        print('Final counts after applying the threshold :', bitstrings)

                        # Remove measurements and all clibits
                        new_qc.remove_final_measurements()
                        new_qc.barrier()

                        copy_qc = copy.deepcopy(new_qc)

                        new_qc = self.judge(new_qc, gate, bitstrings)

                    if gate[0].name == 'rccx':
                        rccx_qregs.append(gate[1])
                        if copy_qc == new_qc:
                            rccx_insteads.append(None)
                        else:
                            rccx_insteads.append(new_qc[-1])

            elif (gate[0].name == 'measure') and (new_qc.cregs == []): # No clibits (First measure)
                new_qc.add_register(*self.qc.cregs)
                new_qc.append(gate[0],gate[1],gate[2])
            # single qubit operator
            else:
                new_qc.append(gate[0],gate[1],gate[2])
                if gate[0].name == 'x':
                    if gate[1][-1] in self.skipDict:
                        if self.skipDict[gate[1][-1]] == -1: pass
                        elif self.skipDict[gate[1][-1]] == 0: self.skipDict[gate[1][-1]] = 1
                        elif self.skipDict[gate[1][-1]] == 1: self.skipDict[gate[1][-1]] = 0
                        else:
                            for i in range(len(self.bellStateManager)):
                                if gate[1][-1] in self.bellStateManager[i]:
                                    self.bellStateManager[i][gate[1][-1]] = not self.bellStateManager[i][gate[1][-1]]
                        if self.INFO: print(f'set {gate[1][-1]} = {self.skipDict[gate[1][-1]]}')
                else:
                    if gate[1][-1] in self.skipDict and self.skipDict[gate[1][-1]] == 2:
                        for i in range(len(self.bellStateManager)):
                            if gate[1][-1] in self.bellStateManager[i]:
                                self.bellStateManager[i].pop(gate[1][-1])
                    self.skipDict.pop(gate[1][-1], None)
                    if self.INFO: print(f' --- remove {gate[1][-1]} from dictionary 3')

            #if new_qc.data[-1][0].name != 'barrier':
            new_qc.barrier()

        nLength = len(new_qc)
        for i in range(nLength):
            if new_qc.data[nLength-i-1][0].name == 'barrier':
                new_qc.data.pop(nLength-i-1)
                
            
        print(f'Number of measurement = {num_meas}')
        return new_qc    
    
    
    @staticmethod
    def threshold(qc, num_ctrl_qubits, threshold_type, backend, zne):
        
        """
        threshold_type = ['constant', 0.2] or ['dynamic', None] or ['high', 0.2]
        """
        
        threshold_name = threshold_type[0]
        threshold_cap  = threshold_type[1]
        
        if threshold_name == 'dynamic':
            prob_u  = 1
            prob_cx = 1

            for gate in qc:

                # 'id' and 'rz' has no error.
                if gate[0].name in ['sx', 'x']:
                    u_gate_error = backend.properties()._gates[gate[0].name][gate[1][0]._index,]['gate_error'][0]
                    prob_u  = prob_u * (1 - u_gate_error)

                if gate[0].name == 'cx':
                    cx_gate_error = backend.properties()._gates[gate[0].name][gate[1][0]._index,gate[1][1]._index]['gate_error'][0]
                    prob_cx = prob_cx * (1-cx_gate_error)

            if zne == 'on':
                prob = prob_u * (3*prob_cx/2 - prob_cx**3/2)
            if zne == 'off':
                prob = prob_u * prob_cx

            threshold = (1 - prob) / (2**num_ctrl_qubits)
            
            # Cap
            if threshold_cap != None:
                if threshold > threshold_cap:
                    threshold = threshold_cap
                    print(f'threshold = {threshold}')
                    
        if threshold_name == 'constant':
            threshold = threshold_cap
            
        # Minimum threshold
        #if threshold < 0.005:
        #    threshold = 0.005
        
        return threshold
