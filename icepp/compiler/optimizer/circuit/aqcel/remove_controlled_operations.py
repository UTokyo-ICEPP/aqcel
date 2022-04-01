import copy
from qiskit import *
from qiskit.compiler import transpile
from qiskit.circuit.library.standard_gates import XGate
from .....pass_manager import pass_manager
from .....compiler.transpiler import transpiler

# Support : All gates except "barrier and RCCX"
# ある制御ゲートを対象にしたときに、時系列上ではそれと同等のゲートも作用させたくない（エラーを増やしてしまう。）
# 実機を使って測定するときに、ancillaは|0>に戻すわけだからresetを途中まで適用することはできないか？
# 分解する前もしくは後にMillerの作業ビットを用いた重複ビット制御削除の導入。


class remove_controlled_operations():
    
    def __init__(self, qc):
        self.qc = qc
        
    
    def run_cc(self):
        
        bistrings_dict = self.measure_bitstrings_cc() # Dictionary of bitstrings on controlled qubits
        
        new_qc = QuantumCircuit(*self.qc.qregs, *self.qc.cregs)
        
        for index, gate in enumerate(self.qc):
            
            if index in bistrings_dict: # Check 'gate' is a controlled gate or not
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

        statevector = quantum_info.Statevector(array)
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
    
    
    @staticmethod
    def judge(new_qc, gate, bitstrings):
        target_qubit = gate[1][-1]
                
        """
        gate[0] : gate class
        gate[1] : qregs
        gate[2] : cregs
        """

        # Toffoli gates
        if gate[0].name in ['ccx', 'rccx']:
            
            if '11' in bitstrings:
                if len(bitstrings) == 1:
                    new_qc.append(XGate(label=None), [target_qubit], gate[2])

                elif '10' not in bitstrings:
                    new_qc.append(XGate(label=None).control(1), [gate[1][1], target_qubit], gate[2])

                elif '01' not in bitstrings:
                    new_qc.append(XGate(label=None).control(1), [gate[1][0], target_qubit], gate[2])

                else: #11と00の時だけようにもう一つ作る？(Qubit connectionを一つ減らせる)
                    new_qc.append(gate[0],gate[1],gate[2])

            else:
                pass # Delete CCX

        # Two-qubit gates
        else:
            if '1' in bitstrings:
                if '0' not in bitstrings:
                    new_qc.append(gate[0].base_gate, [target_qubit], gate[2])
                else:
                    new_qc.append(gate[0],gate[1],gate[2])
            else:
                pass # Delete CU
            
        return new_qc
    
    
    def run_qc(self, shots, backend, backend_tket, threshold_type, zne):

        new_qc = QuantumCircuit(*self.qc.qregs)
        
        rccx_qregs, rccx_insteads = [], [] # List of rccxs of a former part of s decomposed mcu gate
        
        for gate in self.qc:
            
            # Measure a list of bitstrings on controlled qubits by a quantum computer
            if (len(gate[1]) > 1) and (gate[0].name != 'barrier'): # Check 'gate' is a controlled gate or not
                
                if (gate[0].name == 'rccx') and (gate[1] in rccx_qregs):
                    alter_gate = rccx_insteads[-1]
                    if alter_gate == None:
                        pass
                    else:
                        new_qc.append(alter_gate[0],alter_gate[1],alter_gate[2])
                    rccx_qregs.pop(-1)
                    rccx_insteads.pop(-1)
                    
                else:
                
                    #num_ctrl_qubits = gate[0].num_ctrl_qubits
                    num_ctrl_qubits = len(gate[1]) - 1

                    c = ClassicalRegister(num_ctrl_qubits, 'c')
                    new_qc.add_register(c)
                    new_qc.barrier()

                    for n in range(num_ctrl_qubits):
                        new_qc.measure(gate[1][n],c[n])

                    print(new_qc.draw(fold=120))

                    transpiled_qc = transpiler(new_qc, backend, backend_tket, level=3).transpile()
                    threshold  = self.threshold(transpiled_qc, num_ctrl_qubits, threshold_type, backend, zne)
                    results = pass_manager(qc=transpiled_qc, level=1, backend=backend, shots=shots, zne=zne).auto_manager()
                    bitstrings = results[-1] # Final counts

                    for key in list(bitstrings.keys()):
                        if bitstrings[key] < threshold*shots: # Cut-off bitstrings below a threshold
                            bitstrings.pop(key)

                    print('Final counts after applying the threshold :', bitstrings)
                    print('Threshold :',threshold*shots)

                    # Remove measurements and all clibits
                    new_qc.remove_final_measurements()
                    
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
            
            else:
                new_qc.append(gate[0],gate[1],gate[2])
                
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
        
        if threshold_name == 'constant':
            threshold = threshold_cap
        
        # Minimum threshold
        if threshold < 0.005:
            threshold = 0.005
        
        return threshold