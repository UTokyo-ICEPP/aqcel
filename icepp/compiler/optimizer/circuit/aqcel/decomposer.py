from qiskit import *
import copy

# Support : All gates except "barrier"
# Future work : Decompoing rule of mcu

class decompose():
    
    def __init__(self):
        pass
    
    
    def decompose_mcu(self, qc, mode='rccx'): # Decompose multi-controlled gates (the number of controlled qubits > 2) to relative toffolis and two-qubit gates
        
        # Determine the number of anicilla qubits
        num_ancillas=0
        
        mcu_list =[]
        for index, gate in enumerate(qc):           
            if (len(gate[1]) > 3) or (len(gate[1]) == 3 and gate[0].name != 'ccx'): # Check whether multi-controlled gates or not
                
                mcu_list.append(index)
                info = params(gate)
                num_ancillas = max(num_ancillas, info.num_ancillas)

        #Decompose
        if num_ancillas > 0:
            
            #qc = self.decompose_finder(qc, mcu_list)
            
            w = AncillaRegister(num_ancillas, 'w')
            new_qc = QuantumCircuit(*qc.qregs, w, *qc.cregs)
            
            for gate in qc:
                if (len(gate[1]) > 3) or (len(gate[1]) == 3 and gate[0].name != 'ccx'): # Check whether multi-controlled gates or not
                    
                    if mode == 'rccx':
                        new_qc = self.decompose_helper_rccx(new_qc, gate, w)
                    if mode == 'ccx':
                        new_qc = self.decompose_helper_ccx(new_qc, gate, w)
                        
                else:
                    new_qc.append(gate[0],gate[1],gate[2])
                    
        else:
            new_qc = qc
            
        return new_qc
        
        
    @staticmethod    
    def decompose_helper_rccx(qc, gate, w):
        
        info = params(gate)
            
        qc.rccx(info.q_controls[0], info.q_controls[1], w[0])

        for i in range(info.num_ancillas - 1):
            qc.rccx(info.q_controls[i+2], w[i], w[i+1])
            
        if info.gate_type.name == 'cx':
            qc.ccx(w[info.num_ancillas-1], info.q_controls[-1], info.q_target)
        else:
            qc.append(info.gate_type, [ w[info.num_ancillas-1], info.q_target ], gate[2])

        for i in reversed( range(info.num_ancillas - 1) ):
            qc.rccx(info.q_controls[i+2], w[i], w[i+1])

        qc.rccx(info.q_controls[0], info.q_controls[1], w[0])

        return qc
    
    @staticmethod    
    def decompose_helper_ccx(qc, gate, w):
        
        info = params(gate)
            
        qc.ccx(info.q_controls[0], info.q_controls[1], w[0])

        for i in range(info.num_ancillas - 1):
            qc.ccx(info.q_controls[i+2], w[i], w[i+1])
            
        if info.gate_type.name == 'cx':
            qc.ccx(w[info.num_ancillas-1], info.q_controls[-1], info.q_target)
        else:
            qc.append(info.gate_type, [ w[info.num_ancillas-1], info.q_target ], gate[2])

        for i in reversed( range(info.num_ancillas - 1) ):
            qc.ccx(info.q_controls[i+2], w[i], w[i+1])

        qc.ccx(info.q_controls[0], info.q_controls[1], w[0])

        return qc
    
    @staticmethod
    def decompose_finder(qc, mcu_list):
        qc = copy.deepcopy(qc)
        
        common_qubits = [] #これを共通のqubitを前に持っていくアルゴリズム
        
        for i in mcu_list[:-1]:
            
            gate = qc[i]
            qubits = gate[1]
            before_mcu_index, after_mcu_index  = mcu_list[mcu_list.index(i)-1], mcu_list[mcu_list.index(i)+1]

            if len(qubits) >= 3: # MCU

                if common_qubits == []: #初めのゲート
                    after_gate = qc[after_mcu_index]
                    common_qubits = list(set(qubits[:-1]) & set(after_gate[1][:-1]))
                    x,y = i+1, after_mcu_index

                else: #次のゲートから
                    for qubit in common_qubits:
                        if qubit not in qubits[:-1]:
                            common_qubits.remove(qubit)
                    x,y = before_mcu_index+1, i
                            
                for n in range(x, y):
                    not_mcu = qc[n]
                    if not_mcu[1][-1] in common_qubits:
                        common_qubits.remove(not_mcu[1][-1])
                
            if len(common_qubits) >= 2:
                for j, qubit in enumerate(common_qubits):
                    index = qubits.index(qubit)
                    qubits[j], qubits[index] = qubits[index], qubits[j]
            else:
                common_qubits = []
                
        return qc

    
class params():
    
    def __init__(self, gate):
        
        self.gate  = gate
        self.gate_class = self.gate[0]
        self.qregs = self.gate[1]
        self.cregs = self.gate[2]
        
        self.num_ctrl_qubits = self.gate[0].num_ctrl_qubits
        self.gate_type       = self.gate[0].base_gate.control(1) # base gate type of multi-controlled gate (.control(1) adds 1 control to gate)
        
        self.q_controls = self.qregs[:self.num_ctrl_qubits]
        self.q_target   = self.qregs[ self.num_ctrl_qubits]
        #self.q_ancillas = self.qregs[ num_ctrl_qubits + 1 :]
        
        if self.gate_type.name == 'cx': # Need one less ancilla
            self.num_ancillas = self.num_ctrl_qubits - 2
        else:
            self.num_ancillas = self.num_ctrl_qubits - 1