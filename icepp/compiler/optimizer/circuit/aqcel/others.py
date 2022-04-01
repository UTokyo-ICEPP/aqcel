from qiskit import *
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit import Qubit


# Support : All gates except "barrier"


class other_passes():
    
    def __init__(self):
        pass
    
    
    def remove_adjacent(self, qc):
        
        dag = circuit_to_dag(qc)
        
        while dag != self.remove_adjacent_helper(dag):
            dag = self.remove_adjacent_helper(dag)
        new_qc = dag_to_circuit(dag)
        
        return new_qc
    
    
    @staticmethod
    def remove_adjacent_helper(dag):
        
        nodes = dag.op_nodes()
        num_remove = 0
        
        for parent_index in range( len(dag.op_nodes()) - 1): # A parent gate is judged whether it can be removed or not

            
            # Reset the index of a parent gate with the number of removed gates
            if parent_index - num_remove > 0:
                parent_index = parent_index - num_remove
            else:
                parent_index = 0
                
            parent_node = dag.op_nodes()[parent_index]
            
            if parent_node.name not in ('barrier' , 'measure' , 'reset'):
                
                for child_index in range(parent_index+1, len(dag.op_nodes())):
                    child_node = dag.op_nodes()[child_index]
                    
                    if (parent_node.qargs[-1] in child_node.qargs) or (child_node.qargs[-1] in parent_node.qargs):
                        if parent_node.qargs == child_node.qargs:
                            if (parent_node.name == 'rccx') or (parent_node.op.inverse() == child_node.op):
                                dag.remove_op_node(child_node)
                                dag.remove_op_node(parent_node)
                                num_remove += 2
                        break

        return dag
    

    # Remove unused qubits
    @staticmethod
    def remove_qubits(qc):

        register_list = qc.qregs

        # list of used qubits
        used_qubit_list={}
        for gate in qc:
            for qubit in gate[1]:
                used_qubit_list.setdefault(qubit, 1)

        while len(used_qubit_list) != len(qc.qubits):
            
            # These 4 lines were added after experiments except 
            used_qubit_list={}
            for gate in qc:
                for qubit in gate[1]:
                    used_qubit_list.setdefault(qubit, 1)
            
            
            for qubit in qc.qubits:
                if (qubit in used_qubit_list) == False:
                    del_index = qubit._index
                    del_register = qubit._register

                    register_list.remove(del_register)

                    if qubit._register.size > 1:
                        new_register = QuantumRegister(qubit._register.size-1, qubit._register.name)
                        register_list.append(new_register)

                        gate_list=[]
                        for gate in qc:
                            qubit_list = gate[1]
                            for x in range( len(qubit_list) ):
                                if qubit_list[x]._register == del_register:
                                    if qubit_list[x]._index < del_index:
                                        qubit_list[x]=Qubit(new_register, qubit_list[x]._index)
                                    else:
                                        qubit_list[x]=Qubit(new_register, qubit_list[x]._index-1)
                            gate_list.append(gate)

                    else:
                        gate_list=[]
                        for gate in qc:
                            gate_list.append(gate)

                    qc = QuantumCircuit(*register_list, *qc.cregs)

                    for gate in gate_list:
                        qc.append(gate[0],gate[1],gate[2])

                    break

        return qc