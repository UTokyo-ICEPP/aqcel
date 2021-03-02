import numpy as np
from aqcel.optimization import RIIM_tools
from qiskit import *
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import CXGate, CCXGate, XGate, CRYGate, RYGate, MCXGate, U3Gate, CU3Gate
import copy
from qiskit.circuit import Qubit
from qiskit.tools.monitor import job_monitor
import time
from qiskit.tools.visualization import plot_histogram
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal, CompleteMeasFitter, TensoredMeasFitter)
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

#Support Gate : X,Y,Z,H,RX,RY,RZ,U1,U2,U3,SX,SXdg,T,Tdg,C(X,Y,Z,H,RX,RY,RZ,U1,U2,U3,SX),TOFFOLI,MCU


class circuit_optimization():
    
    def __init__(self, circuit, threshold):
        self.circuit = circuit
        self.dag = circuit_to_dag(self.circuit)
        self.threshold = threshold
    
    def slim(self):
        circ0= self.DecomposeMCU(self.circuit)
        dag0 = circuit_to_dag(circ0)
        dag1 = self.RemoveAdjacentGates(dag0)
        dag2 = self.RemoveControlledOperataions(dag1)
        dag3 = self.RemoveAdjacentGates(dag2)
        circ1 = dag_to_circuit(dag3)
        circ2 = self.RemoveQubits(circ1)
        return circ2
    
    def DecomposeMCU(self, circ):
        ancilla = 0
        gate_list=[]
        for gate in circ:
            gate_list.append(gate)
            if (gate[0].name != 'barrier') & (len(gate[1]) > 3):
                if ancilla < len(gate[1])-2:
                    ancilla = len(gate[1])-2
        if ancilla > 0:
            w = QuantumRegister(ancilla, 'w')
            circ = QuantumCircuit(*circ.qregs,w,*circ.cregs)
            for gate in gate_list:
                if (len(gate[1]) > 3) & (gate[0].name !='barrier'):
                    circ = self.Decompose_helper(circ, gate, w)
                else:
                    circ.append(gate[0],gate[1],gate[2])
        return circ
        
    @staticmethod    
    def Decompose_helper(qc,gate, w):
        qregs = gate[1]
        n = len(qregs)-1

        qc.ccx(qregs[0],qregs[1],w[0])
        for i in range(n-2):
            qc.ccx(qregs[i+2],w[i],w[i+1])

        qc.append(gate[0].base_gate.control(1),[w[n-2],qregs[-1]],gate[2])

        for i in reversed(range(n-2)):
            qc.ccx(qregs[i+2],w[i],w[i+1])
        qc.ccx(qregs[0],qregs[1],w[0])

        return qc
    
    def RemoveAdjacentGates(self,dag):
        while dag != self.RemoveAdjacentGates_helper(dag):
            dag = self.RemoveAdjacentGates_helper(dag)
        return dag
    
    @staticmethod
    def RemoveAdjacentGates_helper(dag):
        len_dag = len(dag.op_nodes())
        remove_count = 0
        for target_index in range(len_dag-1): # target gate judged whether it can be removed or not
            if target_index-remove_count > 0:
                target_index = target_index-remove_count
            else:
                target_index = 0
            parent = dag.op_nodes()[target_index]
            
            if (parent.name != 'barrier') & (parent.name != 'measure'):
                child_index=[]
                for child in dag.successors(parent):
                    if child.type == 'op':
                        child_index.append(dag.op_nodes().index(child))
                child_index.sort()
                not_op = 0
                for n,child in enumerate(dag.successors(parent)):
                    if child.type == 'op':
                        if dag.op_nodes().index(child) == child_index[n-not_op]:
                            if (parent.qargs[-1] in child.qargs) or (child.qargs[-1] in parent.qargs):
                                if (parent.qargs == child.qargs):
                                    if (parent.name == 'sx') or (child.name == 'sxdg') or (parent.name == 't') or (child.name == 'tdg'):
                                        if ((parent.name == 'sx') and (child.name == 'sxdg')) or ((parent.name == 'sxdg') and (child.name == 'sx')) or ((parent.name == 't') and (child.name == 'tdg')) or ((parent.name == 'tdg') and (child.name == 't')):
                                            dag.remove_op_node(child)
                                            dag.remove_op_node(parent)
                                            len_dag=len_dag-2
                                            remove_count += 2
                                    elif (parent.name == child.name) & (parent.op.inverse().params == child.op.params):
                                        dag.remove_op_node(child)
                                        dag.remove_op_node(parent)
                                        len_dag=len_dag-2
                                        remove_count += 2
                                break
                    else:
                        not_op +=1
        return dag
    
    def RemoveControlledOperataions(self,dag):
        remove_list=[]
        basis_info=[[0]*dag.num_qubits()]   # Indexes in this list matches indexes of qubits in the circuit
        barrier_count=0
        
        for x in range(len(dag.op_nodes())):
            target = dag.op_nodes()[x]
            
            fig_list=[] # This list includes the information of indexes of controlled qubits and target qubits. First "1" is nothing. Later I will remove)
            for qubit in target.qargs:
                index = dag.qubits.index(qubit)
                fig_list.append(index)
            
            if target.name == 'barrier':
                barrier_count+=1
                
            if barrier_count % 2 == 0: # Remove qubit controls except recurring gates

                bit_state=[] # List of used computational bases on the controlled qubits
                for basis in basis_info:
                    bit_state.append( [basis[fig_list[i]] for i in range(len(fig_list)-1)] )

                # Remove unnecessary qubit controls
                if (len(target.qargs) == 2):
                    cu_counter = dag.named_nodes(target.name).index(target)
                    if ([1] in bit_state) & ([0] not in bit_state):
                        self.cu_to_u(cu_counter, target.op, dag, target.op.params)
                    elif ([1] not in bit_state):
                        remove_list.append(x)
                        
                elif target.name == 'ccx':
                    ccx_counter = dag.named_nodes('ccx').index(target)
                    if ([1,1] in bit_state):
                        if len(bit_state) == 1:
                            self.ccx_to_x(ccx_counter,dag)
                        elif ([0,1] not in bit_state):
                            self.ccx_to_cx(ccx_counter,dag,1)
                        elif ([1,0] not in bit_state):
                            self.ccx_to_cx(ccx_counter,dag,0)
                    else:
                        remove_list.append(x)
            
            # Identify used computational bases list
            if (target.name != 'barrier') and (target.name != 'measure'):
                new_basis=[]
                for basis in basis_info:
                    target_bit = fig_list[-1]

                    if (target.name == 'x') or (target.name == 'y'):
                        basis[target_bit] = int(not basis[target_bit])

                    elif (target.name == 'cx') or (target.name == 'ccx') or (target.name == 'cy'):
                        if [basis[fig_list[i]] for i in range(len(fig_list)-1)] == [1]*(len(fig_list)-1):
                            basis[target_bit] = int(not basis[target_bit])

                    elif (target.name == 'h') or (target.name == 'ry') or (target.name == 'rx') or (target.name == 'u2') or (target.name == 'u3') or (target.name == 'sx'):
                        basis2=copy.copy(basis)
                        basis2[target_bit] = int(not basis[target_bit])
                        new_basis.append(basis2)

                    elif (target.name == 'crx') or (target.name == 'ch') or (target.name == 'cu2') or (target.name == 'cu3') or (target.name == 'cry') or (target.name == 'csx'):
                        if basis[fig_list[0]] == 1:
                            basis2=copy.copy(basis)
                            basis2[target_bit] = int(not basis[target_bit])
                            new_basis.append(basis2)


                for basis in new_basis:
                    basis_info.append(basis)    
                basis_info = self.get_unique_list(basis_info)

        remove_list.sort(reverse=True)
        for x in remove_list:
            dag.remove_op_node(dag.op_nodes()[x])
            
        return dag
    

    # Remove unused qubits
    def RemoveQubits(self,circ):
    
        for _ in range(circ.num_qubits):
            register_list=circ.qregs

            qubit_list={}
            for gate in circ:
                if gate[0].name != 'barrier':
                    for qubit in gate[1]:
                        qubit_list.setdefault(qubit,1)

            for qubit in circ.qubits:
                if (qubit in qubit_list) == False:
                    delete_index = qubit.index
                    delete_register = qubit.register

                    register_list.remove(delete_register)
                    if qubit.register.size > 1:
                        new_register = QuantumRegister(qubit.register.size-1, qubit.register.name)
                        register_list.append(new_register)

                    gate_list=[]
                    for gate in circ:
                        gate_list.append(list(gate))
                        
                    for gate in gate_list:
                        if gate[0].name == 'barrier':
                            if len(gate[1])==1:
                                if (gate[1][x].register == delete_register) & (gate[1][x].index == delete_index):
                                        gate_list.pop(gate_list.index(gate))
                            if len(gate[1])>1:
                                b_qubit_list=[]
                                for qubit in gate[1]:
                                    if qubit != Qubit(delete_register, delete_index):
                                        b_qubit_list.append(qubit)
                                gate[1] = b_qubit_list

                    for gate in gate_list:
                        for x in range(len(gate[1])):
                            if gate[1][x].register == delete_register:
                                if gate[1][x].index < delete_index:
                                    gate[1][x]=Qubit(new_register, gate[1][x].index)
                                else:
                                    gate[1][x]=Qubit(new_register, gate[1][x].index-1)

                    new_list=[]
                    for key in register_list:
                        new_list.append(key)
                    circ = QuantumCircuit(*new_list,*circ.cregs)
                    for gate in gate_list:
                        circ.append(gate[0],gate[1],gate[2])
                    
                    break
                                
        return circ
    

    @staticmethod
    def ccx_to_cx(ccx_counter,dag,controlled_index):
        mini_dag = DAGCircuit()
        p = QuantumRegister(3, "p")
        mini_dag.add_qreg(p)
        mini_dag.apply_operation_back(CXGate(), qargs=[p[1],p[2]])
        # substitute the cx node with the above mini-dag
        ccx_node = dag.op_nodes(op=CCXGate).pop(ccx_counter)

        if controlled_index == 0: 
            dag.substitute_node_with_dag(node=ccx_node, input_dag=mini_dag, wires=[p[1], p[0], p[2]])
        if controlled_index == 1:
            dag.substitute_node_with_dag(node=ccx_node, input_dag=mini_dag, wires=[p[0], p[1], p[2]])
        return dag
    
    @staticmethod
    def ccx_to_x(ccx_counter,dag):
        mini_dag = DAGCircuit()
        p = QuantumRegister(3, "p")
        mini_dag.add_qreg(p)
        mini_dag.apply_operation_back(XGate(), qargs=[p[2]])
        # substitute the cx node with the above mini-dag
        ccx_node = dag.op_nodes(op=CCXGate).pop(ccx_counter)
        dag.substitute_node_with_dag(node=ccx_node, input_dag=mini_dag, wires=[p[0], p[1], p[2]])
        return dag

    @staticmethod
    def cu_to_u(cu_counter,target,dag,params):
        mini_dag = DAGCircuit()
        p = QuantumRegister(2, "p")
        mini_dag.add_qreg(p)
        mini_dag.apply_operation_back(target.base_gate.__class__(*params), qargs=[p[0]])
        # substitute the ry node with the above mini-dag
        cu_node = dag.op_nodes(op=target.__class__).pop(cu_counter)
        dag.substitute_node_with_dag(node=cu_node, input_dag=mini_dag, wires=[p[1], p[0]])
        return dag
    
    @staticmethod
    def get_unique_list(seq):
        seen = []
        return [x for x in seq if x not in seen and not seen.append(x)]