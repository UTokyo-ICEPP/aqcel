import numpy as np
from transpiler.optimization import RIIM_tools
from qiskit import *
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import CXGate, CCXGate, XGate, CRYGate, RYGate, MCXGate, U3Gate, CU3Gate
import copy
from qiskit.circuit import Qubit
from qiskit.tools.monitor import job_monitor
import time
from qiskit.tools.visualization import plot_histogram
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)


class circuit_optimization():
    
    def __init__(self, circuit, slim_level, cut, work_register='None'):
        self.circuit = circuit
        self.slim_level = slim_level
        self.dag = circuit_to_dag(self.circuit)
        self.work_register = work_register
        self.cut = cut
    
    def slim(self):
        noU3_dag   = self.u3_transfer_ry(self.dag)
        delete_dag = self.delete(noU3_dag)
        while delete_dag != self.delete(delete_dag):
            delete_dag = self.delete(delete_dag)
        basis_dag  = self.basis(delete_dag)
        delete_dag = self.delete(basis_dag)
        while delete_dag != self.delete(delete_dag):
            delete_dag = self.delete(delete_dag)
        
        circuit = dag_to_circuit(delete_dag)
        circuit = self.delete_qubit(circuit)
        
        return circuit
    
    def slim_quantum(self, shots, backend):
        noU3_dag   = self.u3_transfer_ry(self.dag)
        delete_dag = self.delete(noU3_dag)
        while delete_dag != self.delete(delete_dag):
            delete_dag = self.delete(delete_dag)
        basis_dag  = self.basis_quantum(delete_dag, shots, backend)
        delete_dag = self.delete(basis_dag)
        while delete_dag != self.delete(delete_dag):
            delete_dag = self.delete(delete_dag)
 
        circuit = dag_to_circuit(delete_dag)
        circuit = self.delete_qubit(circuit)
        
        return circuit, basis_dag
    
    def u3_transfer_ry(self,dag):
        for x in range(len(dag.op_nodes())):
            if dag.op_nodes()[x].name == 'u3':
                if (dag.op_nodes()[x].op.params[1]==0) & (dag.op_nodes()[x].op.params[2]==0):
                    theta=dag.op_nodes()[x].op.params[0]
                    self.u3_to_ry(dag,theta)
                    
            if dag.op_nodes()[x].name == 'cu3':
                if (dag.op_nodes()[x].op.params[1]==0) & (dag.op_nodes()[x].op.params[2]==0):
                    theta=dag.op_nodes()[x].op.params[0]
                    self.cu3_to_cry(dag,theta)             
        return dag

    def delete(self,dag):
        len_dag=len(dag.op_nodes())
        delete_list=[]
        i=0
        for x in range(len_dag-1): # target gate judged whether it can be removed or not
            if x-i>0:
                x=x-i
            else:
                x=0
            if dag.op_nodes()[x].name != 'barrier': #barrier is ignored
                for z in range(len_dag-x-1): #compared gate with target gate
                    for a in range(len(dag.op_nodes()[x].qargs)):
                        for b in range(len(dag.op_nodes()[x+z+1].qargs)):
                            if (dag.op_nodes()[x].qargs[-1]==dag.op_nodes()[x+z+1].qargs[b]) or (dag.op_nodes()[x].qargs[a]==dag.op_nodes()[x+z+1].qargs[-1]): #find the nearest gate that share a same qubit which is a target qubit
                                if (dag.op_nodes()[x].name == 'cry') or (dag.op_nodes()[x].name == 'ry'): 
                                    if (dag.op_nodes()[x].name==dag.op_nodes()[x+z+1].name) & (dag.op_nodes()[x].qargs==dag.op_nodes()[x+z+1].qargs):
                                        if dag.op_nodes()[x].op.params[0] == -(dag.op_nodes()[x+z+1].op.params[0]):
                                            dag.remove_op_node(dag.op_nodes()[x])
                                            dag.remove_op_node(dag.op_nodes()[x+z])
                                            len_dag=len_dag-2
                                            i+=2
                                        else:
                                            dag.op_nodes()[x+z+1].op.params[0] = dag.op_nodes()[x+z+1].op.params[0]+dag.op_nodes()[x].op.params[0]
                                            dag.remove_op_node(dag.op_nodes()[x])
                                            len_dag=len_dag-1
                                            i+=1
                                elif (dag.op_nodes()[x].name==dag.op_nodes()[x+z+1].name) & (dag.op_nodes()[x].qargs==dag.op_nodes()[x+z+1].qargs):
                                        dag.remove_op_node(dag.op_nodes()[x])
                                        dag.remove_op_node(dag.op_nodes()[x+z])
                                        len_dag=len_dag-2
                                        i+=2
                                break   
                        else:
                            continue
                        break
                    else:
                        continue
                    break

        return dag
  
    def basis(self,dag):
        node_list=[]
        remove_list=[]
        ccx_counter=0
        cx_counter =0
        cry_counter=0
        basis_info=[[0]*dag.num_qubits()]   # Indexes in this list matches indexes of qubits in the circuit

        
        barrier_count=0
        for x in range(len(dag.op_nodes())):
            
            if dag.op_nodes()[x].name == 'barrier':
                barrier_count+=1
                
            if barrier_count % 2 == 0: # Remove qubit controls except recurring gates

                fig_list=[1] # This list includes the information of indexes of controlled qubits and target qubits. First "1" is nothing. Later I will remove)

                for a in dag.op_nodes()[x].qargs:
                    for n in range(self.circuit.num_qubits):
                        if a == self.circuit.qubits[n]:
                            fig_list.append(n)

                bit_state={} # List of used computational bases on the controlled qubits
                for basis in basis_info:
                    bit_state.setdefault(tuple([basis[fig_list[i+1]] for i in range(len(fig_list)-2)]),1)

                # Remove unnecessary qubit controls
                if dag.op_nodes()[x].name == 'cx':
                    if (((1,) in bit_state) == True) & (((0,) in bit_state) == False):
                        self.cx_to_x(cx_counter,dag)
                        cx_counter=cx_counter-1
                    elif ((1,) in bit_state) == False:
                        remove_list.append(x)
                    cx_counter+=1

                elif dag.op_nodes()[x].name == 'ccx':
                    if ((1,1) in bit_state) == True:
                        if len(bit_state) == 1:
                            self.ccx_to_x(ccx_counter,dag)
                            ccx_counter=ccx_counter-1
                        elif ((0,1) in bit_state) == False:
                            self.ccx_to_cx(ccx_counter,dag,1)
                            ccx_counter=ccx_counter-1
                            cx_counter+=1
                        elif ((1,0) in bit_state) == False:
                            self.ccx_to_cx(ccx_counter,dag,0)
                            ccx_counter=ccx_counter-1
                            cx_counter+=1                  
                    else:
                        remove_list.append(x)
                    ccx_counter+=1

                elif dag.op_nodes()[x].name == 'cry':
                    if (((1,) in bit_state) == True) & (((0,) in bit_state) == False):
                        self.cry_to_ry(cry_counter, dag, dag.op_nodes()[x].op.params[0])
                        cry_counter=cry_counter-1
                    elif ((1,) in bit_state) == False:
                        remove_list.append(x)
                    cry_counter+=1
            
            if barrier_count % 2 != 0:
                if dag.op_nodes()[x].name == 'cx':
                    cx_counter+=1
                elif dag.op_nodes()[x].name == 'ccx':
                    ccx_counter+=1
                elif dag.op_nodes()[x].name == 'cry':
                    cry_counter+=1
                
            
            # Identify used computational bases list
            fig_list=[1]
            
            for a in dag.op_nodes()[x].qargs:
                for n in range(self.circuit.num_qubits):
                    if a == self.circuit.qubits[n]:
                        fig_list.append(n)

            new_basis=[]
            
            for basis in basis_info:

                if dag.op_nodes()[x].name == 'x':
                    basis[fig_list[1]]=int(not basis[fig_list[1]])
                
                elif (dag.op_nodes()[x].name == 'cx') or (dag.op_nodes()[x].name == 'ccx'):
                    if [basis[fig_list[i+1]] for i in range(len(fig_list)-2)] == [1]*(len(fig_list)-2):
                        basis[fig_list[-1]]=int(not basis[fig_list[-1]])
                    elif basis[fig_list[-1]] == 1:
                        basis[fig_list[-1]]=1
                    else:
                        basis[fig_list[-1]]=0

                elif (dag.op_nodes()[x].name == 'h') or (dag.op_nodes()[x].name == 'ry'):
                    basis2=copy.copy(basis)
                    basis2[fig_list[1]]=int(not basis[fig_list[1]])
                    new_basis.append(basis2)
                    
                elif dag.op_nodes()[x].name == 'cry':
                    if basis[fig_list[1]] == 1:
                        basis2=copy.copy(basis)
                        basis2[fig_list[2]]=int(not basis[fig_list[2]])
                        new_basis.append(basis2)
                    elif basis[fig_list[2]] == 1:
                        basis[fig_list[2]]=1
                    else:
                        basis[fig_list[2]]=0

            
            for basis in new_basis:
                basis_info.append(basis)    
            
            basis_info = self.get_unique_list(basis_info)

        i=0
        for x in remove_list:
            dag.remove_op_node(dag.op_nodes()[x-i])
            i=i+1
            
        return dag
    
    def basis_quantum(self, dag, shots, backend):
        node_list=[]
        remove_counter=0
        ccx_counter=0
        cx_counter =0
        cry_counter=0
        barrier_count=0
        c1 = ClassicalRegister(1,'c1')
        c2 = ClassicalRegister(2,'c2')
        each_circ = QuantumCircuit(*self.circuit.qregs)
        
        for x in range(len(dag.op_nodes())):
            x = x - remove_counter
            yes_remove = 0
            
            if dag.op_nodes()[x].name == 'barrier':
                barrier_count+=1
                
            if barrier_count % 2 == 0: # Remove qubit controls except recurring gates
                # Identify used computational bases list
                if (len(dag.op_nodes()[x].qargs) >= 2) & (dag.op_nodes()[x].name != 'barrier'):
                    num_measure = len(dag.op_nodes()[x].qargs[:-1])
                    if num_measure == 1:
                        each_circ.add_register(c1)
                        for i,ctr in enumerate(dag.op_nodes()[x].qargs[:-1]):
                            each_circ.measure(ctr,c1[i])
                    if num_measure == 2:
                        each_circ.add_register(c2)  
                        for i,ctr in enumerate(dag.op_nodes()[x].qargs[:-1]):
                            each_circ.measure(ctr,c2[i])
                        
                    if backend == 'qasm_simulator':
                        counts = self.measure_bases_sim(each_circ, shots = shots, qc = backend)
                    else:
                        print(each_circ)
                        counts, cutting_line = self.measure_bases(each_circ, shots = shots, qc = backend, num_measure = num_measure, cut = self.cut)
                    
                    basis_info = []
                    for key,value in counts.items():
                        each_basis=[]
                        if value > cutting_line: #Cutting-off line
                            if num_measure == 1:
                                each_basis.append(int(key[0]))
                            if num_measure == 2:
                                each_basis.append(int(key[1]))
                                if len(dag.op_nodes()[x].qargs) == 3:
                                    each_basis.append(int(key[0]))
                            basis_info.append(each_basis)
                            
                    # Remove unnecessary qubit controls
                    if dag.op_nodes()[x].name == 'cx':
                        if (([1] in basis_info) == True) & (([0] in basis_info) == False):
                            self.cx_to_x(cx_counter,dag)
                            cx_counter=cx_counter-1
                        elif ([1] in basis_info) == False:
                            dag.remove_op_node(dag.op_nodes()[x])
                            cx_counter=cx_counter-1
                            remove_counter+=1
                            yes_remove+=1
                        cx_counter+=1

                    elif dag.op_nodes()[x].name == 'ccx':
                        if ([1,1] in basis_info) == True:
                            if len(basis_info) == 1:
                                self.ccx_to_x(ccx_counter,dag)
                                ccx_counter=ccx_counter-1
                            elif ([0,1] in basis_info) == False:
                                self.ccx_to_cx(ccx_counter,dag,1)
                                ccx_counter=ccx_counter-1
                                cx_counter+=1
                            elif ([1,0] in basis_info) == False:
                                self.ccx_to_cx(ccx_counter,dag,0)
                                ccx_counter=ccx_counter-1
                                cx_counter+=1                  
                        else:
                            dag.remove_op_node(dag.op_nodes()[x])
                            ccx_counter=ccx_counter-1
                            remove_counter+=1
                            yes_remove+=1
                        ccx_counter+=1

                    elif dag.op_nodes()[x].name == 'cry':
                        if (([1] in basis_info) == True) & (([0] in basis_info) == False):
                            self.cry_to_ry(cry_counter, dag, dag.op_nodes()[x].op.params[0])
                            cry_counter=cry_counter-1
                        elif ([1] in basis_info) == False:
                            dag.remove_op_node(dag.op_nodes()[x])
                            cry_counter=cry_counter-1
                            remove_counter+=1
                            yes_remove+=1
                        cry_counter+=1
                    

                each_circ_dag = circuit_to_dag(each_circ)
                each_circ_dag.remove_all_ops_named('measure')
                if yes_remove == 0:
                    node = dag.op_nodes()[x]
                    if node.op.name != 'measure':
                        each_circ_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                each_circ = dag_to_circuit(each_circ_dag)
                if len(each_circ.cregs)>0:
                    each_circ.cregs.pop(0)
                

            if barrier_count % 2 != 0:
                if dag.op_nodes()[x].name == 'cx':
                    cx_counter+=1
                elif dag.op_nodes()[x].name == 'ccx':
                    ccx_counter+=1
                elif dag.op_nodes()[x].name == 'cry':
                    cry_counter+=1
     
        return dag


    # Remove unused qubits
    def delete_qubit(self,circ):
    
        for _ in range(circ.num_qubits):
            
            c_register_list={}
            for clbit in circ.clbits:
                c_register_list.setdefault(clbit.register,1)
                
            new_c_list=[]
            for key in c_register_list:
                new_c_list.append(key)
            
            register_list={}
            for qubit in circ.qubits:
                register_list.setdefault(qubit.register,1)

            qubit_list={}
            for gate in circ:
                if gate[0].name != 'barrier':
                    for qubit in gate[1]:
                        qubit_list.setdefault(qubit,1)

            for qubit in circ.qubits:
                if (qubit in qubit_list) == False:
                    delete_index = qubit.index
                    delete_register = qubit.register

                    if delete_register in register_list:
                        register_list.pop(delete_register)

                    if qubit.register.size > 1:
                        new_register = QuantumRegister(qubit.register.size-1, qubit.register.name)
                        register_list.setdefault(new_register,1)

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
                    circ = QuantumCircuit(*new_list,*new_c_list)
                    for gate in gate_list:
                        circ.append(gate[0],gate[1],gate[2])
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
    def cx_to_x(cx_counter,dag):
        mini_dag = DAGCircuit()
        p = QuantumRegister(2, "p")
        mini_dag.add_qreg(p)
        mini_dag.apply_operation_back(XGate(), qargs=[p[0]])
        # substitute the x node with the above mini-dag
        cx_node = dag.op_nodes(op=CXGate).pop(cx_counter)
        dag.substitute_node_with_dag(node=cx_node, input_dag=mini_dag, wires=[p[1], p[0]])
        return dag

    @staticmethod
    def cry_to_ry(cry_counter,dag,theta):
        mini_dag = DAGCircuit()
        p = QuantumRegister(2, "p")
        mini_dag.add_qreg(p)
        mini_dag.apply_operation_back(RYGate(theta), qargs=[p[0]])
        # substitute the ry node with the above mini-dag
        cry_node = dag.op_nodes(op=CRYGate).pop(cry_counter)
        dag.substitute_node_with_dag(node=cry_node, input_dag=mini_dag, wires=[p[1], p[0]])
        return dag

    @staticmethod
    def u3_to_ry(dag,theta):
        mini_dag = DAGCircuit()
        p = QuantumRegister(1, "p")
        mini_dag.add_qreg(p)
        mini_dag.apply_operation_back(RYGate(theta), qargs=[p[0]])
        # substitute the ry node with the above mini-dag
        u3_node = dag.op_nodes(op=U3Gate).pop(0)
        dag.substitute_node_with_dag(node=u3_node, input_dag=mini_dag, wires=[p[0]])
        return dag
    
    @staticmethod
    def cu3_to_cry(dag,theta):
        mini_dag = DAGCircuit()
        p = QuantumRegister(2, "p")
        mini_dag.add_qreg(p)
        mini_dag.apply_operation_back(CRYGate(theta), qargs=[p[0],p[1]])
        # substitute the ry node with the above mini-dag
        cu3_node = dag.op_nodes(op=CU3Gate).pop(0)
        dag.substitute_node_with_dag(node=cu3_node, input_dag=mini_dag, wires=[p[0],p[1]])
        return dag

    @staticmethod
    def get_unique_list(seq):
        seen = []
        return [x for x in seq if x not in seen and not seen.append(x)]
    

    @staticmethod
    def measure_bases(circ, shots, qc, num_measure, cut):
        best_transpiled = transpile(circ, backend=qc, optimization_level=3)
        
        for _ in range(10):
            new_circ_lv3 = transpile(circ, backend=qc, optimization_level=3)
            if new_circ_lv3.__len__() < best_transpiled.__len__():
                best_transpiled = new_circ_lv3
        
        #Calculate cutting-off line
        true_prob_precise = 1
        for gate in best_transpiled:
            if (gate[0].name == 'id') or (gate[0].name == 'u1') or (gate[0].name == 'u2') or (gate[0].name == 'u3'):
                true_prob_precise = true_prob_precise * (1-qc.properties()._gates[gate[0].name][gate[1][0].index,]['gate_error'][0])
            if gate[0].name == 'cx':
                true_prob_precise = true_prob_precise * (1-qc.properties()._gates[gate[0].name][gate[1][0].index,gate[1][1].index]['gate_error'][0])
        
        noise_counts_high = (1 - true_prob_precise) * shots
        noise_counts_low  = noise_counts_high/(2**num_measure)
        noise_counts_mid  = (noise_counts_high + noise_counts_low)/2
        print('high:',noise_counts_high, 'mid:',noise_counts_mid, 'low:',noise_counts_low)
        
        if cut[1] == None:
            if cut[0] == 'high':
                cutting_line = noise_counts_high
            if cut[0] == 'mid':
                cutting_line = noise_counts_mid
            if cut[0] == 'low':
                cutting_line = noise_counts_low

        else:
            if cut[0] == 'high':
                if noise_counts_high < cut[1]*shots:
                    cutting_line = noise_counts_high
                else:
                    cutting_line = cut[1]*shots
            if cut[0] == 'mid':
                if noise_counts_mid < cut[1]*shots:
                    cutting_line = noise_counts_mid
                else:
                    cutting_line = cut[1]*shots
            if cut[0] == 'low':
                if noise_counts_low < cut[1]*shots:
                    cutting_line = noise_counts_low
                else:
                    cutting_line = cut[1]*shots
            if cut[0] == 'Constant':
                cutting_line = cut[1]*shots
                
        if cutting_line < shots*0.05:
            cutting_line = shots*0.05
            
        print('Cut-off line:',cutting_line)
        
        
        #Measurement error mitigation
        job1 = execute(best_transpiled, backend=qc, shots=shots)
        job_monitor(job1)
        result_raw = job1.result()       
        print('Raw counts:',result_raw.get_counts())

        final_layout=[]
        for x in range(num_measure):
            gate = best_transpiled[best_transpiled.__len__() - num_measure + x]
            final_layout.append(gate[1][0].index)

        meas_calibs, state_labels = complete_meas_cal(qubit_list=final_layout, qr=best_transpiled.qregs[0], circlabel='mcal')
        
        job2 = execute(meas_calibs, backend=qc, shots=shots)
        job_monitor(job2)
        cal_results = job2.result()
        
        meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=final_layout, circlabel='mcal')
        meas_filter = meas_fitter.filter
        mitigated_results = meas_filter.apply(result_raw)
        mitigated_counts = mitigated_results.get_counts(0)    
        print('Mitigated counts',mitigated_counts)
        
        return mitigated_counts, cutting_line
    
    @staticmethod
    def measure_bases_sim(circ, shots, qc):
        circ=circ
        backend = Aer.get_backend(qc)
        job = execute(circ, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circ)
        return counts