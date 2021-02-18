# Need to decrease a calculation cost!!

from networkx.algorithms.mis import maximal_independent_set
import itertools
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
import copy


class recognition():

    def __init__(self, circuit, n_patterns, level, min_n_repetition, min_num_nodes, max_num_nodes):
        self._node_list_cache = {}
        self._parent_nodes_cache = {}
        self._child_nodes_cache = {}
        self._debug = False
        self._not_calc_overlapping = False

        self.circuit = circuit
        self.n_patterns = n_patterns
        self.level = level
        self.min_n_repetition = min_n_repetition
        self.max_num_nodes = max_num_nodes
        self.min_num_nodes = min_num_nodes
        self.max_longest_path_length = max_num_nodes # Always "max_longest_path_length = max_num_nodes"
        self.dag = circuit_to_dag(self.circuit)
        self.subdag_dict, self.gate_list = self.designated_gate()

    @staticmethod
    def get_parent_node(dag, node, max_depth):
        import queue
        q = queue.Queue()
        q.put(node)
        used = {node}
        for _ in range(max_depth):
            n_queues = q.qsize()
            for _ in range(n_queues):
                v = q.get()
                for u in dag.predecessors(v):
                    if u in used:
                        continue
                    q.put(u)
                    used.add(u)
        return used

    @staticmethod
    def get_child_node(dag, node, max_depth):
        import queue
        q = queue.Queue()
        q.put(node)
        used = {node}
        for _ in range(max_depth):
            n_queues = q.qsize()
            for _ in range(n_queues):
                v = q.get()
                for u in dag.successors(v):
                    if u in used:
                        continue
                    q.put(u)
                    used.add(u)
        return used

    def check_bad_nodeset(self, node_set, dag, max_num_nodes, max_depth):
        parents = set()
        children = set()
        for node in node_set:
            if (node, max_num_nodes) not in self._parent_nodes_cache:
                ret = self.get_parent_node(dag, node, max_depth)
                self._parent_nodes_cache[node] = ret
            parents.update(self._parent_nodes_cache[node])

            if (node, self.max_num_nodes) not in self._child_nodes_cache:
                ret = self.get_child_node(dag, node, max_depth)
                self._child_nodes_cache[node] = ret
            children.update(self._child_nodes_cache[node])

        for node in node_set:
            if node in parents:
                parents.remove(node)
            if node in children:
                children.remove(node)
        if len(parents & children) > 0:
            return node_set
        else:
            return None

    def get_node_list(self, dag, node, max_depth, i_depth=0):
        if node in self._node_list_cache:
            return self._node_list_cache[node]

        if i_depth == max_depth:
            self._node_list_cache[node] = [(node,)]
            return [(node,)]

        result = set()
        result_children = [self.get_node_list(dag, child, max_depth-1, i_depth + 1) for child in dag.successors(node) if (child.type != 'out' and child.name != 'measure')]
        
        # All combinations between elements in result_children
        for bit in itertools.product([False, True], repeat=len(result_children)):

            comb_list = [[[node]]]
            for flag, comb_child in zip(bit, result_children):
                if flag:
                    comb_list.append(comb_child)
            # Remove node-sets which already have max_num_nodes elements
            comb_list = [[u for u in v if len(u) < self.max_num_nodes - i_depth] for v in comb_list]  

            for comb in itertools.product(*comb_list):  # TODO: Bottleneck of calculation time
                r = []
                for v in comb:
                    r.extend(v)
                r = tuple(sorted(list(set(r))))
                if len(r) <= self.max_num_nodes-i_depth:
                    result.add(r)
   
        # Remove bad node set
        for node_set in list(result):
            ret = self.check_bad_nodeset(node_set, self.dag, self.max_num_nodes, max_depth)
            if ret is not None:
                result.remove(node_set)
        self._node_list_cache[node] = result
        
        result = [v for v in result if len(v) >= self.min_num_nodes-i_depth]

        return result

    @staticmethod
    def nodeset_to_graph(node_list, dag, level):
        G = nx.DiGraph()
        for node in node_list:
            if level == 2:
                G.add_node(node, node_attr=(node.name, node.qargs[-1]))
            else:
                G.add_node(node, node_attr=(node.name, ))
        
        for node in node_list:
            # Ignore parameters of angles
            if (node.op.name=='ry') or (node.op.name=='cry'):
                node.op.params[0]=1
            if (node.op.name=='u3') or (node.op.name=='cu3'):
                if (node.op.params[1]==0)&(node.op.params[2]==0):
                    node.op.params[0]=1
            
            if level == 2:
                for edge in dag.edges(node):
                    child = edge[1]
                    qbit = edge[2]['wire']
                    if child in node_list:
                        G.add_edge(node, child, edge_attr=(str(qbit.index)))
            elif level == 3:
                for edge in dag.edges(node):
                    child = edge[1]
                    qbit = edge[2]['wire']
                    if child in node_list:
                        if node.name[0] != 'c' or node.qargs[-1] == qbit:
                            G.add_edge(node, child, edge_attr=('target'))
                        else:
                            G.add_edge(node, child, edge_attr=('control'))
            else:
                for child in dag.successors(node):
                    if child in node_list:
                        G.add_edge(node, child)
        return G
    
    
    # DAG情報を用いてゲート集合の出現頻度を探ります
    def designated_gate(self):
        
       # Collect all node-set
        node_combination = []
        n_nodes = len(self.dag.op_nodes())
        node_list_sorted = list(self.dag.op_nodes())

        for i, node in enumerate(reversed(node_list_sorted)):
            if node.type == 'out' or node.name == 'measure':
                continue
            node_list = self.get_node_list(self.dag, node, self.max_longest_path_length - 1)
            node_combination.extend(node_list)

        if self._debug:
            print("total num of graphs = ", len(node_combination))

        # Nodeset to DAG
        from collections import defaultdict
        subdag_dict = defaultdict(list)
        for i, node_list in enumerate(node_combination):
            if self._debug and i % 100 == 0:
                print(f"node-set to graph... {i}/{len(node_combination)}")
            dag_sub = self.nodeset_to_graph(node_list, self.dag, self.level)
            if nx.dag_longest_path_length(dag_sub) >= self.max_longest_path_length:
                continue
            if self.level in [2, 3]:
                dag_sub_hash = nx.weisfeiler_lehman_graph_hash(dag_sub, node_attr="node_attr", edge_attr="edge_attr")  # Graph hash. Use for graph counting
            else:
                dag_sub_hash = nx.weisfeiler_lehman_graph_hash(dag_sub, node_attr="node_attr")  # Graph hash. Use for graph counting
            subdag_dict[dag_sub_hash].append(dag_sub)

        # Frequency of node_sets(graph_hashs)
        max_independent_set = {}
        for graph_hash, graphs in subdag_dict.items():
            n_graphs = len(graphs)
            if self._not_calc_overlapping:
                max_independent_set[graph_hash] = n_graphs
            else:
                # Get Maximum independent set for each graph hash
                G = nx.Graph()
                for i in range(n_graphs):
                    G.add_node(i)
                for (i, j) in itertools.combinations(range(n_graphs), 2):
                    if len(set(graphs[i].nodes()) & set(graphs[j].nodes())) > 0:
                        G.add_edge(i, j)
                max_independent_set[graph_hash] = len(maximal_independent_set(G))

        for key, value in list(max_independent_set.items()):  # Only node_sets which repeated more than n times
            if value < self.min_n_repetition:
                del subdag_dict[key]
                del max_independent_set[key]

        # Identify first to N-th node-set as recurring gate-sets
        subdag_dict_filtered = {}
        for i in range(self.n_patterns):
            if self._debug:
                print(f"Extracting maximum patterns... {i}/{self.n_patterns}")

            max_index = 0
            max_key = None
            for k, v in max_independent_set.items():
                g = subdag_dict[k][0]
                if v * len(g.nodes()) > max_index:
                    max_index = v * len(g.nodes())
                    max_key = k

            if max_key is None:
                break

            g0 = subdag_dict[max_key][0]
            subdag_dict_filtered[max_key] = g0
            
            
            # Remove related graph with the main reccuring graph
            for k in list(max_independent_set.keys()):
                g = subdag_dict[k][0]

                from networkx.algorithms import isomorphism
                if self.level in [2, 3]:
                    GM1 = isomorphism.DiGraphMatcher(g0, g, node_match=isomorphism.categorical_node_match(['node_attr'], [None]), edge_match=isomorphism.categorical_edge_match(['edge_attr'], [None]))
                    GM2 = isomorphism.DiGraphMatcher(g, g0, node_match=isomorphism.categorical_node_match(['node_attr'], [None]), edge_match=isomorphism.categorical_edge_match(['edge_attr'], [None]))
                   
                else:
                    GM1 = isomorphism.DiGraphMatcher(g0, g, node_match=isomorphism.categorical_node_match(['node_attr'], [None]))
                    GM2 = isomorphism.DiGraphMatcher(g, g0, node_match=isomorphism.categorical_node_match(['node_attr'], [None]))
                  
                if GM1.subgraph_is_isomorphic():  # check if g is a subgraph of g0
                    del subdag_dict[k]
                    del max_independent_set[k]

                elif GM2.subgraph_is_isomorphic():  # check if g0 is a subgraph of g
                    del subdag_dict[k]
                    del max_independent_set[k]
                    
            g_sub1 = g0.copy()
            g_sub2 = g0.copy()
            g_sub3 = g0.copy()
            for _ in range(int(len(g0)-4)):# "4" can be changed !!
                g_sub1.remove_node(list(g_sub1.nodes)[len(g_sub1.nodes)-1])
                g_sub2.remove_node(list(g_sub2.nodes)[0])
                
            for _ in range(int((len(g0)-3)/2)):# "3" can be changed !!
                g_sub3.remove_node(list(g_sub3.nodes)[len(g_sub3.nodes)-1])
                g_sub3.remove_node(list(g_sub3.nodes)[0])
            

            for k in list(max_independent_set.keys()):
                g = subdag_dict[k][0]

                from networkx.algorithms import isomorphism
                if self.level in [2, 3]:
                    GM3 = isomorphism.DiGraphMatcher(g, g_sub1, node_match=isomorphism.categorical_node_match(['node_attr'], [None]), edge_match=isomorphism.categorical_edge_match(['edge_attr'], [None]))
                    GM4 = isomorphism.DiGraphMatcher(g, g_sub2, node_match=isomorphism.categorical_node_match(['node_attr'], [None]), edge_match=isomorphism.categorical_edge_match(['edge_attr'], [None]))
                    GM5 = isomorphism.DiGraphMatcher(g, g_sub3, node_match=isomorphism.categorical_node_match(['node_attr'], [None]), edge_match=isomorphism.categorical_edge_match(['edge_attr'], [None]))

                else:
                    GM3 = isomorphism.DiGraphMatcher(g, g_sub1, node_match=isomorphism.categorical_node_match(['node_attr'], [None]))
                    GM4 = isomorphism.DiGraphMatcher(g, g_sub2, node_match=isomorphism.categorical_node_match(['node_attr'], [None]))
                    GM5 = isomorphism.DiGraphMatcher(g, g_sub3, node_match=isomorphism.categorical_node_match(['node_attr'], [None]))

                if GM3.subgraph_is_isomorphic():  # check if g_sub1 is a subgraph of g
                    del subdag_dict[k]
                    del max_independent_set[k]
                    
                elif GM4.subgraph_is_isomorphic():  # check if g_sub2 is a subgraph of g
                    del subdag_dict[k]
                    del max_independent_set[k]
                    
                elif GM5.subgraph_is_isomorphic():  # check if g_sub3 is a subgraph of g
                    del subdag_dict[k]
                    del max_independent_set[k]
                    
        
        # Find recurring gates in the circuit
        gate_list=[]
        for _ in range(len(subdag_dict_filtered)):
            gate_list.append([])

        for i, node_list in enumerate(node_combination):
            
            dag_sub = self.nodeset_to_graph(node_list, self.dag, self.level)
            if nx.dag_longest_path_length(dag_sub) >= self.max_longest_path_length:
                continue
            if self.level in [2, 3]:
                dag_sub_hash = nx.weisfeiler_lehman_graph_hash(dag_sub, node_attr="node_attr", edge_attr="edge_attr") 
            else:
                dag_sub_hash = nx.weisfeiler_lehman_graph_hash(dag_sub, node_attr="node_attr") 
            
            for k,value in enumerate(subdag_dict_filtered.items()):
                small_list=[]
                sub_list=[]
                a=nx.weisfeiler_lehman_graph_hash(value[1], node_attr="node_attr", edge_attr="edge_attr")
                if a == dag_sub_hash:
                    gate_index=0
                    for node in self.dag.op_nodes():
                        if (node.name == node_list[0].name) & (node.qargs == node_list[0].qargs):
                            gate_index+=1
                        if str(node.op) == str(node_list[0].op):  
                            small_list.append(gate_index)
                            small_list.append(len(node_list))
                            small_list.append(node.name)
                            small_list.append(node.qargs)
                            sub_list.append(small_list)
                            sub_list.append(list(node_list))
                            gate_list[k].append(sub_list)
                            break

        return subdag_dict_filtered, gate_list

    

    # Output recurring gates as quantum circuits
    def quantum_pattern(self):

        if len(self.subdag_dict) == 0:  # If there is no recurring gate, output meaningless circuit
            circ_0 = QuantumCircuit(1)
            return circ_0

        designed_gates=[]
        for graph_hash in self.subdag_dict.keys():
            g = self.subdag_dict[graph_hash]
            circ_max = QuantumCircuit(*self.circuit.qregs, *self.circuit.cregs)
            for node in nx.topological_sort(g):
                circ_max.append(node.op, node.qargs, node.cargs)
            designed_gates.append(circ_max)

        circ_max = QuantumCircuit(*self.circuit.qregs, *self.circuit.cregs)
        for graph_hash in self.subdag_dict.keys():
            g = self.subdag_dict[graph_hash]
            for node in nx.topological_sort(g):
                circ_max.append(node.op, node.qargs, node.cargs)
            circ_max.barrier([i for i in range(self.circuit.num_qubits)])

        return circ_max, designed_gates

    # Output the circuit in which recurring gates are surrounded by barrier
    def gate_set_finder(self):
        circ2 = QuantumCircuit(*self.circuit.qregs,*self.circuit.cregs)
        
        newgate_list=[]
        for onegate_list in self.gate_list:
            for smallgate_list in onegate_list:
                newgate_list.append(smallgate_list)
                
        gate_index_list=[0]*len(newgate_list)
        index_list=[0]*len(newgate_list)
        delete_list=[]
        
        for j in range(len(self.circuit)):
            
            for i, gate_list in enumerate(newgate_list):
                if (self.circuit[j][0].name == gate_list[0][2]) & (self.circuit[j][1] == gate_list[0][3]):
                    gate_index_list[i]+=1
                    if gate_index_list[i] == gate_list[0][0]:
                        circ2.barrier(*self.circuit.qregs)
                        circ2.append(self.circuit[j][0],self.circuit[j][1],self.circuit[j][2])
                        gate_list[1].pop(0)
                        index_list[i]=j
                        delete_list.append(j)
                        
                        for l in range(gate_list[0][1]-1):
                            for k in range(len(self.circuit)-j):
                                if delete_list.count(j+k)==0:
                                    if (self.circuit[j+k][0].name == gate_list[1][l].name) & (self.circuit[j+k][1] == gate_list[1][l].qargs):
                                        circ2.append(self.circuit[j+k][0],self.circuit[j+k][1],self.circuit[j+k][2])
                                        delete_list.append(j+k)
                                        break
                        circ2.barrier(*self.circuit.qregs)
            
            if delete_list.count(j)==0:
                circ2.append(self.circuit[j][0],self.circuit[j][1],self.circuit[j][2])

        return circ2