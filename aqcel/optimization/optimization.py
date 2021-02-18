from aqcel.optimization.pattern import recognition
from aqcel.optimization.slim import circuit_optimization


class optimizer():
    
    def __init__(self, circuit, slim_level, pattern_level=3, n_patterns=5, min_num_nodes=4, max_num_nodes=7, min_n_repetition=4):
        self.circuit = circuit
        self.slim_level = slim_level
        self.pattern_level = pattern_level
        
        self.n_patterns = n_patterns
        self.min_n_repetition = min_n_repetition
        self.max_num_nodes = max_num_nodes
        self.min_num_nodes = min_num_nodes
        
        if self.slim_level == 2:
            self.circ_max, self.designated_gates, self.barrier_circuit, self.gate_lists = self.pattern()
        
    def pattern(self):
        
        example = recognition(self.circuit, self.n_patterns, self.pattern_level, self.min_n_repetition, self.min_num_nodes, self.max_num_nodes)
        circ_max, designated_gates = example.quantum_pattern()
        barrier_circuit  = example.gate_set_finder()
            
        return circ_max, designated_gates, barrier_circuit, example.gate_list
        
    def slimer(self):
        
        if self.slim_level == 1: 
            circuit = self.circuit
        if self.slim_level == 2:
            circuit = self.barrier_circuit
            
        example = circuit_optimization(circuit, threshold=None)
        optimized_circ = example.slim()
        
        return optimized_circ