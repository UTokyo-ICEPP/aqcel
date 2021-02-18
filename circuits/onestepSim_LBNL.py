#Run with L371 False and then again with True

from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute
import numpy as np
import matplotlib.pyplot as plt
import math
from qiskit.tools.visualization import circuit_drawer
from qiskit import Aer #, IBMQ
from qiskit.providers.aer import noise

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

### Parameters ###

events = 2000 # 1000000
eps = 0.001
gL = 2
gR = 1

### Define the splitting functions and Sudakov factors ###

def ptype(x):
    if x=='000':
        return '0'    
    if x=='001':
        return 'phi'   
    if x=='100':
        return 'f1'   
    if x=='101':
        return 'f2'   
    if x=='110':
        return 'af1'   
    if x=='111':
        return 'af2'   
    else:
        return "NAN"

def P_f(t, g):
    alpha = g**2 * Phat_f(t)/ (4 * math.pi)
    return alpha

def Phat_f(t):
    return(math.log(t))

def Phat_phi(t):
    return(math.log(t))

def Delta_f(t, g):
    return math.exp(P_f(t,g))

def runQuantum(gLR,dophisplit):

    # coupling constants
    gp = math.sqrt(abs((gL - gR)**2 + 4 * gLR * gLR ))

    if (gL > gR):
        gp = -gp

    gA = (gL + gR - gp)/2
    gB = (gL + gR + gp)/2

    # compute the u value, which we label qS
    qS = math.sqrt(abs((gp + gL - gR)/ (2 * gp)))

    def P_phi(t, gA, gB):
        alpha = gA**2 * Phat_phi(t)/ (4 * math.pi) + gB**2 * Phat_phi(t)/ (4 * math.pi)
        return dophisplit*alpha

    def Delta_phi(t, gA, gB):
        return math.exp(P_phi(t,gA,gB))

    ### Define function whihc implements a C^(n)(U) operation, where U is a real 2x2 unitary matrix ###
    ## qc is the quantum circuit
    ## entry is the first entry of U
    ## n is the number of qubits on which we control
    ## q is the list of qubits on which we control
    ## controls specifies if the control qubits must be 0 or 1

    def Cnx(qc,n,q,controls,w,targ):
        
        for i in range(n):
            if controls[i]==0:
                qc.x(q[i])

        qc.ccx(q[0],q[1],w[0])
        
        for i in range(n-2):
            qc.ccx(q[i+2],w[i],w[i+1])

        qc.cx(w[n-2],targ)

        for i in reversed(range(n-2)):
            qc.ccx(q[i+2],w[i],w[i+1])
        
        qc.ccx(q[0],q[1],w[0])
        
        for i in range(n):
            if controls[i]==0:
                qc.x(q[i])

        return(qc)

    def Cn(qc,entry,n,q,controls,w,targ):
        
        for i in range(n):
            if controls[i]==0:
                qc.x(q[i])

        qc.ccx(q[0],q[1],w[0])
        
        for i in range(n-2):
            qc.ccx(q[i+2],w[i],w[i+1])

        qc.cu3(2*np.arccos(entry),0,0,w[n-2],targ)

        for i in reversed(range(n-2)):
            qc.ccx(q[i+2],w[i],w[i+1])
        
        qc.ccx(q[0],q[1],w[0])
        
        for i in range(n):
            if controls[i]==0:
                qc.x(q[i])

        return(qc)

    ### Functions for step by step evolution ###

    def step1(qc,gA,gB):
        
        ## Simulation of the first step ##
        
        t_up = eps**(0)
        t_mid =  eps**(0.5)
        t_low =  eps**(1)    

        # count particles
        
        qcount = [p0[0],p0[1],p0[2]]
        cCount1 = [0,0,1]
        cCount2 = [1,0,1]
        w1 = [w[0],w[1]]
        Cnx(qc,3,qcount,cCount1,w1,a[0])
        Cnx(qc,3,qcount,cCount2,w1,b[0])
        
        # decide emission
        delta1 = math.sqrt(Delta_f(t_low, gA)) / math.sqrt(Delta_f(t_up, gA))
        delta2 = math.sqrt(Delta_f(t_low, gB)) / math.sqrt(Delta_f(t_up, gA))
        qc.cu3(2*np.arccos(delta1),0,0,a[0],e[0])
        qc.cu3(2*np.arccos(delta2),0,0,b[0],e[0])
        
        # create history
        
        qem = [e[0],p0[0],p0[1],p0[2]]
        cCount3 = [1,0,0,1]
        cCount4 = [1,1,0,1]
        w2 = [w[0],w[1],w[2]]
        Cn(qc,0,4,qem,cCount3,w2,h1[0])
        Cn(qc,0,4,qem,cCount4,w2,h1[0])
        
        Cnx(qc,3,qcount,cCount1,w1,a[0])
        Cnx(qc,3,qcount,cCount2,w1,b[0])
        
        # adjust the particle state
        
        qc.cx(h1[0],p1[0])
        
        
        return(qc)


    ## Quantum Simulation ##

    # Number of qubits necessary for each register

    Np0 = 3
    Np1 = 3
    Nh1 = 1
    Ne = 1
    Na = 1
    Nphi = 1
    Nb = 1
    Nw = 4

    # Create a Quantum Registers and Quantum Circuit

    w = QuantumRegister(Nw,'w')
    p0 = QuantumRegister(Np0, 'p0')
    p1 = QuantumRegister(Np1, 'p1')
    h1 = QuantumRegister(Nh1, 'h1')
    e = QuantumRegister(Ne, 'e')
    phi = QuantumRegister(Nphi, 'phi')
    a = QuantumRegister(Na, 'a')
    b = QuantumRegister(Nb, 'b')

    qc = QuantumCircuit()
    qc.add_register(w)
    qc.add_register(p0)
    qc.add_register(p1)
    qc.add_register(h1)
    qc.add_register(e)
    qc.add_register(phi)
    qc.add_register(a)
    qc.add_register(b)

    # set p0 into a superposition of f1 and f2

    qc.x(p0[2])
    #qc.x(p0[0])
    #qc.h(p0[0])

    # rotate initial state into 1/2 basis

    qc.cu3(2*np.arcsin(-qS),0,0,p0[2],p0[0])



    ## Run the quantum circuit ##
    backend_sim = Aer.get_backend('qasm_simulator')
    qc1=step1(qc,gA,gB)

    # rotate states back into 1/2 basis
    qc1.cu3(2*np.arcsin(qS),0,0,p0[2],p0[0])
    qc1.cu3(2*np.arcsin(qS),0,0,p1[2],p1[0])


    # Create classical registers and measurements

    #cw = ClassicalRegister(Nw,'cw')
    cp0 = ClassicalRegister(Np0, 'cp0')
    cp1 = ClassicalRegister(Np1, 'cp1')
    #cp2 = ClassicalRegister(Np2, 'cp2')
    #ch1 = ClassicalRegister(Nh1, 'ch1')
    #ch2 = ClassicalRegister(Nh2, 'ch2')
    #ce = ClassicalRegister(Ne, 'ce')
    #cphi = ClassicalRegister(Nphi, 'cphi')
    #ca = ClassicalRegister(Na, 'ca')
    #cb = ClassicalRegister(Nb, 'cb')


    meas = QuantumCircuit(w,p0,p1,h1,e,phi,a,b,cp0,cp1)
    meas.barrier(w,p0,p1,h1,e,phi,a,b)
    meas.measure(p0,cp0)
    meas.measure(p1,cp1)
    
    circ = qc1+meas

    return circ