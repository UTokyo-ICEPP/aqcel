#Run with L371 False and then again with True

from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
import numpy as np
import math
from qiskit.circuit.library.standard_gates import RYGate


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

    def Cnx(qc,n,q,controls,targ):
        
        for i in range(n):
            if controls[i]==0:
                qc.x(q[i])

        qc.mcx(q,targ)
        
        for i in range(n):
            if controls[i]==0:
                qc.x(q[i])

        return(qc)

    def Cn(qc,entry,n,q,controls,targ):
        
        for i in range(n):
            if controls[i]==0:
                qc.x(q[i])
                
        qc.append(RYGate(2*np.arccos(entry)).control(n),q+[targ], [])
        
        for i in range(n):
            if controls[i]==0:
                qc.x(q[i])

        return(qc)

    ### Functions for step by step evolution ###

    def step1(qc,gA,gB):
        
        ## Simulation of the first step ##
        
        t_up = eps**((0)/2)
        t_mid =  eps**((0+0.5)/2)
        t_low =  eps**((0+1)/2)    

        # count particles
        
        qcount = [p0[0],p0[1],p0[2]]
        cCount1 = [0,0,1]
        cCount2 = [1,0,1]
        Cnx(qc,3,qcount,cCount1,a[0])
        Cnx(qc,3,qcount,cCount2,b[0])
        
        # decide emission
        delta1 = math.sqrt(Delta_f(t_low, gA)) / math.sqrt(Delta_f(t_up, gA))
        delta2 = math.sqrt(Delta_f(t_low, gB)) / math.sqrt(Delta_f(t_up, gA))
        qc.cry(2*np.arccos(delta1),a[0],e[0])
        qc.cry(2*np.arccos(delta2),b[0],e[0])
        
        # create history
        
        qem = [e[0],p0[0],p0[1],p0[2]]
        cCount3 = [1,0,0,1]
        cCount4 = [1,1,0,1]
        Cn(qc,0,4,qem,cCount3,h1[0])
        Cn(qc,0,4,qem,cCount4,h1[0])
        
        Cnx(qc,3,qcount,cCount1,a[0])
        Cnx(qc,3,qcount,cCount2,b[0])
        
        # adjust the particle state
        
        qc.cx(h1[0],p1[0])
        
        
        return(qc)

    def step2(qc,gA,gB):
        
        ## Simulation of the first step ##
        
        t_up = eps**((1)/2)
        t_mid =  eps**((1+0.5)/2)
        t_low =  eps**((1+1)/2)    

        # count particles
        
        qc.x(p0[0])
        qc.ccx(p0[0],p0[2],a[0])
        qc.x(p0[0])
        
        qc.ccx(p0[0],p0[2],b[0])
        
        qc.cx(p1[0],phi[0])
        
        # decide emission
        delta_a = math.sqrt(Delta_f(t_low, gA)) / math.sqrt(Delta_f(t_up, gA))
        delta_b = math.sqrt(Delta_f(t_low, gB)) / math.sqrt(Delta_f(t_up, gB))
        delta_aphi = math.sqrt(Delta_phi(t_low, gA, gB)*Delta_f(t_low,gA)) / (math.sqrt(Delta_phi(t_up, gA, gB)*Delta_f(t_up,gA)))
        delta_bphi = math.sqrt(Delta_phi(t_low, gA, gB)*Delta_f(t_low,gB)) / (math.sqrt(Delta_phi(t_up, gA, gB)*Delta_f(t_up,gB)))
        
        q_a = [a[0],phi[0]]
        q_b = [b[0],phi[0]]
        c_a = [1,0]
        c_aphi = [1,1]
        c_b = [1,0]
        c_bphi = [1,1]
        Cn(qc,delta_a,2,q_a,c_a,e[1])
        Cn(qc,delta_aphi,2,q_a,c_aphi,e[1])
        Cn(qc,delta_b,2,q_b,c_b,e[1])
        Cn(qc,delta_bphi,2,q_b,c_bphi,e[1])
        
        # create history
        
        #first we go over p0
        entry_h_a = 0
        entry_h_aphi = math.sqrt(1-(P_f(t_mid,gA)/(P_f(t_mid,gA)+P_phi(t_mid,gA,gB))))
        entry_h_b = 0
        entry_h_bphi = math.sqrt(1-(P_f(t_mid,gB)/(P_f(t_mid,gB)+P_phi(t_mid,gA,gB))))
        
        qcontrol1 = [e[1],p0[0],p0[2],phi[0]]
        controls_a = [1,0,1,0]
        controls_aphi = [1,0,1,1]
        controls_b = [1,1,1,0]
        controls_bphi = [1,1,1,1]
        
        Cn(qc,entry_h_a,4,qcontrol1,controls_a,h2[0])
        Cn(qc,entry_h_aphi,4,qcontrol1,controls_aphi,h2[0])
        Cn(qc,entry_h_b,4,qcontrol1,controls_b,h2[0])
        Cn(qc,entry_h_bphi,4,qcontrol1,controls_bphi,h2[0])
        
        #apply U-'s
        qc.x(p0[0])
        qc.ccx(p0[0],p0[2],a[0])
        qc.x(p0[0])
        qc.ccx(p0[0],p0[2],b[0])
        
        #now go over p1
        
        entry_h_phi = 0
        qcontrol2 = [e[1],p1[0],p1[1],p1[2],h2[0]]
        controls_phi = [1,1,0,0,0]
        Cn(qc,entry_h_phi,5,qcontrol2,controls_phi,h2[1])
        
        #apply U-
        
        qc.cx(p1[0],phi[0])
        
        # adjust the particle state
        
        # if p0 emitted
        qc.cx(h2[0],p2[0])
        
        # if p1 emitted
        qc.cx(h2[1],p2[2])
        qc.cx(h2[1],p1[2])
        qc.cry(2*np.arccos(1/(math.sqrt(2))),h2[1],p2[1])
        entry_r = gA/(math.sqrt(gA*gA+gB*gB))
        qc.cry(2*np.arccos(entry_r),h2[1],p2[0])
        qc.x(p2[1])
        qc.ccx(h2[1],p2[1],p1[1])
        qc.x(p2[1])
        qc.x(p2[0])
        qc.ccx(h2[1],p2[0],p1[0])
        qc.x(p2[0])
        
        return(qc)



    ## Quantum Simulation ##

    # Number of qubits necessary for each register

    Np0 = 3
    Np1 = 3
    Np2 = 3
    Nh1 = 1
    Nh2 = 2
    Ne = 2
    Nphi = 2
    Na = 2
    Nb = 2

    # Create a Quantum Registers and Quantum Circuit

    p0 = QuantumRegister(Np0, 'p0')
    p1 = QuantumRegister(Np1, 'p1')
    p2 = QuantumRegister(Np2, 'p2')
    h1 = QuantumRegister(Nh1, 'h1')
    h2 = QuantumRegister(Nh2, 'h2')
    e = QuantumRegister(Ne, 'e')
    phi = QuantumRegister(Nphi, 'phi')
    a = QuantumRegister(Na, 'a')
    b = QuantumRegister(Nb, 'b')
    
    # Create classical registers and measurements

    cp = ClassicalRegister(Np0+Np1+Np2, 'cp')



    qc = QuantumCircuit(p0,p1,p2,h1,h2,e,phi,a,b,cp)

    # set p0 into a superposition of f1 and f2

    qc.x(p0[2])
    #qc.h(p0[0])

    # rotate initial state into 1/2 basis

    qc.cry(2*np.arcsin(-qS),p0[2],p0[0])



    ## Run the quantum circuit ##
    qc1=step1(qc,gA,gB)
    qc2=step2(qc1,gA,gB)

    # rotate states back into 1/2 basis
    qc2.cry(2*np.arcsin(qS),p0[2],p0[0])
    qc2.cry(2*np.arcsin(qS),p1[2],p1[0])
    qc2.cry(2*np.arcsin(qS),p2[2],p2[0])

    #Measure
    #qc.barrier()
    for x in range(Np0+Np1+Np2):
        qc2.measure(x,x)

    return qc2