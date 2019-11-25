# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:22:39 2019

@author: Timothy Fleck

This script calculates the expected value of turns remaining for an individual
player in the board game Chutes and Ladders. The graph at the end shows the EV
(vertical axis) for each square number (horizontal axis). Square 0 represents
the beginning of the game, before the first spin.
"""

from numpy import ones
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from random import randint

#Build a compressed sparse row matrix to store the expected value equations.
#Inputs are: start= starting square, e.g. 0
#            finish = finish square, e.g. 100
#            spin_positions = number of positions on the spinner, e.g. 6
#            chutes is a a dictionary of chutes, with the top (start) as the key,
#               and the bottom (end) as the value.
#            ladders is a dictionary of ladders, with the bottom (start) as the
#               key, and the top (end) as the value.

#b is an array of constants used in the equation.
def buildcsr(start,finish,spin_positions,chutes,ladders):
    b = ones(((finish-start),1),dtype=float)*spin_positions   
    #cusp is the last square before the highest spin would take the player
    #past the finish.
    cusp = finish-spin_positions    
    #data, row_reg, and col_ind are the building blocks of the CSR matrix.
    data = []
    row_reg = [0]
    col_ind = []   
    for s in range(start,cusp):        
        #add chutes to the matrix
        if s in chutes:
            e=chutes[s]
            data.append(-1)
            col_ind.append(e)
            data.append(1)
            col_ind.append(s)
            row_reg.append(row_reg[-1]+2)
            b[s]=0
        #add ladders to the matrix
        elif s in ladders:
            e=ladders[s]
            data.append(1)
            col_ind.append(s)
            if e<finish:
                data.append(-1)
                col_ind.append(e)
                row_reg.append(row_reg[-1]+2)
            else:
                row_reg.append(row_reg[-1]+1)
            b[s]=0
        #For other squares, the EV is the sum of EVs of next 6 squares plus 1.
        else:           
            data.append(spin_positions)
            col_ind.append(s)
            for spin in range(1,spin_positions+1):
                data.append(-1)
                col_ind.append(s+spin)
            row_reg.append(row_reg[-1]+1+spin_positions)
    #After the cusp, the equations change to reflect spins that take the player
    #past the finish. The chutes and the ladders are calculated the same way
    #as pre-cus chutes and ladders.
    for s in range(cusp,finish):
        if s in chutes:
            e=chutes[s]
            data.append(-1)
            col_ind.append(e)
            data.append(1)
            col_ind.append(s)
            row_reg.append(row_reg[-1]+2)
            b[s]=0
        elif s in ladders:
            e=ladders[s]
            data.append(1)
            col_ind.append(s)
            if e<finish:
                data.append(-1)
                col_ind.append(e)
                row_reg.append(row_reg[-1]+2)
            else:
                row_reg.append(row_reg[-1]+1)
            b[s]=0
        else:
            n = finish-s
            data.append(n)
            col_ind.append(s)
            for s1 in range(s+1,finish):
                data.append(-1)
                col_ind.append(s1)
            row_reg.append(row_reg[-1]+n)
    csm = csr_matrix((data,col_ind,row_reg))
    return csm,b

def testCL(p,finish,spin_positions,n):
     if p == finish:
        return n
     else:
        a=randint(1,spin_positions)
        if p+a>finish:
            return testCL(p,finish,spin_positions,n+1)
        else:
            return testCL(p+a,finish,spin_positions,n+1)

if __name__ == "__main__":
        
    CLcsr,b = buildcsr(0,100,6,{16:6,47:26,49:11,56:53,62:19,64:60,87:24,93:73,95:75,98:78},{1:38,4:14,9:31,21:42,28:84,36:44,51:67,71:91,80:100})
    #CLblank,b = buildcsr(0,100,6,{},{})
    x = spsolve(CLcsr,b)
    for i,j in enumerate(x):
        print(i,j)
    
    from matplotlib.pyplot import plot
    
    plot(x)
