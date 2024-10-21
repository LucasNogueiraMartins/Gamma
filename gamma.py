import numpy as np
import torch
import sympy as sp

from scipy.sparse import csr_matrix, vstack, hstack, eye
from sympy import I

def to_latex(γ):
    test = sp.Matrix(γ)
    return print(sp.latex(test))

def lift(up):
    zero = csr_matrix(up.shape, dtype=np.int8)
    result1 = vstack([up, zero])
    result2 = vstack([zero, -up])
    result = hstack([result1, result2])
    return result

def lift2(n):
    id = eye(n, dtype=np.int8, format='csr')
    zero = csr_matrix((n,n), dtype=np.int8)
    result1 = vstack([zero, zero])
    result2 = vstack([id, zero])
    result = hstack([result1, result2])
    return result

def lift3(n):
    id = eye(n, dtype=np.int8, format='csr')
    zero = csr_matrix((n,n), dtype=np.int8)
    result1 = vstack([zero, id])
    result2 = vstack([zero, zero])
    result = hstack([result1, result2])
    return result

def εs(sign):
    εu = np.ones((γ[0].shape[0]), dtype = np.int8)

    s0 = int(sign[0]+'1')
    s1 = int(sign[1]+'1')
    s2 = int(sign[2]+'1')
    s3 = int(sign[3]+'1')
    s4 = int(sign[4]+'1')

    c0 = (γ[0] +s0*I*γ[5])/2
    c1 = (γ[1] +s1*I*γ[6])/2
    c2 = (γ[2] +s2*I*γ[7])/2
    c3 = (γ[3] +s3*I*γ[8])/2
    c4 = (γ[4] +s4*I*γ[9])/2

    c = c0@c1@c2@c3@c4

    return c@εu

def εc(sign):
    εd = np.ones((γ[0].shape[0]), dtype = np.int8)

    s0 = int(sign[0]+'1')
    s1 = int(sign[1]+'1')
    s2 = int(sign[2]+'1')
    s3 = int(sign[3]+'1')
    s4 = int(sign[4]+'1')

    c0 = (γ[0] + s0*I*γ[5])/2
    c1 = (γ[1] + s1*I*γ[6])/2
    c2 = (γ[2] + s2*I*γ[7])/2
    c3 = (γ[3] + s3*I*γ[8])/2
    c4 = (γ[4] + s4*I*γ[9])/2

    c = c0@c1@c2@c3@c4

    return c@εd

number_of_Γs = 5

Γ = np.empty(number_of_Γs, dtype=object)
up = csr_matrix([[0,1],[0,0]], dtype = np.int8)
Γ[0] = up

for i in range(1,number_of_Γs):
    Γ[i] = lift2(Γ[0].shape[0])
    for j in range(i):
        Γ[j] = lift(Γ[j])

Γbar = np.empty(number_of_Γs, dtype=object)
down = np.array([[0,0],[1,0]], dtype = np.int8)
down = csr_matrix(down, dtype = np.int8)
Γbar[0] = down

for i in range(1,number_of_Γs):
    Γbar[i] = lift3(Γbar[0].shape[0])
    for j in range(i):
        Γbar[j] = lift(Γbar[j])

s0 = Γbar[0]@Γ[0] - Γ[0]@Γbar[0]
s1 = Γbar[1]@Γ[1] - Γ[1]@Γbar[1]
s2 = Γbar[2]@Γ[2] - Γ[2]@Γbar[2]
s3 = Γbar[3]@Γ[3] - Γ[3]@Γbar[3]
s4 = Γbar[4]@Γ[4] - Γ[4]@Γbar[4]

# check the chirality
chiral = s0@s1@s2@s3@s4

# obtaining the coords and data of chiral
chiral_coords = chiral.nonzero()
chiral_data = chiral.data
stack = [[chiral_coords[0].tolist()[i], chiral_data.tolist()[i]] for i in range(len(chiral_data))]

# sort stack by the second element of each sublist
stack.sort(key=lambda x: x[1])

order = [stack[i][0] for i in range(len(stack))]

# permute the rows of the chiral matrix using order
chiral = chiral[order]
chiral = chiral[:,order]

for i in range(5):
    Γ[i] = Γ[i][order]
    Γ[i] = Γ[i][:,order]
    Γbar[i] = Γbar[i][order]
    Γbar[i] = Γbar[i][:,order]

chiral = chiral.toarray()

for i in range(number_of_Γs):
    Γ[i] = Γ[i].toarray()

for i in range(number_of_Γs):
    Γbar[i] = Γbar[i].toarray()

γ = [(Γ[i]+Γbar[i]) for i in range(number_of_Γs)] + [I*(Γ[i]-Γbar[i]) for i in range(number_of_Γs)]

C = γ[0]@γ[1]@γ[2]@γ[3]@γ[4]

γu = [γ[i][0:16,16:32]@C[0:16,16:32] for i in range(10)]
γd = [C[16:32,0:16]@γ[i][16:32,0:16] for i in range(10)]

γd = np.array(γd)
γu = np.array(γu)

γ = np.array(γ)

C_gpu = torch.tensor(C.astype(np.complex128), device='cuda')
γ_gpu = torch.tensor(γ.astype(np.complex128), device='cuda')