# Gamma

A module `gamma.py` is introduced in which it construct the 32x32 Hermitian gamma matrices of Euclidean R^10.

The matrices being constructed are

`γ` - `numpy.array` of shape (10,32,32) and `dtype=object`. They are the gamma matrices, where the first index correspond to the ten spatial directions and the other two indices correspond to the spinorial indices ranging over the 32 chiral and anti-chiral values. For each spatial direction, say `a`, `γ[a,:,:]` is an Hermtian matrix full of zeros except for the 16 x 16 off-diagonal blocks. This is because gamma matrices switch chirality and our basis diagonalize the chiral matrix.

`chiral` - `numpy.array` of shape (32,32) and `dtype=int8`. Diagonal matrix with the first 16 entries valued -1 and the last 16 entries valued +1.

`C` - `numpy.array` of shape (32,32) and `dtype=int8`. It is the charge conjugations matrix. In our basis, the first 5 spatial directions are real while the last five directions are imaginary, so `C` commutes with `γ[i,:,:]` for `i` in range `(0,5)` and anti-commutes with `γ[i,:,:]` for `i` in range `(5,10)`.

`γ_gpu` - `torch.tensor` of shape (10,32,32), `dtype=torch.complex128` and `device='cuda:0'`. They are essentially the gamma matrices constructed in GPU using torch.

`C_gpu` - `torch.tensor` of shape (10,32,32), `dtype=torch.complex128` and `device='cuda:0'`. They are essentially the charge conjugation matrices constructed in GPU using torch.
