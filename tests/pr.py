import tequila as tq
import pytest
import numpy
import HybridBase as cl

mol = cl.Molecule(geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6", basis_set='6-31g', select={}, condense=False, two_qubit=True)
tqmol = tq.Molecule(basis_set='6-31g', geometry="H 0.0 0.0 0.0\nH 0.0 0.0 1.6")
U = mol.make_ansatz("SPA", edges=[(0, 1)],optimize=False)
U1 = tqmol.make_ansatz("SPA", edges=[(0, 1)],optimize=False)
H = mol.make_hamiltonian()
H1 = tqmol.make_hamiltonian()
print(H)
print(tq.compile_circuit(U))
E = tq.ExpectationValue(H=H, U=U)
E1 = tq.ExpectationValue(H=H1, U=U1)
result = tq.minimize(E)
result1 = tq.minimize(E1)
