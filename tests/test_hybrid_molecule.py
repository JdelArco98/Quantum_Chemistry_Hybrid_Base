import tequila as tq
import pytest
import numpy
import HybridBase as cl
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6","H 0.0 0.0 0.0\nLi 0.0 0.0 1.6","He 0. 0. 0."])
@pytest.mark.parametrize("basis_set",["6-31g"])
@pytest.mark.parametrize("select",["","FFFFFFFFFFF","BFBFBFBFBFBFBFBFB"])
def test_mix_hamiltonian_2e(system,basis_set,select):
    mol= cl.Molecule(geometry=system,basis_set=basis_set,select=select,backend='pyscf')
    tqmol=tq.Molecule(basis_set=basis_set,geometry=system,backend='pyscf')
    U = mol.make_ansatz("SPA",edges=[(0,1)])
    U1=tqmol.make_ansatz("SPA",edges=[(0,1)])
    H = mol.make_hamiltonian()
    H1 = tqmol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    E1 = tq.ExpectationValue(H=H1, U=U1)
    result = tq.minimize(E,silent=True)
    result1 = tq.minimize(E1,silent=True)
    assert numpy.isclose(result1.energy,result.energy,10**-5)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6","H 0.0 0.0 0.0\nLi 0.0 0.0 1.6","He 0. 0. 0."])
@pytest.mark.parametrize("basis_set",["6-31g"])
@pytest.mark.parametrize("select",["","FFFFFFFFFFF","BFBFBFBFBFBFBFBFB"])
def test_mix_hamiltonian_2e_two_qubits(system,basis_set,select):
    mol= cl.Molecule(geometry=system,basis_set=basis_set,select=select,condense=False,two_qubit=True,backend='pyscf')
    tqmol=tq.Molecule(basis_set=basis_set,geometry=system,backend='pyscf')
    U = mol.make_ansatz("SPA",edges=[(0,1)])
    U1=tqmol.make_ansatz("SPA",edges=[(0,1)])
    H = mol.make_hamiltonian()
    H1 = tqmol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    E1 = tq.ExpectationValue(H=H1, U=U1)
    result = tq.minimize(E)
    result1 = tq.minimize(E1)
    assert numpy.isclose(result1.energy,result.energy,10**-5)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize("select",["FBFBFFBFBFBFB","BFBFBFBFBFBFBF","FFFFFFFFFFF"])
def test_mix_hamiltonian_4e(system,select):
    mol= cl.Molecule(geometry=system,basis_set="sto-6g",select=select,backend='pyscf')
    tqmol=tq.Molecule(basis_set="sto-6g",geometry=system,backend='pyscf')
    if(system=="H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"):
        edges = [(0, 2, 4), (1, 3, 5)]
    else:
        edges = [(0, 2), (1, 3)]
    U1=tqmol.make_ansatz("SPA",edges=edges)
    H1 = tqmol.make_hamiltonian()
    U = mol.make_ansatz("SPA",edges=edges)
    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    E1 = tq.ExpectationValue(H=H1, U=U1)
    result = tq.minimize(E,silent=True)
    result1 = tq.minimize(E1,silent=True)
    assert numpy.isclose(result1.energy,result.energy,10**-5)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize("select",["FBFBFBFBFBFBFB","BFBFBFBFBFBFBFB","FFFFFFFFFFFF"])
def test_mix_hamiltonian_4e_two_qubits(system,select):
    mol= cl.Molecule(geometry=system,basis_set="sto-6g",select=select,integral_tresh=0.,condense=False,two_qubit=True,backend='pyscf')
    tqmol=tq.Molecule(basis_set="sto-6g",geometry=system,backend='pyscf')
    if(system=="H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"):
        edges = [(0, 2, 4), (1, 3, 5)]
    else:
        edges = [(0, 2), (1, 3)]
    U1=tqmol.make_ansatz("SPA",edges=edges)
    U = mol.make_ansatz("SPA",edges=edges)
    H = mol.make_hamiltonian()
    H1 = tqmol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    E1 = tq.ExpectationValue(H=H1, U=U1)
    result = tq.minimize(E)
    result1 = tq.minimize(E1)
    fin=tq.simulate(E,result.variables)
    fin1 = tq.simulate(E1, result1.variables)
    assert numpy.isclose(fin,fin1,10**-5)

@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6","H 0.0 0.0 0.0\nLi 0.0 0.0 1.6"])
@pytest.mark.parametrize("select",["FFFFFFFFFFFFFFF","FBFBFBFBFBFBFB"]) #every selection should work, but to perfome allowed excitations
@pytest.mark.parametrize("excit",[[(0,4)],[(0,2),(1,3)],[(1,5)]])
def test_exc_gate_energy_2e(system,select,excit):
    mol_mix = cl.Molecule(geometry=system, basis_set="6-31g", select=select,condense=False,backend='pyscf')
    mol_jw = tq.Molecule(basis_set="6-31g", geometry=system,backend='pyscf')

    U_jw = mol_jw.prepare_reference()
    U_mix = mol_mix.make_reference()

    U_jw += mol_jw.make_excitation_gate(indices=excit,angle="a")
    U_mix += mol_mix.make_excitation_gate(indices=excit,angle="a")

    H_jw = mol_jw.make_hamiltonian()
    H_mix = mol_mix.make_hamiltonian()

    E_jw = tq.ExpectationValue(H=H_jw, U=U_jw)
    E_mix = tq.ExpectationValue(H=H_mix, U=U_mix)

    result_jw = tq.minimize(E_jw)
    result_mix = tq.minimize(E_mix)

    Em_jw=tq.simulate(E_jw, result_jw.variables)
    Em_mix = tq.simulate(E_mix, result_mix.variables)
    assert numpy.isclose(Em_jw,Em_mix,10**-4)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize("select",["JJJJJJJJJ","JHJHJHJHJH"]) #every selection should work, but to perfome allowed excitations
@pytest.mark.parametrize("excit",[ [(0,4)] , [(2,4),(3,5)] , [(1,5)] , [(2,6),(3,7),(1,5)] , [(2,4),(3,5),(1,7),(0,6)] ])
def test_exc_gate_energy_4e(system,select,excit):
    mol_mix = cl.Molecule(geometry=system, basis_set="sto-6g", select=select,condense=False,backend='pyscf')
    mol_jw = tq.Molecule(basis_set="sto-6g", geometry=system,backend='pyscf')

    U_jw = mol_jw.prepare_reference()
    U_mix = mol_mix.make_reference()

    U_jw += mol_jw.make_excitation_gate(indices=excit, angle="a")
    U_mix += mol_mix.make_excitation_gate(indices=excit, angle="a")

    H_jw = mol_jw.make_hamiltonian()
    H_mix = mol_mix.make_hamiltonian()

    E_jw = tq.ExpectationValue(H=H_jw, U=U_jw)
    E_mix = tq.ExpectationValue(H=H_mix, U=U_mix)

    result_jw = tq.minimize(E_jw)
    result_mix = tq.minimize(E_mix)

    Em_jw = tq.simulate(E_jw, result_jw.variables)
    Em_mix = tq.simulate(E_mix, result_mix.variables)

    assert numpy.isclose(Em_jw, Em_mix, 10 ** -4)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6","H 0.0 0.0 0.0\nLi 0.0 0.0 1.6"])
@pytest.mark.parametrize("select",["FFFFFFFFFFF","FBFBFBFBFB"]) #every selection should work, but to perfome allowed excitations easily
@pytest.mark.parametrize("excit",[ [(0,4)] , [(0,2),(1,3)] , [(1,5)] ])
def test_exc_gate_fidelity_2e(system,select,excit):
    mol_mix = cl.Molecule(geometry=system, basis_set="6-31g", select=select,condense=False,backend='pyscf')
    mol_jw = tq.Molecule(basis_set="6-31g", geometry=system,backend='pyscf')

    U_jw = mol_jw.prepare_reference()
    U_mix = mol_mix.prepare_reference()

    U_jw += mol_jw.make_excitation_gate(indices=excit,angle="a")
    U_mix += mol_mix.make_excitation_gate(indices=excit,angle="a")

    H_jw = mol_jw.make_hamiltonian()
    H_mix = mol_mix.make_hamiltonian()

    E_jw = tq.ExpectationValue(H=H_jw, U=U_jw)
    E_mix = tq.ExpectationValue(H=H_mix, U=U_mix)

    result_jw = tq.minimize(E_jw)
    result_mix = tq.minimize(E_mix)

    U_mix += mol_mix.to_jw()
    U_mix.n_qubits = U_jw.n_qubits
    wfn_jw = tq.simulate(U_jw,variables=result_jw.variables)
    wfn_mix = tq.simulate(U_mix,variables=result_mix.variables)
    F = abs(wfn_jw.inner(wfn_mix))
    assert numpy.isclose(F,1.0,10**-4)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize("select",["FFFFFFFFFFF","FBFBFBFBFB"]) #every selection should work, but to perfome allowed excitations easily
@pytest.mark.parametrize("excit",[ [(0,4)] , [(2,4),(3,5)] , [(1,5)] , [(2,6),(3,7),(1,5)], [(2,4),(3,5),(1,7),(0,6)] ])
def test_exc_gate_fidelity_4e(system,select,excit):
    mol_mix = cl.Molecule(geometry=system, basis_set="sto-6g", select=select,condense=False,backend='pyscf')
    mol_jw = tq.Molecule(basis_set="sto-6g", geometry=system,backend='pyscf')

    U_jw = mol_jw.prepare_reference()
    U_mix = mol_mix.prepare_reference()

    U_jw += mol_jw.make_excitation_gate(indices=excit,angle="a")
    U_mix += mol_mix.make_excitation_gate(indices=excit,angle="a")

    H_jw = mol_jw.make_hamiltonian()
    H_mix = mol_mix.make_hamiltonian()

    E_jw = tq.ExpectationValue(H=H_jw, U=U_jw)
    E_mix = tq.ExpectationValue(H=H_mix, U=U_mix)

    result_jw = tq.minimize(E_jw)
    result_mix = tq.minimize(E_mix)

    U_mix += mol_mix.to_jw()
    U_mix.n_qubits=U_jw.n_qubits
    wfn_jw = tq.simulate(U_jw,variables=result_jw.variables)
    wfn_mix = tq.simulate(U_mix,variables=result_mix.variables)
    F = abs(wfn_jw.inner(wfn_mix))

    assert numpy.isclose(F,1.0,10**-4)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"])
@pytest.mark.parametrize("select",["BBBBBBBBBBBBB", "FFFFFFFFFFFFFFFFFFFFF","FBFBFBFBFB"]) #every selection should work, but to perfome allowed excitations easily
@pytest.mark.parametrize("two_qubit",[True, False])
def test_reference(system, select, two_qubit):
    molecule = tq.Molecule(geometry=system, basis_set="sto-3g",backend='pyscf')
    hf = molecule.compute_energy(method="hf")

    hybrid = cl.Molecule(geometry=system, basis_set="sto-3g", select=select, two_qubit=two_qubit,backend='pyscf')
    U = hybrid.prepare_reference()
    H = hybrid.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    test = tq.simulate(E)
    assert numpy.isclose(hf, test, atol=1.e-4)

    # test if HybridMolecule kills the PySCF interface
    hf2 = molecule.compute_energy(method="hf")

    assert numpy.isclose(hf, hf2, atol=1.e-4)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2","H 0.0 0.0 0.0\nH 0.0 0.0 1.6","H 0.0 0.0 0.0\nLi 0.0 0.0 1.6"])
@pytest.mark.parametrize("hcb_optimization",[True, False])
@pytest.mark.parametrize("neglect_z",[True, False])
@pytest.mark.parametrize("order",[1, 2])
def test_UpCCGD_hcb(system,hcb_optimization,neglect_z,order):

    mol = cl.Molecule(geometry=system,basis_set="sto-6g",select="",condense=False,backend="pyscf")
    tqmol = tq.Molecule(geometry=system,basis_set="sto-6g",backend="pyscf")

    H = mol.make_hamiltonian()
    tqH = tqmol.make_hamiltonian()

    U = mol.make_ansatz("UPCCGD",include_reference=True,hcb_optimization=hcb_optimization,neglect_z=neglect_z,order=order)
    tqU = tqmol.make_ansatz("UPCCGD",include_reference=True,hcb_optimization=hcb_optimization,neglect_z=neglect_z,order=order)

    E = tq.ExpectationValue(H=H, U=U)
    tqE = tq.ExpectationValue(H=tqH, U=tqU)
    res = tq.minimize(E, silent=True)
    fin = tq.simulate(E, res.variables)
    tqres = tq.minimize(tqE, silent=True)
    tqfin = tq.simulate(tqE, tqres.variables)
    wfn =tq.simulate(U + mol.to_jw(), variables=res.variables)
    tqwfn = tq.simulate(tqU , variables=tqres.variables)
    assert numpy.isclose(fin,tqfin,atol=1.e-4)
    assert numpy.isclose(abs(tqwfn.inner(wfn)),1.0,atol=1.e-4)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2","H 0.0 0.0 0.0\nH 0.0 0.0 1.6","H 0.0 0.0 0.0\nLi 0.0 0.0 1.6"])
@pytest.mark.parametrize("hcb_optimization",[True, False])
@pytest.mark.parametrize("neglect_z",[True, False])
@pytest.mark.parametrize("mix_sd",[True, False])
def test_UpCCGSD_jw(system,hcb_optimization,neglect_z,mix_sd):
    if hcb_optimization and mix_sd:
        assert True
    else:
        mol = cl.Molecule(geometry=system, basis_set="sto-6g", select="FFFFFFFFFFFFFFFFFFFFF", condense=False,backend='pyscf')
        tqmol = tq.Molecule(geometry=system, basis_set="sto-6g",backend='pyscf')

        H = mol.make_hamiltonian()
        tqH = tqmol.make_hamiltonian()

        U = mol.make_ansatz("UPCCGSD", include_reference=True, hcb_optimization=hcb_optimization, neglect_z=neglect_z,
                            mix_sd=mix_sd)
        tqU = tqmol.make_ansatz("UPCCSGD", include_reference=True, hcb_optimization=hcb_optimization,
                                neglect_z=neglect_z, mix_sd=mix_sd)

        E = tq.ExpectationValue(H=H, U=U)
        tqE = tq.ExpectationValue(H=tqH, U=tqU)
        res = tq.minimize(E, silent=True)
        fin = tq.simulate(E, res.variables)
        tqres = tq.minimize(tqE, silent=True)
        tqfin = tq.simulate(tqE, tqres.variables)
        wfn = tq.simulate(U , variables=res.variables)
        tqwfn = tq.simulate(tqU, variables=tqres.variables)
        assert numpy.isclose(fin, tqfin, atol=1.e-4)
        assert numpy.isclose(abs(tqwfn.inner(wfn)), 1.0, atol=1.e-4)
@pytest.mark.parametrize("system",["H 0.0 0.0 0.0\nH 0.0 0.0 1.6\nH 0.0 0.0 3.2\nH 0.0 0.0 4.8","H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2","H 0.0 0.0 0.0\nH 0.0 0.0 1.6","H 0.0 0.0 0.0\nLi 0.0 0.0 1.6"])
@pytest.mark.parametrize("add_singles",[True, False])
def test_UCCSD_jw(system,add_singles):
    '''
    Take care, expensive test
    '''
    mol = cl.Molecule(geometry=system, basis_set="sto-6g", select="FFFFFFFFFFFFFFFFFFFFF", condense=False,backend='pyscf')
    tqmol = tq.Molecule(geometry=system, basis_set="sto-6g",backend='pyscf')

    H = mol.make_hamiltonian()
    tqH = tqmol.make_hamiltonian()

    U = mol.make_ansatz("UCCSD",add_singles=add_singles)
    tqU = tqmol.make_ansatz("UCCSD",add_singles=add_singles)

    E = tq.ExpectationValue(H=H, U=U)
    tqE = tq.ExpectationValue(H=tqH, U=tqU)
    res = tq.minimize(E, silent=True)
    fin = tq.simulate(E, res.variables)
    tqres = tq.minimize(tqE, silent=True)
    tqfin = tq.simulate(tqE, tqres.variables)
    wfn = tq.simulate(U, variables=res.variables)
    tqwfn = tq.simulate(tqU, variables=tqres.variables)
    assert numpy.isclose(fin, tqfin, atol=1.e-4)
    assert numpy.isclose(abs(tqwfn.inner(wfn)), 1.0, atol=1.e-4)
