from pyscf import gto, scf
from pyscf.tools import cubegen
import tequila as tq
from tequila import TequilaException
def plot_MO(molecule:tq.Molecule=None, filename:str=None, orbital:list=None, print_orbital:bool=True, density:bool=False, mep:bool=False):
    '''
    Small function to save the MOs into Cube files
    Parameters
    ----------
    filename : Cube file will be saved as name+orb_index
    orbital: index of the orbitals to save
    molecule: molecule to plot the orbitals from
    print_orbital: whether to print the MOs
    density: whether to print the electron density
    mep: whether to plot the molecular electrostatic potential
    TODO ADD PATH TO SAVE FILES
    '''
    if molecule is None:
        raise TequilaException("No Molecule to save orbitals from")
    if orbital is None:
        orbital = [*range(len(molecule.integral_manager.orbital_coefficients))]
    if filename is None:
        filename = molecule.parameters.name
    pmol = gto.Mole()
    pmol.build(atom=molecule.parameters.geometry, basis=molecule.parameters.basis_set, spin=0)
    if(density or mep):
        mf = scf.RHF(pmol).run()
        mf.mo_coeff=molecule.integral_manager.orbital_coefficients
    if(print_orbital):
        for i in orbital:
            cubegen.orbital(pmol, str(i) +"_" + filename + "_MO.cube", molecule.integral_manager.orbital_coefficients[:, i])
    if(density):
        cubegen.density(pmol, filename + '_density.cube', mf.make_rdm1())
    if(mep):
        cubegen.mep(pmol, filename + '_mep.cube', mf.make_rdm1())