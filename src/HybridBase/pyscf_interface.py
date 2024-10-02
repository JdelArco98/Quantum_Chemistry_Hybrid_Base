from tequila import TequilaException
from .QuantumChemistryHybridBase import QuantumChemistryHybridBase
from tequila.quantumchemistry import ParametersQC, NBodyTensor
import pyscf

import numpy, typing


class OpenVQEEPySCFException(TequilaException):
    pass



class QuantumChemistryPySCF(QuantumChemistryHybridBase):
    def __init__(self, parameters: ParametersQC,select: typing.Union[str,dict]={},
                 transformation: typing.Union[str, typing.Callable] = None,
                 *args, **kwargs):

        if "one_body_integrals" not in kwargs:

            geometry = parameters.get_geometry()
            pyscf_geomstring = ""
            for atom in geometry:
                pyscf_geomstring += "{} {} {} {};".format(atom[0], atom[1][0], atom[1][1], atom[1][2])

            if "point_group" in kwargs:
                point_group = kwargs["point_group"]
            else:
                point_group = None

            mol = pyscf.gto.Mole()
            mol.atom = pyscf_geomstring
            mol.basis = parameters.basis_set
            mol.charge = parameters.charge

            if point_group is not None:
                if point_group.lower() != "c1":
                    mol.symmetry = True
                    mol.symmetry_subgroup = point_group
                else:
                    mol.symmetry = False
            else:
                mol.symmetry = True

            mol.build(parse_arg=False)

            # solve restricted HF
            mf = pyscf.scf.RHF(mol)
            mf.verbose = False
            if "verbose" in kwargs:
                mf.verbose = kwargs["verbose"]

            mf.kernel()

            # only works if point_group is not C1
            # otherwise PySCF uses a different SCF object
            # irrep information is however not critical to tequila
            if hasattr(mf, "get_irrep_nelec"):
                self.irreps = mf.get_irrep_nelec()
            else:
                self.irreps = None

            orbital_energies = mf.mo_energy

            # compute mo integrals
            mo_coeff = mf.mo_coeff
            h_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
            g_ao = mol.intor('int2e', aosym='s1')
            S = mol.intor_symmetric("int1e_ovlp")
            g_ao = NBodyTensor(elems=g_ao, ordering="mulliken")

            self.pyscf_molecule = mol
            self.point_group = mol.symmetry_subgroup

            kwargs["overlap_integrals"] = S
            kwargs["two_body_integrals"] = g_ao
            kwargs["one_body_integrals"] = h_ao
            kwargs["orbital_coefficients"] = mo_coeff
            kwargs["orbital_type"] = "hf"

            if "nuclear_repulsion" not in kwargs:
                kwargs["nuclear_repulsion"] = mol.energy_nuc()
        super().__init__(parameters=parameters, transformation=transformation,select=select, *args, **kwargs)
    def update_select(self,select:typing.Union[str,dict,list,tuple],n_orb:int):
        '''
        Parameters
        ----------
        select: codification of the transformation for each MO.

        Returns
        -------
        Updates the MO cofication data. Returns Instance of the class
        '''

        def verify_selection_str(select:str,n_orb:int)->dict:
            """
            Internal function
            Checks if the orbital selection string has the appropiated lenght, otherwise corrects it
            :return : corrected selection dict
            """
            sel = {}
            if (len(select) == n_orb):
                pass
            elif (len(select) > n_orb):
                select = select[:n_orb]
            else:
                while (len(select) <  n_orb):
                    select += "B"
            for i in range(len(select)):
                if select[i] in ["F","B"]:
                    sel.update({i: select[i]})
                else:
                    TequilaException(f"Warning, encoding character not recognised on position {i}: {select[i]}.\n Please choose between F (Fermionic) and B (Bosonic).")
            return sel
        def verify_selection_dict(select:dict,n_orb:int)->dict:
            """
            Internal function
            Checks if the orbital selection dictionary has the appropiated lenght, otherwise corrects it
            :return : corrected selection dict
            """
            sel = {}
            for i in select:
                if i< n_orb:
                    sel.update({i:select[i]})
            for o in range(n_orb):
                if o not in select.keys():
                    select.update({o: "B"})
                elif select[o] not in ["F","B"]:
                    TequilaException("Warning, encoding character not recognised on entry {it}.\n Please choose between F (Fermionic) and B (Bosonic).".format(it={o:select[o]}))
            return sel
        def verify_selection_list(select:typing.Union[list,tuple],n_orb:int)->dict:
            """
            Internal function
            Checks if the orbital selection string has the appropiated lenght, otherwise corrects it
            :return : corrected selection dict
            """
            select = [*select]
            sel = {}
            if (len(select) ==  n_orb):
                pass
            elif (len(select) >  n_orb):
                select = select[: n_orb]
            else:
                while (len(select) <  n_orb):
                    select.append("B")
            for i in range(len(select)):
                if select[i] in ["F","B"]:
                    sel.update({i: select[i]})
                else:
                    TequilaException(f"Warning, encoding character not recognised on position {i}: {select[i]}.\n Please choose between F (Fermionic) and B (Bosonic).")
            return sel
        def select_to_list(select:dict):
            """
            Internal function
            Read the select string to make the proper Fer and Bos lists
            :return : list of MOs for the Bos, MOs and SOs for the Fer space
            """

            hcb = 0
            jworb = 0
            BOS_MO = []
            FER_MO = []
            FER_SO = []
            for i in select:
                if (select[i] == "B"):
                    BOS_MO.append(i)
                    hcb += 1
                elif (select[i] == "F"):
                    FER_MO.append(i)
                    FER_SO.append(2 * i)
                    FER_SO.append(2 * i + 1)
                    jworb += 1
                else:
                    print("Warning, codification not recognized: ,", i, " returning void lists")
                    return [], [], []
            self.bos_orb = hcb
            self.fer_orb = jworb
            return BOS_MO, FER_MO, FER_SO
        if type(select) is str:
            self.select = verify_selection_str(select=select,n_orb=n_orb)
        elif type(select) is dict:
            self.select = verify_selection_dict(select=select,n_orb=n_orb)
        elif type(select) is list or type(select) is tuple:
            self.select = verify_selection_list(select=select,n_orb=n_orb)
        else:
            try:
                self.select = verify_selection_list(select=select,n_orb=n_orb)
            except:
                TequilaException(f"Warning, encoding format not recognised: {type(select)}.\n Please choose either a Str, Dict, List or Tuple.")
        self.BOS_MO, self.FER_MO, self.FER_SO= select_to_list(self.select)
        return self
    def compute_fci(self, *args, **kwargs):
        from pyscf import fci
        c, h1, h2 = self.get_integrals(ordering="chem")
        norb = self.n_orbitals
        nelec = self.n_electrons
        e, fcivec = fci.direct_spin1.kernel(h1, h2.elems, norb, nelec, **kwargs)
        return e + c

    def compute_energy(self, method: str, *args, **kwargs) -> float:
        method = method.lower()

        if method == "hf":
            return self._get_hf(do_not_solve=False, **kwargs).e_tot
        elif method == "mp2":
            return self._run_mp2(**kwargs).e_tot
        elif method == "cisd":
            hf = self._get_hf(do_not_solve=False, **kwargs)
            return self._run_cisd(hf=hf, **kwargs).e_tot
        elif method == "ccsd":
            return self._run_ccsd(**kwargs).e_tot
        elif method == "ccsd(t)":
            ccsd = self._run_ccsd(**kwargs)
            return ccsd.e_tot + self._compute_perturbative_triples_correction(ccsd=ccsd, **kwargs)
        elif method == "fci":
            return self.compute_fci(**kwargs)
        else:
            raise TequilaException("unknown method: {}".format(method))

    def _get_hf(self, do_not_solve=True, **kwargs):
        c, h1, h2 = self.get_integrals(ordering="mulliken")
        norb = self.n_orbitals
        nelec = self.n_electrons

        mo_coeff = numpy.eye(norb)
        mo_occ = numpy.zeros(norb)
        mo_occ[:nelec // 2] = 2

        pyscf_mol = pyscf.gto.M(verbose=0, parse_arg=False)
        pyscf_mol.nelectron = nelec
        pyscf_mol.incore_anyway = True  # ensure that custom integrals are used
        pyscf_mol.energy_nuc = lambda *args: c

        hf = pyscf.scf.RHF(pyscf_mol)
        hf.get_hcore = lambda *args: h1
        hf.get_ovlp = lambda *args: numpy.eye(norb)
        hf._eri = pyscf.ao2mo.restore(8, h2.elems, norb)

        if do_not_solve:
            hf.mo_coeff = mo_coeff
            hf.mo_occ = mo_occ
        else:
            hf.kernel(numpy.diag(mo_occ))

        return hf

    def _run_ccsd(self, hf=None, **kwargs):
        from pyscf import cc
        if hf is None:
            hf = self._get_hf()
        ccsd = cc.RCCSD(hf)
        ccsd.kernel()
        return ccsd

    def _compute_perturbative_triples_correction(self, ccsd=None, **kwargs) -> float:
        if ccsd is None:
            ccsd = self._run_ccsd(**kwargs)
        ecorr = ccsd.ccsd_t()
        return ecorr

    def _run_cisd(self, hf=None, **kwargs):
        from pyscf import ci
        if hf is None:
            hf = self._get_hf(**kwargs)
        cisd = ci.RCISD(hf)
        cisd.kernel()
        return cisd

    def _run_mp2(self, hf=None, **kwargs):
        from pyscf import mp
        if hf is None:
            hf = self._get_hf(**kwargs)
        mp2 = mp.MP2(hf)
        mp2.kernel()
        return mp2

    def __str__(self):
        base = super().__str__()
        try:
            if hasattr(self, "pyscf_molecule"):
                base += "{:15} : {} ({})\n".format("point_group", self.pyscf_molecule.groupname,
                                                   self.pyscf_molecule.topgroup)
            if hasattr(self, "irreps"):
                base += "{:15} : {}\n".format("irreps", self.irreps)
        except:
            return base
        return base