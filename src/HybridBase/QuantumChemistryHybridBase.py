import copy
import warnings
from dataclasses import dataclass
from tequila import TequilaException, BitString, TequilaWarning
from tequila.hamiltonian import QubitHamiltonian
from tequila import Molecule
from tequila.hamiltonian.paulis import Sp, Sm, Zero,I

from tequila.circuit import QCircuit, gates
from tequila.objective.objective import Variable, Variables, ExpectationValue

from tequila.simulators.simulator_api import simulate
from tequila.utils import to_float
from tequila.quantumchemistry.chemistry_tools import ActiveSpaceData, FermionicGateImpl, prepare_product_state, ClosedShellAmplitudes, \
    Amplitudes, ParametersQC, NBodyTensor, IntegralManager
from tequila.quantumchemistry.qc_base import QuantumChemistryBase as qc_base
import typing, numpy, numbers
from itertools import product
import tequila.grouping.fermionic_functions as ff
from .encodings import known_encodings
import pyscf

class QuantumChemistryHybridBase(qc_base): #Should I heredate the QuantumChemistryBase? or something with the quantumchemistry.__init__()
    bos_mo = []
    fer_mo = []
    fer_so = []

    def __init__(self, parameters: ParametersQC,select: typing.Union[str,dict]={},transformation: typing.Union[str, typing.Callable] = None, active_orbitals: list = None,
                 frozen_orbitals: list = None, orbital_type: str = None,reference_orbitals: list = None, orbitals: list = None, *args, **kwargs):
        '''
        Parameters
        ----------
        select: codification of the transformation for each MO.
        parameters: the quantum chemistry parameters handed over as instance of the ParametersQC class (see there for content)
        transformation1: the fermion to qubit transformation (default is JordanWigner).
        transformation2: the boson to qubit transformation (default is Hard-Core Boson).
        active_orbitals: list of active orbitals (others will be frozen, if we have N-electrons then the first N//2 orbitals will be considered occpied when creating the active space)
        frozen_orbitals: convenience (will be removed from list of active orbitals)
        reference_orbitals: give list of orbitals that shall be considered occupied when creating a possible active space (default is the first N//2). The indices are expected to be total indices (including possible frozen orbitals in the counting)
        orbitals: information about the orbitals (should be in OrbitalData format, can be a dictionary)
        args
        kwargs
        '''
        if ("condense" in kwargs):
            self.condense = kwargs["condense"]
            kwargs.pop("condense")
        else:
            self.condense = True
        if ("two_qubit" in kwargs):
            self.two_qubit = kwargs["two_qubit"]
            kwargs.pop("two_qubit")
            if self.two_qubit: self.condense=False
        else:
            self.two_qubit = False
        if ("integral_tresh" in kwargs):
            self.integral_tresh = kwargs["integral_tresh"]
            kwargs.pop('integral_tresh')
        else:
            self.integral_tresh = 1.e-6
        self.parameters = parameters
        n_electrons = parameters.n_electrons
        if "n_electrons" in kwargs:
            n_electrons = kwargs["n_electrons"]

        if reference_orbitals is None:
            reference_orbitals = [i for i in range(n_electrons // 2)]
        self._reference_orbitals = reference_orbitals

        if orbital_type is None:
            orbital_type = "unknown"

        # no frozen core with native orbitals (i.e. atomics)
        overriding_freeze_instruction = orbital_type is not None and orbital_type.lower() == "native"
        # determine frozen core automatically if set
        # only if molecule is computed from scratch and not passed down from above
        overriding_freeze_instruction = overriding_freeze_instruction or n_electrons != parameters.n_electrons
        overriding_freeze_instruction = overriding_freeze_instruction or frozen_orbitals is not None
        if not overriding_freeze_instruction and self.parameters.frozen_core:
            n_core_electrons = self.parameters.get_number_of_core_electrons()
            if frozen_orbitals is None:
                frozen_orbitals = [i for i in range(n_core_electrons // 2)]

        # initialize integral manager
        if "integral_manager" in kwargs:
            self.integral_manager = kwargs["integral_manager"]
        else:
            self.integral_manager = self.initialize_integral_manager(active_orbitals=active_orbitals,
                                                                     reference_orbitals=reference_orbitals,
                                                                     orbitals=orbitals, frozen_orbitals=frozen_orbitals,
                                                                     orbital_type=orbital_type,
                                                                     basis_name=self.parameters.basis_set, *args,
                                                                     **kwargs)

        if orbital_type is not None and orbital_type.lower() == "native":
            self.integral_manager.transform_to_native_orbitals()

        self.update_select(select,n_orb=self.n_orbitals)
        self.transformation = self._initialize_transformation(transformation=transformation,select=self.select,two_qubit=self.two_qubit,condense=self.condense)
        self.up_then_down = self.transformation.up_then_down
        self._rdm1 = None
        self._rdm2 = None
    #Select Related Functions
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
                while (len(select) < n_orb):
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
                if select[i]<n_orb:
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
            if (len(select) == n_orb):
                pass
            elif (len(select) > n_orb):
                select = select[:n_orb]
            else:
                while (len(select) < n_orb):
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
                    return [], [],[]
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
    # Tranformation Related Function
    def _initialize_transformation(self, transformation=None, *args, **kwargs):
        """
        Helper Function to initialize the Fermion-to-Qubit Transformation
        Parameters
        ----------
        transformation: name of the transformation (passed down from __init__
        args
        kwargs

        Returns
        -------

        """

        if transformation is None:
            transformation = "JordanWigner"

        # filter out arguments to the transformation
        trafo_args = {k.split("__")[1]: v for k, v in kwargs.items() if
                      (hasattr(k, "lower") and "transformation__" in k.lower())}

        trafo_args["n_electrons"] = self.n_electrons
        trafo_args["n_orbitals"] = self.n_orbitals
        trafo_args["select"]= self.select
        if hasattr(transformation, "upper"):
            # format to conventions
            transformation = transformation.replace("_", "").replace("-", "").upper()
            encodings = known_encodings()
            if transformation in encodings:
                transformation = encodings[transformation](**trafo_args)
            else:
                raise TequilaException(
                    "Unkown Fermion-to-Qubit encoding {}. Try something like: {}".format(transformation,
                                                                                         list(encodings.keys())))
        return transformation
    # Hamiltonian Related Funcions
    def make_hamiltonian(self)->QubitHamiltonian:
        '''
        Returns
        -------
        Qubit Hamiltonian in the Fermion/Bosont-to-Qubit transformations defined in self.parameters
        '''
        def make_fermionic_hamiltonian() -> QubitHamiltonian:
            '''
            Internal function
            Returns
            -------
            Fermionic part of the total Hamiltonian
            '''
            H_FER = []
            for i in self.FER_SO:
                for j in self.FER_SO:
                    if ((i % 2) == (j % 2)):
                        H_FER.append((((j,1),(i,0)),self.h1[j // 2][i // 2]))
                    for k in self.FER_SO:
                        for l in self.FER_SO:
                            if ((i % 2) == (l % 2) and (k % 2) == (j % 2)):
                                H_FER.append((((l,1),(k,1),(j,0),(i,0)),0.5 * self._h2.elems[l // 2][k // 2][j // 2][i // 2]))
            return self.transformation(H_FER)
            pass
        def make_bosonic_hamiltonian() -> QubitHamiltonian:
            '''
            Internal function
            Returns
            -------
            Bosonic part of the total Hamiltonian
            '''
            H_BOS = []
            for i in self.BOS_MO:
                for j in self.BOS_MO:
                    e1 = self._h2.elems[j][j][i][i]
                    if (i == j):
                        e1 += 2 * self.h1[i][i]
                    if not self.two_qubit:
                        H_BOS.append((((2 * j, 1), (2 * i, 0)), e1))
                    else: H_BOS.append((((2 * j, 1),(2 * j+1, 1),(2 * i+1, 0),(2 * i, 0)), e1))
                    if (i != j):
                        e2 = 2 * self._h2.elems[j][i][i][j] - self._h2.elems[j][i][j][i]
                        if not self.two_qubit:
                            H_BOS.append((((2 * i, 1), (2 * i, 0), (2 * j, 1), (2 * j, 0)), e2))
                        else:
                            H_BOS.append((((2*i,1),(2*i+1,1),(2*i,0),(2*i+1,0),(2*j,1),(2*j+1,1),(2*j,0),(2*j+1,0)),e2))
            return self.transformation(H_BOS)
        def make_interaction_hamiltonian() -> QubitHamiltonian:
            '''
            Returns
            -------
            Fermionic-Bosonic Interaction part of the total Hamiltonian
            '''
            H_INT = []
            h2 = self._h2.elems
            for j in self.BOS_MO:
                for k in self.FER_SO:
                    for l in self.FER_SO:
                        if (k % 2) != (l % 2):  # hhjj
                            if not self.two_qubit:
                                H_INT.append((((2*j,1),(l,0),(k,0)),((l % 2) - (k % 2)) * 0.5 * h2[j][j][l // 2][k // 2]))
                            else: H_INT.append((((2*j+k%2,1),(2*j+l%2,1),(l,0),(k,0)),0.5 * h2[j][j][l // 2][k // 2]))
                        if (k % 2) != (l % 2):  # jjhh
                            if not self.two_qubit:
                                H_INT.append((((l,1),(k,1),(2*j,0)),((k % 2) - (l % 2)) * 0.5 * h2[l // 2][k // 2][j][j]))
                            else: H_INT.append((((l,1),(k,1),(2*j+k%2,0),(2*j+l%2,0)),0.5 * h2[l // 2][k // 2][j][j]))
                        if ((k % 2) == (l % 2)):
                            if not self.two_qubit:
                                e2 = 2 * h2[k // 2][j][j][l // 2] + 2 * h2[j][k // 2][l // 2][j] - h2[k // 2][j][l // 2][j] - h2[j][k // 2][j][l // 2]
                                H_INT.append((((2*j,1),(2*j,0),(k,1),(l,0)),0.5 * e2))
                            else:
                                e2 = h2[j][k // 2][l // 2][j]+h2[k // 2][j][j][l // 2]
                                H_INT.append((((j*2,1),(j*2,0),(k,1),(l,0)),e2))
                                H_INT.append((((j*2+1,1),(j*2+1,0),(k,1),(l,0)),e2))
                                H_INT.append((((k,1),(l,0),(j*2+k%2,1),(j*2+k%2,0)),-0.5*(h2[k // 2][j][l // 2][j]+h2[j][k // 2][j][l // 2])))

            return self.transformation(H_INT)
        self.C, self.h1, self._h2 = self.get_integrals(ordering="openfermion")
        H = make_fermionic_hamiltonian() + make_bosonic_hamiltonian() + make_interaction_hamiltonian() + self.C
        return H.simplify(self.integral_tresh)

    def compute_rdms(self, U: QCircuit = None, variables: Variables = None, spin_free: bool = True,
                     get_rdm1: bool = True, get_rdm2: bool = True, ordering="dirac", use_hcb: bool = False,
                     rdm_trafo: QubitHamiltonian = None, evaluate=True):
        """
        Computes the one- and two-particle reduced density matrices (rdm1 and rdm2) given
        a unitary U. This method uses the standard ordering in physics as denoted below.
        Note, that the representation of the density matrices depends on the qubit transformation
        used. The Jordan-Wigner encoding corresponds to 'classical' second quantized density
        matrices in the occupation picture.

        We only consider real orbitals and thus real-valued RDMs.
        The matrices are set as private members _rdm1, _rdm2 and can be accessed via the properties rdm1, rdm2.

        .. math :
            \\text{rdm1: } \\gamma^p_q = \\langle \\psi | a^p a_q | \\psi \\rangle
                                     = \\langle U 0 | a^p a_q | U 0 \\rangle
            \\text{rdm2: } \\gamma^{pq}_{rs} = \\langle \\psi | a^p a^q a_s a_r | \\psi \\rangle
                                             = \\langle U 0 | a^p a^q a_s a_r | U 0 \\rangle

        Parameters
        ----------
        U :
            Quantum Circuit to achieve the desired state \\psi = U |0\\rangle, non-optional
        variables :
            If U is parametrized, then need to hand over a set of fixed variables
        spin_free :
            Set whether matrices should be spin-free (summation over spin) or defined by spin-orbitals
        get_rdm1, get_rdm2 :
            Set whether either one or both rdm1, rdm2 should be computed. If both are needed at some point,
            it is recommended to compute them at once.
        rdm_trafo :
            The rdm operators can be transformed, e.g., a^dagger_i a_j -> U^dagger a^dagger_i a_j U,
            where U represents the transformation. The default is set to None, implying that U equas the identity.
        evaluate :
            if true, the tequila expectation values are evaluated directly via the tq.simulate command.
            the protocol is optimized to avoid repetation of wavefunction simulation
            if false, the rdms are returned as tq.QTensors
        Returns
        -------
        """
        # Check whether unitary circuit is not 0
        if U is None:
            raise TequilaException('Need to specify a Quantum Circuit.')
        # Check whether transformation is BKSF.
        # Issue here: when a single operator acts only on a subset of qubits, BKSF might not yield the correct
        # transformation, because it computes the number of qubits incorrectly in this case.
        # A hotfix such as for symmetry_conserving_bravyi_kitaev would require deeper changes, thus omitted for now
        if type(self.transformation).__name__ == "BravyiKitaevFast":
            raise TequilaException(
                "The Bravyi-Kitaev-Superfast transformation does not support general FermionOperators yet.")
        # Set up number of spin-orbitals and molecular orbitals respectively

        # Check whether unitary circuit is not 0
        if U is None:
            raise TequilaException('Need to specify a Quantum Circuit.')

        if not spin_free:
            raise TequilaException(
                "HybridMolecule.compute_rdms : spin_free = False not implemented\nSuggesting to compute spin-free RMDs and map to spin RDM yourself")

        def _build_1bdy_operators_mix() -> list:
            """ Returns BOS one-body operators as a symmetry-reduced list of QubitHamiltonians """
            # Exploit symmetry pq = qp (not changed by spin-summation)
            ops = []
            for p in range(self.n_orbitals):
                for q in range(p + 1):
                    if (self.select[p] == "F" and self.select[q] == "F"):
                        op_tuple = [(((2 * p, 1), (2 * q, 0)),1)]
                        op = self.transformation(op_tuple)
                        op_tuple = [(((2 * p + 1, 1), (2 * q + 1, 0)),1)]
                        op += self.transformation(op_tuple)
                    elif (p == q and self.select[p] == "B"):
                        op_tuple = [(((2 * p, 1), (2 * q, 0)),2)]
                        op = self.transformation(op_tuple)
                    else:
                        op = Zero()
                    ops += [op]
            return ops

        def __case_2bdy(i: int, j: int, k: int, l: int) -> int:
            ''' Returns 1 if allowed term all in J, 2 if all H, 3 mixed, 0 else'''
            list = [self.select[i], self.select[j], self.select[k], self.select[l]]
            b = list.count("B")
            f = list.count("F")
            if (f == 4):
                return 1
            elif (b == 4):
                return 2
            elif (f == b):
                return 3
            else:
                return 0

        def _build_2bdy_operators_mix() -> list:
            """ Returns BOS two-body operators as a symmetry-reduced list of QubitHamiltonians """
            # Exploit symmetries pqrs = qpsr (due to spin summation, '-pqsr = -qprs' drops out)
            #                and      = rspq
            ops = []
            for p, q, r, s in product(range(self.n_orbitals), repeat=4):
                if p * self.n_orbitals + q >= r * self.n_orbitals + s and (p >= q or r >= s):
                    case = __case_2bdy(p, q, r, s)
                    if (case == 1):  # case JJJJ
                        # Spin aaaa
                        op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0)),1)] if (p != q and r != s) else [((),0)]
                        op = self.transformation(op_tuple)
                        op -= op.dagger()
                        # Spin abba
                        op_tuple = [(((2 * p, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r, 0)),1)] if (2 * p != 2 * q + 1 and 2 * s + 1 != 2 * r) else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        # Spin baab
                        op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * s, 0), (2 * r + 1, 0)),1)] if (2 * p + 1 != 2 * q and 2 * s != 2 * r + 1) else [((),0)]
                        op += self.transformation(op_tuple)
                        # Spin bbbb
                        op_tuple = [(((2 * p + 1, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0)),1)] if (p != q and r != s) else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        ops += [op]
                    elif (case == 2):  # case HHHH
                        # Spin aaaa+ bbbb dont allow p=q=r=s  orb ijji
                        op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * q, 0), (2 * p, 0)),-2)] if (p != q and r != s and p == s and q == r) else [((),0)]
                        op = self.transformation(op_tuple)
                        # Spin abba+ baab allow p=q=r=s orb iijj
                        op_tuple = [(((2 * p, 1), (2 * s, 0)),1)] if (p == q and s == r) else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        # Spin abba+ baab dont allow p=q=r=s orb ijij
                        op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * s, 0)),4)] if (p != q and r != s and p == r and s == q) else [((),0)]
                        op += self.transformation(op_tuple)
                        ops += [op]
                    elif (case == 3):  # case HJJH+JHHJ+HJHJ+JHJH+HHJJ+JJHH
                        # uddu+duud hhjj
                        op_tuple = [(((2 * p, 1), (2 * r + 1, 0), (2 * s, 0)),0.5)] if (p == q and self.select[p] == "B") else [((),0)]
                        opa = self.transformation(op_tuple)
                        op = opa - opa.dagger()
                        op_tuple = [(((2 * p, 1), (2 * r, 0), (2 * s + 1, 0)),0.5)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        # uddu+duud jjhh
                        op_tuple = [(((2 * p, 1), (2 * q + 1, 1), (2 * r, 0)),0.5)] if r == s and self.select[r] == "B" else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * r, 0)),0.5)] if r == s and self.select[r] == "B" else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        # uddu+duud hjjh
                        op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * p, 0)),-.5)] if (p == s and self.select[p] == "B") else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        op_tuple = [(((2 * p, 1), (2 * q + 1, 1), (2 * r + 1, 0), (2 * p, 0)),-.5)] if (p == s and self.select[p] == "B") else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        # uddu+duud jhhj
                        op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * q, 0), (2 * s, 0)),-.5)] if (r == q and self.select[r] == "B") else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * q, 0), (2 * s + 1, 0)),-.5)] if (r == q and self.select[r] == "B") else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        # dddd+uuuu hjhj
                        op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * p, 0), (2 * s, 0)),1)] if p == r and self.select[p] == "B" else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        op_tuple = [(((2 * p, 1), (2 * q + 1, 1), (2 * p, 0), (2 * s + 1, 0)),1)] if p == r and self.select[p] == "B" else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        # dddd+uuuu jhjh
                        op_tuple = [(((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * q, 0)),1)] if (s == q and self.select[s] == "B") else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        op_tuple = [(((2 * p + 1, 1), (2 * q, 1), (2 * r + 1, 0), (2 * q, 0)),1)] if (s == q and self.select[s] == "B") else [((),0)]
                        opa = self.transformation(op_tuple)
                        op += opa - opa.dagger()
                        ops += [op]
                    else:
                        ops += [Zero()]
            return ops

        def _assemble_rdm1_mix(evals) -> numpy.ndarray:
            """
            Returns BOS one-particle RDM built by symmetry conditions
            """
            N = self.n_orbitals
            rdm1 = numpy.zeros([N, N])
            ctr: int = 0
            for p in range(N):
                for q in range(p + 1):
                    rdm1[p, q] = evals[ctr]
                    # Symmetry pq = qp
                    rdm1[q, p] = rdm1[p, q]
                    ctr += 1
            return rdm1

        def _assemble_rdm2_mix(evals) -> numpy.ndarray:
            """ Returns spin-free two-particle RDM built by symmetry conditions """
            ctr: int = 0
            rdm2 = numpy.zeros([self.n_orbitals, self.n_orbitals, self.n_orbitals, self.n_orbitals])
            for p, q, r, s in product(range(self.n_orbitals), repeat=4):
                if p * self.n_orbitals + q >= r * self.n_orbitals + s and (p >= q or r >= s):
                    rdm2[p, q, r, s] = evals[ctr]
                    # Symmetry pqrs = rspq
                    rdm2[r, s, p, q] = rdm2[p, q, r, s]
                    ctr += 1
            # Further permutational symmetry: pqrs = qpsr
            for p, q, r, s in product(range(self.n_orbitals), repeat=4):
                if p >= q or r >= s:
                    rdm2[q, p, s, r] = rdm2[p, q, r, s]
            return rdm2

        # Build operator lists
        qops = []
        qops += _build_1bdy_operators_mix() if get_rdm1 else []
        qops += _build_2bdy_operators_mix() if get_rdm2 else []
        # Compute expected values
        evals = simulate(ExpectationValue(H=qops, U=U, shape=[len(qops)]), variables=variables)
        # Split expectation values in 1- and 2-particle expectation values
        if get_rdm1:
            len_1 = self.n_orbitals * (self.n_orbitals + 1) // 2
        else:
            len_1 = 0
        evals_1, evals_2 = evals[:len_1], evals[len_1:]
        rdm1 = []
        rdm2 = []
        # Build matrices using the expectation values
        rdm1 = _assemble_rdm1_mix(evals_1) if get_rdm1 else rdm1
        rdm2 = _assemble_rdm2_mix(evals_2) if get_rdm2 else rdm2
        if get_rdm2:
            rdm2_ = NBodyTensor(elems=rdm2, ordering="dirac")
            rdm2_.reorder(to=ordering)
            rdm2 = rdm2_.elems

        if get_rdm1:
            if get_rdm2:
                self._rdm2 = rdm2
                self._rdm1 = rdm1
                return rdm1, rdm2
            else:
                self._rdm1 = rdm1
                return rdm1
        elif get_rdm2:
            self._rdm2 = rdm2
            return rdm2
        else:
            warnings.warn("compute_rdms called with instruction to not compute?", TequilaWarning)
    def optimize_orbitals(self,molecule, circuit:QCircuit=None, vqe_solver=None, pyscf_arguments=None, silent=False, vqe_solver_arguments=None, initial_guess=None, return_mcscf=False, molecule_factory=None,molecule_arguments=None ,*args, **kwargs):
        """
        Interface with tq.quantumchemistry.optimize_orbitals
        Parameters
        ----------
        molecule: The tequila molecule whose orbitals are to be optimized
        circuit: The circuit that defines the ansatz to the wavefunction in the VQE
                 can be None, if a customized vqe_solver is passed that can construct a circuit
        vqe_solver: The VQE solver (the default - vqe_solver=None - will take the given circuit and construct an expectationvalue out of molecule.make_hamiltonian and the given circuit)
                    A customized object can be passed that needs to be callable with the following signature: vqe_solver(H=H, circuit=self.circuit, molecule=molecule, **self.vqe_solver_arguments)
        pyscf_arguments: Arguments for the MCSCF structure of PySCF, if None, the defaults are {"max_cycle_macro":10, "max_cycle_micro":3} (see here https://pyscf.org/pyscf_api_docs/pyscf.mcscf.html)
        silent: silence printout
        vqe_solver_arguments: Optional arguments for a customized vqe_solver or the default solver
                              for the default solver: vqe_solver_arguments={"optimizer_arguments":A, "restrict_to_hcb":False} where A holds the kwargs for tq.minimize
                              restrict_to_hcb keyword controls if the standard (in whatever encoding the molecule structure has) Hamiltonian is constructed or the hardcore_boson hamiltonian
        initial_guess: Initial guess for the MCSCF module of PySCF (Matrix of orbital rotation coefficients)
                       The default (None) is a unit matrix
                       predefined commands are
                            initial_guess="random"
                            initial_guess="random_loc=X_scale=Y" with X and Y being floats
                            This initialized a random guess using numpy.random.normal(loc=X, scale=Y) with X=0.0 and Y=0.1 as defaults
        return_mcscf: return the PySCF MCSCF structure after optimization
        molecule_arguments: arguments to pass to molecule_factory or default molecule constructor | only change if you know what you are doing
        molecule_factory: callable function creates the molecule class
        args: just here for convenience
        kwargs: just here for conveniece

        Returns
        -------
            Optimized Tequila Hybrid Molecule
        """
        pass
    # Cicuit Related Functions
    def verify_excitation(self, indices: typing.Iterable[typing.Tuple[int, int]], warning:bool=True)->bool:
        """
        Checks if the Bosonic restriction are accomplished by the excitation
        TODO: Generalize for >2 electrons
        Parameters
        ----------
        :param indices: turple of pair turples (a_i,b_j) where the electron is excited from a_i to b_j
        :warning bool: change the action in case of forbiden excitations, from raising and Exception to return False
        :return : True if the excitations are allowed, raises an exception otherwise
        Returns
        -------
            Optimized Tequila Hybrid Molecule
        """
        pass

    def UR(self, i, j, angle=None, label=None, control=None, assume_real=True, *args, **kwargs):
        """
        Convenience function for orbital rotation circuit (rotating spatial orbital i and j) with standard naming of variables
        See arXiv:2207.12421 Eq.6 for UR(0,1)
        Parameters:
        ----------
            indices:
                tuple of two spatial(!) orbital indices
            angle:
                Numeric or hashable type or tequila objective. Default is None and results
                in automatic naming as ("R",i,j)
            label:
                can be passed instead of angle to have auto-naming with label ("R",i,j,label)
                useful for repreating gates with individual variables
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
        """
        pass

    def UC(self, i, j, angle=None, label=None, control=None, assume_real=True, *args, **kwargs):
        """
        Convenience function for orbital correlator circuit (correlating spatial orbital i and j through a spin-paired double excitation) with standard naming of variables
        See arXiv:2207.12421 Eq.22 for UC(1,2)

        Parameters:
        ----------
            indices:
                tuple of two spatial(!) orbital indices
            angle:
                Numeric or hashable type or tequila objective. Default is None and results
                in automatic naming as ("R",i,j)
            label:
                can be passed instead of angle to have auto-naming with label ("R",i,j,label)
                useful for repreating gates with individual variables
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
        """
        pass

    def make_excitation_gate(self, indices: typing.Iterable[typing.Tuple[int, int]], angle, control=None, assume_real=True, **kwargs)->QCircuit:
        """
        Initialize a fermionic excitation gate defined as

        .. math::
            e^{-i\\frac{a}{2} G}
        with generator defines by the indices [(p0,q0),(p1,q1),...] or [p0,q0,p1,q1 ...]
        .. math::
            G = i(\\prod_{k} a_{p_k}^\\dagger a_{q_k} - h.c.)

        Parameters
        ----------
            indices:
                List of tuples that define the generator
            angle:
                Numeric or hashable type or tequila objective
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
        """
        pass

    def make_excitation_generator(self, indices: typing.Iterable[typing.Tuple[int, int]], form: str = None, remove_constant_term: bool = True,neglect_z:bool=False) -> QubitHamiltonian:
        """
        Notes
        ----------
        Creates the transformed hermitian generator of UCC type unitaries:
              M(a^\dagger_{a_0} a_{i_0} a^\dagger{a_1}a_{i_1} ... - h.c.)
              where the qubit map M depends is self.transformation

        Parameters
        ----------
        indices : typing.Iterable[typing.Tuple[int, int]] :
            List of tuples [(a_0, i_0), (a_1, i_1), ... ] - recommended format, in spin-orbital notation (alpha odd numbers, beta even numbers)
            can also be given as one big list: [a_0, i_0, a_1, i_1 ...]
        form : str : (Default value None):
            Manipulate the generator to involution or projector
            set form='involution' or 'projector'
            the default is no manipulation which gives the standard fermionic excitation operator back
        remove_constant_term: bool: (Default value True):
            by default the constant term in the qubit operator is removed since it has no effect on the unitary it generates
            if the unitary is controlled this might not be true!
        Returns
        -------
        type
            1j*Transformed qubit excitation operator, depends on self.transformation
        """
        pass
    #Latter add Ansatzs and Algorithm
