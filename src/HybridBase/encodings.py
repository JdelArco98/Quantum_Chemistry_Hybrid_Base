"""
Collections of Fermion-to-Qubit encodings known to tequila
Adapted to the encoding
"""
import typing
from tequila import TequilaException
from tequila.circuit.circuit import QCircuit
from tequila.circuit.gates import X, CNOT
from tequila.hamiltonian.paulis import Sp,Sm,Z
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian
import openfermion
from tequila.quantumchemistry.encodings import EncodingBase as EB
from copy import deepcopy
def known_encodings():
    # convenience for testing and I/O
    encodings = {
        "JordanWigner": JordanWigner,
        # "BravyiKitaev": BravyiKitaev,
        # "BravyiKitaevFast": BravyiKitaevFast,
        # "BravyiKitaevTree": BravyiKitaevTree,
        # "TaperedBravyiKitaev": TaperedBravyKitaev
    }
    # aliases
    encodings = {**encodings,
                 "ReorderedJordanWigner": lambda **kwargs: JordanWigner(up_then_down=True, **kwargs),
                 # "ReorderedBravyiKitaev": lambda **kwargs: BravyiKitaev(up_then_down=True, **kwargs),
                 # "ReorderedBravyiKitaevTree": lambda **kwargs: BravyiKitaevTree(up_then_down=True, **kwargs),
                 }
    return {k.replace("_", "").replace("-", "").upper(): v for k, v in encodings.items()}

class JordanWigner(EB):
    _ucc_support = True
    def __init__(self, n_electrons, n_orbitals,select:dict, up_then_down=False ,*args, **kwargs):
        self.select = select
        if ("condense" in kwargs):
            self.condense = kwargs["condense"]
            kwargs.pop("condense")
        else:
            self.condense = True
        if ("two_qubit" in kwargs):
            self.two_qubit = kwargs["two_qubit"]
            kwargs.pop("two_qubit")
        else:
            self.two_qubit = False
        if self.two_qubit : self.condense = False
        super().__init__(n_electrons, n_orbitals, up_then_down)
        self.FER_SO,self.pos=self.select_to_list()

    pass

    def select_to_list(self):
        """
        Internal function
        Read the select string to make the proper Fer and Bos lists
        :return : list of MOs for the Bos, MOs and SOs for the Fer space
        """
        hcb = 0
        FER_SO = []
        sel = self.select
        pos = {}
        up = self.up_then_down
        two = self.two_qubit
        for i in sel:
            if (sel[i]=="B"):
                pos[2*i] = i+(i-hcb)*(not up)
                if two:
                    pos[2*i+1]  =  i+self.n_orbitals*up+(not up)*(i+1)
                    FER_SO.append(pos[2*i])
                    FER_SO.append(pos[2*i+1])
                elif self.condense:
                    hcb += 1
            else:
                pos[2*i]  =  i+(i-hcb)*(not up)
                pos[2*i+1]  =  i-hcb+up*self.n_orbitals+(not up)*(i+1)
                FER_SO.append(pos[2*i])
                FER_SO.append(pos[2*i+1])
        # FER_SO.sort()
        return FER_SO,pos

    def __call__(self, fermion_operator: openfermion.FermionOperator, *args, **kwargs) -> QubitHamiltonian:
        """
        :param fermion_operator:
            an openfermion FermionOperator
        :return:
            The openfermion QubitOperator of this class ecoding
        """
        fop = self.do_transform(fermion_operator=fermion_operator, *args, **kwargs)
        return self.post_processing(fop)

    def post_processing(self, op, *args, **kwargs):
        return op
    def do_transform(self, fermion_operator: typing.Union[openfermion.FermionOperator,list,tuple], *args, **kwargs) -> QubitHamiltonian:
        """
            Transform the fermi_operator list in its respective gates similar to openfermion function but taking
            into account the mixed condification
            return: qop: the sum of the different operator product multiplied by a prefactor
        """
        def cre(i: int) -> QubitHamiltonian():
            """
            Internal function
            Creates a creation operator on the JW codification acting on the i-th SO
            :param i: number of the SO in full-Fermionic-not-upthendown over the creation operators acts
            :return :creation operator acting on the corresponding qubits
            """
            d = self.pos
            a = Sm(d[i])
            if self.two_qubit or (self.select[i//2]=="F"):
                for n in self.FER_SO[:self.FER_SO.index(d[i])]:
                    a *= Z(n)
            return a
        def anni(i: int) -> QubitHamiltonian():
            """
            Internal function
            Creates a annihilation operator on the JW codification acting on the i-th SO
            :param i: number of the SO in full-Fermionic-not-upthendown over the creation operators acts
            :return :annihilation operator acting on the corresponding qubits
            """
            d = self.pos
            a = Sp(d[i])
            if self.two_qubit or (self.select[i//2] == "F"):
                for n in self.FER_SO[:self.FER_SO.index(d[i])]:
                    a *= Z(n)
            return a

        qop = QubitHamiltonian()
        if type(fermion_operator) is openfermion.FermionOperator:
            lista = list(fermion_operator.terms.items())
        else:
            lista = [*fermion_operator]
        for pair in lista:
            index = pair[0]
            temp = pair[1]
            for term in index:
                if (not self.two_qubit) and (self.select[term[0]//2]=="B" and (term[0]%2)):
                    pass
                else:
                    if (term[1]):
                        temp *= cre(term[0])
                    else:
                        temp *= anni(term[0])
            qop += temp
        return qop

    def hcb_to_me(self, *args, **kwargs):
        '''
        all: CNOT for all orbs, including FER
        bos: CNOT only for the Bos orbitals
        '''
        if "all" in kwargs:
            all = kwargs["all"]
            kwargs.pop("all")
        else: all = False
        if "bos" in kwargs:
            bos = kwargs["bos"]
            kwargs.pop("bos")
        else: bos = False
        if bos and self.condense:
            raise TequilaException("hcb_to_me called with bos with condensed encoding")
        if bos and all:
            raise TequilaException("hcb_to_me called with all and bos both True")
        if all and self.condense:
            raise TequilaException("hcb_to me asked for all orbitals in condensed encoding")
        U = QCircuit()
        pos = deepcopy(self.pos)
        if not self.two_qubit and bos:
            for i in range(self.n_orbitals):
                if self.select[i]=='B':
                    pos[2*i+1] = i +self.n_orbitals*self.up_then_down+(not self.up_then_down)*(i+1)
        for i in range(self.n_orbitals):
            if bos and self.select[i]=='B':
                U += X(target=pos[2 * i + 1], control=pos[2 * i])
            if not bos and (all or self.two_qubit or self.select[i]=='F'):
                U += X(target=pos[2*i+1], control=pos[2*i])
        return U

    def me_to_jw(self) -> QCircuit:
        return QCircuit()

    def jw_to_me(self) -> QCircuit:
        return QCircuit()

    def map_state(self, state: list, *args, **kwargs):
        state = state + [0] * (self.n_orbitals - len(state)-[*self.select.values()].count("B")*self.condense)
        result = [0] * len(state)
        for i in range(len(state)//2):
            if self.select[i] == 'B':
                if self.two_qubit: pass
                else: state[2*i+1] = 0
        if self.up_then_down:
            up = [state[2 * i] for i in range(self.n_orbitals)]
            down = [state[2 * i + 1] for i in range(self.n_orbitals)]
            if self.condense:
                for i in self.select:
                    if self.select[i]=='B': down.pop(i)
            return up + down
        elif self.condense:
            for i in [*self.select.keys()][::-1]:
                if self.select[i] == 'B': state.pop(2*i+1)
            return state
        else:
            return state

    def up(self,i):
        return self.pos[2*i]

    def down(self, i):
        return self.pos[2*i+1]