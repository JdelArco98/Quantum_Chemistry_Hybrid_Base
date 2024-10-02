## Installation
Will work on OSX and Linux (no PySCF on Windows)

Install this program with all the dependencies like this:

```
conda create -n myenv python=3.9
conda activate myenv

cd Hybrid_Base
pip install -e .
```

## Hybrid Molecule
Create your Hybrid Molecule. \
Example :

```
import HybridBase as hm # first time might take some seconds
import tequila as tq

molecule  = hm.Molecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5",basis_set="sto-3g",select="BBFBF")
print(molecule.select)
```

it can be also initialized as

```
import HybridBase as hm
import tequila as tq
molecule  = hm.Molecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5",basis_set="sto-3g",select={2:"F",4:"F"})
print(molecule.select)
```

## Construct your circuit circuit
The SPA circuit (and all the automatically built circuits) are already adapted to your encoding

```
Uspa = molecule.make_ansatz("SPA",edges=[(0,1)])
```
however, one can also  build its own circuits:

```
U = tq.QCircuit() # see more on https://github.com/tequilahub/tequila-tutorials/blob/main/BasicUsage.ipynb
U += molecule.prepare_reference() # Prepares the reference HF state if any other provided
U += molecule.UC(0,2,angle=(0,2,"a")) #Paired 2e excitation from MO 0 to MO 2
U += molecule.UR(2,4,angle=(2,4,"UR")) # Two One-electron excitation: MO 2_up->4_up + 2_down->4_down TAKE CARE ENCODING
U += molecule.make_excitation_gate(indices=[(0,4),(1,8)],angle=tq.Variable('a')) #Generic excitation

```
## Minimize the Energy of the Circuit Expectation Value

```
H = molecule.make_hamiltonian() # The molecular Hamiltonian for a given Encoding is automatically built. For custom Hamiltonians please check tutorial above
exp = tq.ExpectationValue(H=H,U=U) #Create the Expectation Value Object
mini = tq.minimize(objective=exp,silent=False,initial_values={}) #Then you minimize the energy. You can provide initial variables
print('Minimized Angles:\n',mini.angles)
print('Minimized Energy: ', mini.energy)
```

## Optimize your Orbitals
Molecular orbitals can be optimized taking advantage of this Hybrid Encoding

```
result = molecule.optimize_orbitals(molecule=molecule,circuit=Uspa,initial_guess='random') #Since random guess, may take some time
omol = result.molecule
print("Opt SPA Energy = ",result.energy)
print("Select: ",omol.select)
```


Find this example in the test file

\
