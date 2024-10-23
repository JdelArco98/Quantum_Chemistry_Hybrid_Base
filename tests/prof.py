import tequila as tq
import HybridBase as hb
import profile


def a():
    print("HOLa")
profile.run('hb.Molecule(geometry="H 0. 0. 0. \n Li 0. 0. 1.5",basis_set="sto-3g",select={2:"F",4:"F"})')
