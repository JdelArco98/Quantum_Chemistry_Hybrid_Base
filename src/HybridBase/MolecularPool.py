from tequila.apps.adapt.adapt import MolecularPool as MainMP


class MolecularPool(MainMP):
    def make_indices_singles(self, generalized=False):
        indices = []
        for p in range(self.molecule.n_electrons // 2):
            for q in range(self.molecule.n_electrons // 2, self.molecule.n_orbitals):
                if(hasattr(self.molecule,"select")):
                    if (self.molecule.select[q] == "F" and self.molecule.select == "F"):
                        indices.append([(2 * p, 2 * q)])
                        indices.append([(2 * p + 1, 2 * q + 1)])
                else:
                    indices.append([(2*p, 2*q)])
                    indices.append([(2*p+1, 2*q+1)])
        if not generalized:
            return indices

        for p in range(self.molecule.n_orbitals):
            for q in range(p + 1, self.molecule.n_orbitals):
                if(hasattr(self.molecule,"select")):
                    if (self.molecule.select[q] == "F" and self.molecule.select == "F"):
                        if [(2 * p, 2 * q)] in indices:
                            continue
                        indices.append([(2 * p, 2 * q)])
                        indices.append([(2 * p + 1, 2 * q + 1)])
                else:
                    if [(2*p, 2*q)] in indices:
                        continue
                    indices.append([(2*p, 2*q)])
                    indices.append([(2*p+1, 2*q+1)])
        return self.sort_and_filter_unique_indices(indices)
    def make_indices_doubles(self, generalized=False, paired=True):
        indices = []
        for p in range(self.molecule.n_electrons//2):
            for q in range(self.molecule.n_electrons//2, self.molecule.n_orbitals):
                indices.append([(2*p, 2*q),(2*p+1, 2*q+1)])

        if not generalized:
            return indices

        for p in range(self.molecule.n_orbitals):
            for q in range(p+1, self.molecule.n_orbitals):
                idx = [(2 * p, 2 * q), (2 * p + 1, 2 * q + 1)]
                if idx in indices:
                    continue
                indices.append(idx)
        if not paired:
            indices += self.make_indices_doubles_all(generalized=generalized)

        return self.sort_and_filter_unique_indices(indices)
    def make_indices_doubles_all(self, generalized=False):
        from itertools import combinations
        singles = self.make_indices_singles(generalized=generalized)
        unwrapped = [x[0] for x in singles]
        # now make all combinations of singles
        if(hasattr(self.molecule,"select")):
            indices = [x for x in combinations(unwrapped, 2) if self.molecule.verify_excitation(x)]
        else:
            indices = [x for x in combinations(unwrapped, 2)]
        return self.sort_and_filter_unique_indices(indices)
