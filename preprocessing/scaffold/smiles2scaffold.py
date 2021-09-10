import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


# # test generate scaffold
# s = 'Cc1cc(Oc2nccc(CCC)c2)ccc1'
# scaffold = generate_scaffold(s)
# print(scaffold)
# assert scaffold == 'c1ccc(Oc2ccccn2)cc1'


def scaffold_split(smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds. Deterministic split
    :param smiles_list: list of smiles
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :return: list of train, valid, test indices corresponding to the
    scaffold split
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)

        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    scaffold_group = np.zeros(len(smiles_list), dtype=np.int64)
    for i, index in enumerate(all_scaffold_sets):
        scaffold_group[index] = i

    np.save('scaffold_group', scaffold_group)

    return train_idx, valid_idx, test_idx


if __name__ == '__main__':
    df = pd.read_csv('mol.csv.gz')
    smiles_list = df['smiles'].tolist()
    train_idx, valid_idx, test_idx = scaffold_split(smiles_list)
