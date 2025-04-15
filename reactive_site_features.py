import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import numpy as np
from rdkit.Chem import Descriptors


def Global_FromInchi(inchi: str):
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return {name: np.nan for name, _ in Descriptors._descList}

    res = {}
    for key, func in Descriptors._descList:
        res[key] = func(mol)

    return res


def Global_FromSmiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: np.nan for name, _ in Descriptors._descList}

    res = {}
    for key, func in Descriptors._descList:
        res[key] = func(mol)

    return res


def Local_FromSmiles(smiles: str, atom_idx_begin: int, atom_idx_end: int, radius: int):
    # print(smiles)
    # print(index)
    if np.isnan(atom_idx_begin) or np.isnan(atom_idx_end):
        return {name: np.nan for name, _ in _localDescList}
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return {name: np.nan for name, _ in _localDescList}
    mol = AllChem.RemoveHs(mol)

    lst_atoms, lst_bonds = _GetAtomEnvironment(mol, atom_idx_begin, atom_idx_end, radius=radius)
    res = {}
    for key, func in _localDescList:
        res[key] = func(mol, lst_atoms=lst_atoms, lst_bonds=lst_bonds)

    return res


def Local_FromInchi(inchi: str, index: int, radius: int):
    if np.isnan(index):
        return {name: np.nan for name, _ in _localDescList}
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return {name: np.nan for name, _ in _localDescList}
    mol = AllChem.RemoveHs(mol)

    lst_atoms, lst_bonds = _GetAtomEnvironment(mol, atom_idx_begin=index, atom_idx_end=index, radius=radius)
    res = {}
    for key, func in _localDescList:
        res[key] = func(mol, lst_atoms=lst_atoms, lst_bonds=lst_bonds)

    return res


def _GetAtomEnvironment(mol, atom_idx_begin, atom_idx_end, radius=4):
    bla = 0
    res1 = Chem.rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius - bla, int(atom_idx_begin))
    res2 = Chem.rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius - bla, int(atom_idx_end))

    lst_atoms = []
    lst_bonds = []
    for bond_idx in res1:
        lst_bonds.append(bond_idx)
        lst_atoms.append(mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx())
        lst_atoms.append(mol.GetBondWithIdx(bond_idx).GetEndAtomIdx())
    for bond_idx in res2:
        lst_bonds.append(bond_idx)
        lst_atoms.append(mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx())
        lst_atoms.append(mol.GetBondWithIdx(bond_idx).GetEndAtomIdx())
    lst_atoms = list(set(lst_atoms))
    lst_bonds = list(set(lst_bonds))
    return lst_atoms, lst_bonds


def _Calc_NumAromaticAtoms(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a.GetIsAromatic() for a in lst]
    return np.sum(lst, dtype=float)


def _Calc_NumNonAromaticAtoms(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [not a.GetIsAromatic() for a in lst]
    return np.sum(lst, dtype=np.float64)


def _Calc_NumAromaticBonds(mol, lst_atoms, lst_bonds):
    lst = [mol.GetBondWithIdx(a) for a in lst_bonds]
    lst = [a.GetIsAromatic() for a in lst]
    return np.sum(lst, dtype=np.float64)


def _Calc_NumNonAromaticBonds(mol, lst_atoms, lst_bonds):
    lst = [mol.GetBondWithIdx(a) for a in lst_bonds]
    lst = [not a.GetIsAromatic() for a in lst]
    return np.sum(lst, dtype=np.float64)


def _Calc_NumHeteroAtoms(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a for a in lst if a.GetAtomicNum() not in [1, 6]]
    return np.float64(len(lst))


def _Calc_NumHomoatoms(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a for a in lst if a.GetAtomicNum() in [1, 6]]
    return np.float64(len(lst))


def _Calc_NumNitrogens(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a for a in lst if a.GetAtomicNum() == 7]
    return np.float64(len(lst))


def _Calc_NumCouplingNitrogens(mol, lst_atoms, lst_bonds):
    # Number of nitrogens that have at least one H
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a for a in lst if a.GetAtomicNum() == 7]
    lst = [a for a in lst if (a.GetNumExplicitHs() + a.GetNumImplicitHs()) > 0]
    return np.float64(len(lst))


def _Calc_NumAtoms(mol, lst_atoms, lst_bonds):
    return np.float64(len(lst_atoms))


def _Calc_NumBonds(mol, lst_atoms, lst_bonds):
    return np.float64(len(lst_bonds))


def _Calc_FragmentWeight(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a.GetMass() for a in lst]
    return np.sum(lst, dtype=np.float64)


def _Calc_MeanAtomDegree(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a.GetDegree() for a in lst]
    return np.mean(lst, dtype=np.float64)


def _Calc_FractionInRings(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a.IsInRing() for a in lst]
    try:
        return np.sum(lst, dtype=np.float64) / len(lst)
    except:
        pass


def _Calc_NumConjugatedBonds(mol, lst_atoms, lst_bonds):
    lst = [mol.GetBondWithIdx(a) for a in lst_bonds]
    lst = [a.GetIsConjugated() for a in lst]
    return np.sum(lst, dtype=np.float64)


def _Calc_NumNonConjugatedBonds(mol, lst_atoms, lst_bonds):
    lst = [mol.GetBondWithIdx(a) for a in lst_bonds]
    lst = [not a.GetIsConjugated() for a in lst]
    return np.sum(lst, dtype=np.float64)


def _Calc_Num_SP2(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a.GetHybridization() == Chem.rdchem.HybridizationType.SP2 for a in lst]
    return np.sum(lst, dtype=np.float64)


def _Calc_Num_SP3(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a.GetHybridization() == Chem.rdchem.HybridizationType.SP3 for a in lst]
    return np.sum(lst, dtype=np.float64)


def _Calc_Num_SPother(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a.GetHybridization() not in [Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3] for a in
           lst]
    return np.sum(lst, dtype=np.float64)


def _Calc_Frac_SP2(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a.GetHybridization() == Chem.rdchem.HybridizationType.SP2 for a in lst]
    return np.sum(lst, dtype=np.float64) / len(lst)


def _Calc_Frac_SP3(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a.GetHybridization() == Chem.rdchem.HybridizationType.SP3 for a in lst]
    return np.sum(lst, dtype=np.float64) / len(lst)


def _Calc_Frac_SPother(mol, lst_atoms, lst_bonds):
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a.GetHybridization() not in [Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3] for a in
           lst]
    return np.sum(lst, dtype=np.float64) / len(lst)


def _Calc_HeteroHs(mol, lst_atoms, lst_bonds):
    # Number of Heteroatoms with Hs
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [(a.GetNumExplicitHs() + a.GetNumImplicitHs()) for a in lst if a.GetAtomicNum() not in [1, 6]]
    return np.sum(lst, dtype=np.float64)


def _Calc_NumOxygens(mol, lst_atoms, lst_bonds):
    # Number of Oxygens
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a for a in lst if a.GetAtomicNum() == 8]
    return np.float64(len(lst))


def _Calc_NumSulfur(mol, lst_atoms, lst_bonds):
    # Number of Sulfurs
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a for a in lst if a.GetAtomicNum() == 16]
    return np.float64(len(lst))


def _Calc_NumSulfurNoOxy(mol, lst_atoms, lst_bonds):
    # Number of Sulfurs without an Oxygen as neighbor
    lst = [mol.GetAtomWithIdx(a) for a in lst_atoms]
    lst = [a for a in lst if a.GetAtomicNum() == 16]
    lst = [a for a in lst if all([b.GetAtomicNum() != 8 for b in a.GetNeighbors()])]
    return np.float64(len(lst))


_localDescList = [
    ('NumAromaticAtoms', _Calc_NumAromaticAtoms),
    ('NumNonAromaticAtoms', _Calc_NumNonAromaticAtoms),
    ('NumAromaticBonds', _Calc_NumAromaticBonds),
    ('NumNonAromaticBonds', _Calc_NumNonAromaticBonds),
    ('NumHeteroAtoms', _Calc_NumHeteroAtoms),
    ('NumHomoatoms', _Calc_NumHomoatoms),
    ('NumNitrogens', _Calc_NumNitrogens),
    ('NumCouplingNitrogens', _Calc_NumCouplingNitrogens),
    ('NumAtoms', _Calc_NumAtoms),
    ('NumBonds', _Calc_NumBonds),
    ('MeanAtomDegree', _Calc_MeanAtomDegree),
    ('FractionInRings', _Calc_FractionInRings),
    ('NumConjugatedBonds', _Calc_NumConjugatedBonds),
    ('NumNonConjugatedBonds', _Calc_NumNonConjugatedBonds),
    ('NumSP2', _Calc_Num_SP2),
    ('NumSP3', _Calc_Num_SP3),
    ('NumSPother', _Calc_Num_SPother),
    ('FracSP2', _Calc_Frac_SP2),
    ('FracSP3', _Calc_Frac_SP3),
    ('FracSPother', _Calc_Frac_SPother),
    ('HeteroHs', _Calc_HeteroHs),
    ('NumOxygens', _Calc_NumOxygens),
    ('NumSulfur', _Calc_NumSulfur),
    ('NumSulfurNoOxy', _Calc_NumSulfurNoOxy),
]
if __name__ == '__main__':
    import pandas as pd

    results = []
    # df=pd.read_csv('dataset/raw/USPTO/data/uspto_full_new.csv')
    smi = 'C[C@@H]1COCCN1c1cc(C2([S@](C)(=N)=O)CC2)nc(-c2cc[nH]c3nccc2-3)n1'
    # try:
    #     for i,smi in enumerate(df['products']):
    #         results.append(list(Local_FromSmiles(smiles=smi, index=5, radius=3).values()))
    # except:
    #     print('exceptions')
    results = list(Local_FromSmiles(smiles=smi, atom_idx_begin=5, atom_idx_end=6, radius=2).values())
    print(results)
