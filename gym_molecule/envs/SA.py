import math
import pickle

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import os
import os.path as op

_fscores = None


def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(os.getcwd(), name)
        # name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    # macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


lgs = []
mols = []
smis = [
    'O=C1NN=C(CC2=CC=C(F)C(C(N3CCN(C4=CC(NC5=CC(C6=CC=C(F)C=C6OC)=C(F)C=N5)=NC=C4)CC3)=O)=C2)C7=CC=CC=C71',
    'O=C(NC(N=C1)=CC(C(C=CC(F)=C2)=C2OC)=C1F)N3CCN(C(C4=CC(CC5=NNC(C6=C5C=CC=C6)=O)=CC=C4F)=O)CC3',
    'O=C(NN=C1CC2=C(F)C(C(N3CCN(C4=CN=C(C=C4)NC(N=C5)=CC(C(C(OC)=C6)=CC=C6F)=C5F)CC3)=O)=CC=C2)C7=C1C=CC=C7',
    'FC(C=C1OC)=CC=C1C2=C(F)C=NC(NC3=NC=CC(CN4CCN(C(C5=C(F)C=CC(CC6=NNC(C7=C6C=CC=C7)=O)=C5)=O)CC4)=C3)=C2',
    'O=C(NC(N=C1)=CC(C(C=CC(F)=C2)=C2OC)=C1F)[C@@H]3CC[C@H](NC(C4=C(F)C=CC(CC(C5=C6C=CC=C5)=NNC6=O)=C4)=O)CC3',
    'O=C(NN=C1CC2=CC=C(F)C(C(N3CCN(C4=NC(NC5=CC(C6=CC=C(F)C=C6OC)=NC=N5)=NC=C4)CC3)=O)=C2)C7=C1C=CC=C7',
    'FC1=CC=C(C=C1C(N2CCN(C3=CN=C(C=C3)NC4=CC(C(C(OC)=C5)=CC=C5F)=NC=N4)CC2)=O)CC6=NNC(C7=C6C=CC=C7)=O',
    'COC1=CC(F)=CC=C1C2=NC=NC(NC3=NC=CC(N4CCN(C(C5=CC(CC6=NNC(C7=C6C=CC=C7)=O)=CC=C5F)=O)CC4)=C3)=C2',
    'FC(C=NC(NC1=NC=NC(N2CCN(C(C3=CC(CC4=NNC(C5=C4C=CC=C5)=O)=CC=C3F)=O)CC2)=C1)=C6)=C6C(C(OC)=C7)=CC=C7F',
    'FC1=CC=C(C2=NC(NC3=CC(N4CCN(C(C5=CC(CC6=NNC(C7=CC=CC=C67)=O)=CC=C5F)=O)CC4)=CC=N3)=NC=C2F)C(OC)=C1',
    'FC1=CC=C(C2=NC(NC3=CC=C(N4CCN(C(C5=CC(CC6=NNC(C7=CC=CC=C67)=O)=CC=C5F)=O)CC4)C=N3)=NC=C2F)C(OC)=C1',
    'O=C(NC(N=C1)=CC(C(C=CC(F)=C2)=C2OC)=C1Cl)N3CCN(C(C4=CC(CC5=NNC(C6=CC=CC=C56)=O)=CC=C4F)=O)CC3',
    'FC1=CC=C(C2=CC(NC3=CC(N4CCN(C(C5=CC(CC6=NNC(C7=CC=CC=C67)=O)=CC=C5F)=O)CC4)=CC=N3)=NC=C2Cl)C(OC)=C1',
    'FC1=CC=C(C2=NC(NC3=NC=CC(CN(CC4)CCN4C(C5=CC(CC(C6=CC=CC=C67)=NNC7=O)=CC=C5F)=O)=C3)=NC=C2F)C(OC)=C1',
    'O=C(NC(N=C1)=NC(C(C=CC(F)=C2)=C2OC)=C1F)N3CCN(C(C4=CC(CC5=NNC(C6=C5C=CC=C6)=O)=CC=C4F)=O)CC3'
]
readFragmentScores("fpscores")
for m in smis:
    ms = Chem.MolFromSmiles(m)
    mols.append(ms)
    s = calculateScore(ms)
    lgs.append('SAScore : {}'.format(str(round(s,2))))
from rdkit.Chem import Draw

img = Draw.MolsToGridImage(mols[:10], molsPerRow=5, subImgSize=(300, 300), legends=lgs[:10], returnPNG=False)
img.save('filename.tiff')
