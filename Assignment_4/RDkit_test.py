from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as d
import rdkit.Chem.Fragments as f
import rdkit.Chem.Lipinski as l
from rdkit.Chem import AllChem
import numpy as np

m = Chem.MolFromSmiles('Cc1ccccc1')
print('Number of Atoms: ', m.GetNumAtoms())

print('Calculate exact mol feature', d.CalcExactMolWt(m))

print('Fragments: ',f.fr_Al_COO(m))

print('Lipinski: ',l.HeavyAtomCount(m))

fp = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=124)
print('Precence or absence of substructures: ', np.array(fp))