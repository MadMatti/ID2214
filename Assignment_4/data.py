import pandas as pd
import numpy as np
from IPython.display import display
# import all rdkit needed libraries
import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

def load_data():
    df = pd.read_csv('Resources/training_smiles.csv')
    return df

def get_mol(df):
    df['mol'] = df['SMILES'].apply(rdkit.Chem.MolFromSmiles)
    return df

def feature_extraction(df):
    df['num_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
    df['num_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
    df['exact_mol_wt'] = df['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    df['AI_COO'] = df['mol'].apply(lambda x: Descriptors.fr_Al_COO(x))
    df['morgan_fp'] = df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=124))

    return df

if __name__ == '__main__':
    df = feature_extraction(get_mol(load_data()))

