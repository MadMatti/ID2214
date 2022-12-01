import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from scipy import stats
# import all rdkit needed libraries
import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

filename = 'Assignment_4/Resources/training_smiles.csv'

def load_data():
    df = pd.read_csv(filename, index_col=0)
    return df

def get_mol(df):
    df['mol'] = df['SMILES'].apply(rdkit.Chem.MolFromSmiles)
    df.drop('SMILES', axis=1, inplace=True)
    return df

def feature_extraction(df):
    df['num_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
    df['num_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
    df['exact_mol_wt'] = df['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    df['AI_COO'] = df['mol'].apply(lambda x: Descriptors.fr_Al_COO(x))
    df['morgan_fp'] = df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=124))

    return df

def data_cleaning(df):
    df.drop('mol', axis=1, inplace=True)
    print(df)
    print(df.info())
    print(df.shape)
    print(df.drop_duplicates().shape) # no duplicates
    print(df.isnull().sum()) # no missing values

    with pd.option_context('display.max_columns', 40):
        print(df.describe(include='all'))

    # outliers detections
    ZSCORE_THREASHOLD = 4
    zscore = np.abs(stats.zscore(df.select_dtypes(include=["float", "int64"])))

    is_inlier = ~ (zscore > ZSCORE_THREASHOLD).any(axis=1)
    df = df[is_inlier]
    print(df.info()) # 4.2% of the data is removed -> accetable number



    





if __name__ == '__main__':
    data_cleaning(feature_extraction(get_mol(load_data())))
    
