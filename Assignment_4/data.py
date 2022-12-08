import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sn
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

    # convert morgan_fp to string
    df['morgan_fp'] = df['morgan_fp'].apply(lambda x: x.ToBitString())

    df.drop('mol', axis=1, inplace=True)
    df.drop('num_heavy_atoms', axis=1, inplace=True)

    return df

def feature_extraction_1(df1):
    #add additional features
    df1["n_Atoms"] = df1['mol'].map(lambda x: x.GetNumAtoms())
    df1["ExactMolWt"] = df1['mol'].map(lambda x: Descriptors.ExactMolWt(x))
    df1["Fragments"] = df1['mol'].map(lambda x: Descriptors.fr_Al_COO(x))
    df1["HeavyAtomCount"] = df1['mol'].map(lambda x: x.GetNumHeavyAtoms())
    df1["MorganFingerPrint"] = df1['mol'].map(lambda x: AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=124).ToBitString())
    
    #split fingerprint per bit 
    df1_fingerprints = df1['MorganFingerPrint'].str.split('', expand=True)
    df1 = pd.concat([df1,df1_fingerprints],axis=1)
    df1.columns = df1.columns.astype(str)
    df1 = df1.drop(columns=['0','125'])
    
    #drop HeavyAtomCount since , except for 3 row, the column is equal to n_Atoms. 
    df1 = df1.drop(columns=['HeavyAtomCount','MorganFingerPrint', 'mol'])
   
    #Set type of binary values to int
    for column in df1.columns[-125:]:
        df1[column] = df1[column].astype('int64')
    
    return df1

def data_cleaning(df):
    # move label to last position
    last_column = df.pop('ACTIVE')
    df['ACTIVE'] = last_column
    # print(df)
    # print(df.info())
    # print(df.shape)
    print(df.drop_duplicates().shape) # no duplicates
    print(df.isnull().sum()) # no missing values

    # with pd.option_context('display.max_columns', 40):
    #     print(df.describe(include='all'))

    # outliers detections
    # ZSCORE_THREASHOLD = 4
    # zscore = np.abs(stats.zscore(df.select_dtypes(include=["float", "int64"])))

    # is_inlier = ~ (zscore > ZSCORE_THREASHOLD).any(axis=1)
    # df = df[is_inlier]
    # print(df.info()) # 4.2% of the data is removed -> accetable number

    return df

def data_analysis(df):
    # check correlation among features
    df_encoded = df.copy()
    #df_encoded.drop('INDEX', axis=1, inplace=True)
    corr_matrix = df_encoded.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.show()
    # data are highly correlated, we may want to drop number of heavy atoms

    # check distribution of the label
    numY, numN = df.ACTIVE.value_counts()
    print(numY, numN)
    df.ACTIVE.value_counts().plot(kind='pie',autopct='%1.0f%%', colors=['royalblue','red'])
    plt.title("Label Distribution")
    plt.xlabel("ACTIVE = 1730")
    plt.ylabel("INACTIVE = 154528")
    plt.show()
    

    





if __name__ == '__main__':
    data_analysis(data_cleaning(feature_extraction(get_mol(load_data()))))
    
