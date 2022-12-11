import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sn
from tabulate import tabulate
# import all rdkit needed libraries
import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Fragments
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors

from sklearn.feature_selection import SelectKBest, chi2, RFECV, RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold


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

    return df

def feature_extraction_complete(df):
    df['num_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
    df['num_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
    df['exact_mol_wt'] = df['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    df['AI_COO'] = df['mol'].apply(lambda x: Descriptors.fr_Al_COO(x))
    df['morgan_fp'] = df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=124))
    # convert morgan_fp to string
    df['morgan_fp'] = df['morgan_fp'].apply(lambda x: x.ToBitString())

    df['num_hetero_atoms'] = df['mol'].apply(lambda x: Lipinski.NumHeteroatoms(x))
    df['num_bonds'] = df['mol'].apply(lambda x: x.GetNumBonds())
    df['num_rotatable_bonds'] = df['mol'].apply(lambda x: rdMolDescriptors.CalcNumRotatableBonds(x))
    df['num_aliphatic_rings'] = df['mol'].apply(lambda x: rdMolDescriptors.CalcNumAliphaticRings(x))
    df['num_aromatic_rings'] = df['mol'].apply(lambda x: rdMolDescriptors.CalcNumAromaticRings(x))
    df['num_saturated_rings'] = df['mol'].apply(lambda x: rdMolDescriptors.CalcNumSaturatedRings(x))
    df['Num_radical_electrons'] = df['mol'].apply(lambda x: Descriptors.NumRadicalElectrons(x))
    df['Num_valence_electrons'] = df['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    df['Ar_N'] = df['mol'].apply(lambda x: Fragments.fr_Ar_N(x))
    df['COO'] = df['mol'].apply(lambda x: Fragments.fr_COO(x))
    df['amide'] = df['mol'].apply(lambda x: Fragments.fr_amide(x))
    df['benzene'] = df['mol'].apply(lambda x: Fragments.fr_benzene(x))
    df['ester'] = df['mol'].apply(lambda x: Fragments.fr_ester(x))
    df['nitro'] = df['mol'].apply(lambda x: Fragments.fr_nitro(x))
    df['nitro_arom'] = df['mol'].apply(lambda x: Fragments.fr_nitro_arom(x))
    df['NHOH_count'] = df['mol'].apply(lambda x: Lipinski.NHOHCount(x))
    df['NO_count'] = df['mol'].apply(lambda x: Lipinski.NOCount(x))
    df['num_H_acceptors'] = df['mol'].apply(lambda x: Lipinski.NumHAcceptors(x))
    df['num_H_donors'] = df['mol'].apply(lambda x: Lipinski.NumHDonors(x))
    df['num_saturated_rings'] = df['mol'].apply(lambda x: rdMolDescriptors.CalcNumSaturatedRings(x))
    # 30 features extracted

    df.drop('mol', axis=1, inplace=True)

    return df

def all_features_to_csv(df):
    df['num_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
    df['num_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
    df['exact_mol_wt'] = df['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    df['AI_COO'] = df['mol'].apply(lambda x: Descriptors.fr_Al_COO(x))
    df['morgan_fp'] = df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=124))
    # convert morgan_fp to string
    df['morgan_fp'] = df['morgan_fp'].apply(lambda x: x.ToBitString())

    df['num_hetero_atoms'] = df['mol'].apply(lambda x: Lipinski.NumHeteroatoms(x))
    df['num_bonds'] = df['mol'].apply(lambda x: x.GetNumBonds())
    df['num_rotatable_bonds'] = df['mol'].apply(lambda x: rdMolDescriptors.CalcNumRotatableBonds(x))
    df['num_aliphatic_rings'] = df['mol'].apply(lambda x: rdMolDescriptors.CalcNumAliphaticRings(x))
    df['num_aromatic_rings'] = df['mol'].apply(lambda x: rdMolDescriptors.CalcNumAromaticRings(x))
    df['num_saturated_rings'] = df['mol'].apply(lambda x: rdMolDescriptors.CalcNumSaturatedRings(x))
    df['Num_radical_electrons'] = df['mol'].apply(lambda x: Descriptors.NumRadicalElectrons(x))
    df['Num_valence_electrons'] = df['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    df['Ar_N'] = df['mol'].apply(lambda x: Fragments.fr_Ar_N(x))
    df['COO'] = df['mol'].apply(lambda x: Fragments.fr_COO(x))
    df['amide'] = df['mol'].apply(lambda x: Fragments.fr_amide(x))
    df['benzene'] = df['mol'].apply(lambda x: Fragments.fr_benzene(x))
    df['ester'] = df['mol'].apply(lambda x: Fragments.fr_ester(x))
    df['nitro'] = df['mol'].apply(lambda x: Fragments.fr_nitro(x))
    df['nitro_arom'] = df['mol'].apply(lambda x: Fragments.fr_nitro_arom(x))
    df['NHOH_count'] = df['mol'].apply(lambda x: Lipinski.NHOHCount(x))
    df['NO_count'] = df['mol'].apply(lambda x: Lipinski.NOCount(x))
    df['num_H_acceptors'] = df['mol'].apply(lambda x: Lipinski.NumHAcceptors(x))
    df['num_H_donors'] = df['mol'].apply(lambda x: Lipinski.NumHDonors(x))
    df['num_saturated_rings'] = df['mol'].apply(lambda x: rdMolDescriptors.CalcNumSaturatedRings(x))
    # 30 features extracted

    df.drop('mol', axis=1, inplace=True)

    df.to_csv('all_features.csv')

def univariate_selection(df):
    df1 = df.copy()
    y = df1['ACTIVE']
    X = df1.drop('ACTIVE', axis=1)

    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    
    #print(tabulate(featureScores.nlargest(10,'Score'), tablefmt='psql')) # print 10 best features
    return featureScores.nlargest(10, 'Score')

def feature_importance(df):
    df1 = df.copy()
    y = df1['ACTIVE']
    X = df1.drop('ACTIVE', axis=1)

    model = Pipeline(steps=[('scaler', StandardScaler()),
                            ('extreme', ExtraTreesClassifier())])

    #model = ExtraTreesClassifier()
    model.fit(X,y)
    #print(model[1][1].feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model[1][1].feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    #plt.show()
    return feat_importances.nlargest(10)

def correlation_matrix(df):
    df1 = df.copy()
    y = df1['ACTIVE']
    X = df1.drop('ACTIVE', axis=1)

    #get correlations of each features in dataset
    corrmat = df1.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sn.heatmap(df1[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    #plt.show()

def final_selection(all_features):

    univariate = univariate_selection(all_features)
    importance = feature_importance(all_features)

    uni_list = univariate['Specs'].tolist()
    imp_list = importance.index.tolist()
    
    # List containing element from both selection techniques
    final_list = list(set(uni_list + imp_list))
    final_list.append('ACTIVE') # add the target column

    # now check the correlation matrix of this features
    final_features = all_features[final_list] # df with only wanted features

    #split fingerprint per bit 
    df_fingerprints = final_features['morgan_fp'].str.split('', expand=True)
    final_features = pd.concat([final_features,df_fingerprints],axis=1)
    final_features.columns = final_features.columns.astype(str)

    # now we can drop featuers that are highly correlated
    final_features.drop(['Num_valence_electrons', 'num_heavy_atoms', 'num_bonds', 'num_hetero_atoms', 'NO_count', 'morgan_fp'], axis=1, inplace=True)

    corrmat = final_features.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sn.heatmap(final_features[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    #plt.show()

    return final_features

def feature_selection(all_features):
    # feature selection using Recursive Feature Elimination

    X = all_features.drop('ACTIVE', axis=1)
    y = all_features['ACTIVE']

    # Selecting the most important features according to ExtraTreesClassifier
    model = Pipeline(steps=[('scaler', StandardScaler()),
                            ('extreme', ExtraTreesClassifier())])
    model.fit(X, y)
    rfecv_selector = Pipeline(steps=[('scaler', StandardScaler()),
                    ('rfecv', RFECV(estimator=model[1][1], cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1, step=1))])
    #rfecv_selector = RFECV(estimator=model[1][1], cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1, min_features_to_select=10, step=1)
    rfecv_selector.fit(X, y)
    features_arr = rfecv_selector.get_feature_names_out().tolist()
    features_arr.append('ACTIVE')
    print(features_arr)
    return all_features[features_arr]

    
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
    #data_analysis(data_cleaning(feature_extraction(get_mol(load_data()))))
    # all_features_to_csv(get_mol(load_data()))
    features = pd.read_csv('Assignment_4/resources/all_features.csv', index_col=0)
    # univariate_selection(data_cleaning(features))
    # feature_importance(data_cleaning(features))
    # correlation_matrix(data_cleaning(features))
    #final_selection(data_cleaning(features))
    feature_selection(data_cleaning(features))
    
