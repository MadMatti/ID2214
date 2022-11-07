import pandas as pd
import numpy as np
from IPython.display import display


def create_bins(df, nobins=10, bintype="equal-width"):
    df1 = df.copy()
    binning = {}

    for col in df1.columns:
        if col not in ['CLASS', 'ID']:
            if df1[col].dtype == "int64" or df1[col].dtype == "float64":
                if bintype == "equal-width":
                    df1[col], binning[col] = pd.cut(df1[col],nobins,labels=False, retbins=True)
                elif bintype == "equal-size":
                    df1[col], binning[col] = pd.qcut(df[col],nobins, labels=False, retbins=True, duplicates='drop')

                binning[col][0] = -np.inf
                binning[col][-1] = np.inf
                df1[col] = df1[col].astype('category')
                df1[col] = df1[col].cat.set_categories([str(i) for i in df1[col].cat.categories], rename = True)

    return df1, binning

def apply_bins(df, binning):
    df1 = df.copy()

    for col in df1:
        if col in binning.keys():
            df1[col] = pd.cut(df1[col], binning[col], labels=False)
            df1[col] = df1[col].astype('category')
            df1[col] = df1[col].cat.set_categories([str(i) for i in df1[col].cat.categories], rename = True)

    return df1











def test():
    glass_train_df = pd.read_csv("Assignment_1/glass_train.csv")

    glass_test_df = pd.read_csv("Assignment_1/glass_test.csv")

    glass_train_disc, binning = create_bins(glass_train_df,nobins=10,bintype="equal-size")
    print("binning:")
    for f in binning:
        print("{}:{}".format(f,binning[f]))

    print()    
    glass_test_disc = apply_bins(glass_test_df,binning)
    display("glass_test_disc",glass_test_disc)

if __name__ == "__main__":
    test()

    

