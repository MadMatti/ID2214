import pandas as pd
import numpy as np
from IPython.display import display
# import all rdkit needed libraries
import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import data

data.feature_extraction(data.get_mol(data.load_data()))





