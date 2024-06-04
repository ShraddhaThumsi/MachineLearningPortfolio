import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import utils.ReadAndExtractBBBPData as DataReader
from sklearn.model_selection import train_test_split
np.random.seed(45)
tf.random.set_seed(45)

csv_path = './data/BBBP.csv'
df = DataReader.readBBBdata(csv_path)

df['molecule'] = df['smiles'].apply(DataReader.get_molecule_from_smiles)

df = df[df['molecule'].notna()]
print('trying to convert molecule object to graph')
X = DataReader.consruct_graph_from_smiles(df['smiles'])
y=df['p_np']
print(type(X[0]))
# df['graph_from_molecule']=df['molecule'].apply(DataReader.construct_graph_from_molecule)
# X= df['graph_from_molecule']
# y=df['p_np']
# print('successfully converted molecule object to graph')
#
# X_train,X_rem,y_train,y_rem = train_test_split(X,y,test_size=0.2)
# X_val,X_test,y_val,y_test = train_test_split(X_rem,y_rem,test_size=0.1)
# print('shape of training set: ',X_train.shape)
# print('shape of validation set: ',X_val.shape)
# print('shape of test set: ',X_test.shape)
# print(type(list(X_train)[0][0]))
# print(f"Name:\t{df.name[100]}\nSMILES:\t{df.smiles[100]}\nBBBP:\t{df.p_np[100]}")
# molecule = DataReader.get_molecule_from_smiles(df.iloc[100].smiles)
# print("Molecule:")
# print(molecule)
#
