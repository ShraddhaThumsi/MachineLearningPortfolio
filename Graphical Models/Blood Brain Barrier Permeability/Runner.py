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
import utils.PrepEvalModel as Modl
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
# Shuffle array of indices ranging from 0 to 2049
permuted_indices = np.random.permutation(np.arange(df.shape[0]))

# Train set: 80 % of data
train_index = permuted_indices[: int(df.shape[0] * 0.8)]
x_train = DataReader.consruct_graph_from_smiles(df.iloc[train_index].smiles)
y_train = df.iloc[train_index].p_np
y_train = y_train.astype('float32')
print(type(x_train))
print(type(list(x_train)[0]))


# Valid set: 19 % of data
valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.99)]
x_valid =DataReader.consruct_graph_from_smiles(df.iloc[valid_index].smiles)
y_valid = df.iloc[valid_index].p_np
y_valid = y_valid.astype('float32')



# Test set: 1 % of data
test_index = permuted_indices[int(df.shape[0] * 0.99) :]
x_test = DataReader.consruct_graph_from_smiles(df.iloc[test_index].smiles)
y_test = df.iloc[test_index].p_np
y_test = y_test.astype('float32')




train_dataset = Modl.MPNNDataset(x_train,y_train)

validation_dataset = Modl.MPNNDataset(x_valid,y_valid)

test_dataset = Modl.MPNNDataset(x_test,y_test)

