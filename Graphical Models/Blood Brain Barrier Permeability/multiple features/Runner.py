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

X = DataReader.consruct_graph_from_smiles(df['smiles'])
y=df['p_np']

# Shuffle array of indices ranging from 0 to 2049
permuted_indices = np.random.permutation(np.arange(df.shape[0]))

# Train set: 80 % of data
train_index = permuted_indices[: int(df.shape[0] * 0.8)]
x_train = DataReader.consruct_graph_from_smiles(df.iloc[train_index].smiles)
y_train = df.iloc[train_index].p_np
y_train = y_train.astype('float32')



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
mpnn = Modl.MPNNModel(
    atom_dim=x_train[0][0][0].shape, bond_dim=x_train[1][0][0].shape,
)
mpnn.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),metrics=[keras.metrics.AUC(name="AUC")])
print('model compilation is complete')
history = mpnn.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    verbose=2,
    class_weight={0: 2.0, 1: 0.5},
)

molecules = [DataReader.consruct_graph_from_smiles(df.smiles.values[index]) for index in test_index]
y_true = [df.p_np.values[index] for index in test_index]
y_pred = tf.squeeze(mpnn.predict(test_dataset), axis=1)

legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f}" for i in range(len(y_true))]
MolsToGridImage(molecules, molsPerRow=4, legends=legends)