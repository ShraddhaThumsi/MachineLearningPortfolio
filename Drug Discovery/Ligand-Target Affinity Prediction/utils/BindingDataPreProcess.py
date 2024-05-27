import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
import tensorflow_gnn as tfgnn
from rdkit import Chem
from rdkit.Chem import rdchem
from transformers import BertTokenizer, TFBertModel
import numpy as np

def extract_ligand_data(df):

    subdf = df[['Ligand SMILES', 'BindingDB Target Chain Sequence', 'Kd (nM)', 'EC50 (nM)', 'kon (M-1-s-1)', 'koff (s-1)', 'pH','Temp (C)']]
    print(subdf.shape)
    return df

def smiles_to_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return mol
    except Exception as e:
        print(e)
        return None


