import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
import tensorflow_gnn as tfgnn
from rdkit import Chem
from rdkit.Chem import rdchem

import numpy as np

def extract_ligand_data(df):

    subdf = df[['Ligand SMILES', 'BindingDB Target Chain Sequence', 'Kd (nM)', 'EC50 (nM)', 'kon (M-1-s-1)', 'koff (s-1)', 'pH','Temp (C)']]
    print(subdf.shape)
    return df

def smiles_to_molecule(smiles):
    #reading SMILE data into a molecule
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return mol
    except Exception as e:
        print(e)
        return None

def protein_sequence_embeddings(df):
    # Amino acids are represented by letters, so it is sufficient to represent the letters by integers 1 to 26
    possible_letters_for_amino_acids='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    amino_acids_to_int = {aa:i+1 for i,aa in enumerate(possible_letters_for_amino_acids)}

    def sequence_to_int(seq):
        seq = seq.upper()
        return [amino_acids_to_int[s] for s in seq if s in amino_acids_to_int]
    df['amino_acid_numerical_representation'] = df['BindingDB Target Chain Sequence'].apply(sequence_to_int)

    return df


