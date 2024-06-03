from rdkit import Chem
import pandas as pd
import numpy as np
import tensorflow as tf
#data source credits "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
#python utils credits https://www.kaggle.com/code/thumsishraddha/week11-12-graph-neural-network-gnn/edit

def readBBBdata(path):
    df = pd.read_csv(path,usecols=[1,2,3])
    return df

def get_properties_of_atom(atom):
    return (['Atomic Degree',
             'Is Atom Aromatic',
             'Atom Hybridization',
             'Number of Hydrogens',
             'Total Valency',
             'Number of Radical Electrons',
             'Formal Charge',
             'Atomic Mass',
             'Is Atom in Ring'],
            [atom.GetDegree(),
            float(atom.GetIsAromatic()),
            atom.GetHybridization(),
            atom.GetTotalNumHs(),
             atom.GetTotalValence(),
             atom.GetNumRadicalElectrons(),
             atom.GetFormalCharge(),
             atom.GetMass(),
             float(atom.IsInRing())])

def get_properties_of_bond(bond):
    index_dict = {'aromatic':0,'single':1,'double':2,'triple':3}
    return (['Bond Type',
            'Is Aromatic',
             'Is Conjugated',
             'Bond is in ring'],
            [index_dict[bond.GetBondType().name.lower()],
             float(bond.GetIsAromatic()),
             float(bond.GetIsConjugated()),
             float(bond.IsInRing())])

def get_molecule_from_smiles(smiles_string):
    return Chem.MolFromSmiles(smiles_string)

def construct_graph_from_molecule(molecule):

    atom_properties = []
    bond_properties = []

    num_atoms = molecule.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms))
    for id,atom in enumerate(molecule.GetAtoms()):
        atom_properties.append(get_properties_of_atom(atom)[1])
        adjacency_matrix[id,id]=1
        #bond_properties.append(get_properties_of_bond(None))
        for n_id,neighbor in enumerate(atom.GetNeighbors()):
            bond=molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            bond_properties.append(get_properties_of_bond(bond)[1])
            atom_properties.append(get_properties_of_atom(neighbor)[1])
            adjacency_matrix[id,n_id]=1
            adjacency_matrix[n_id,id]=1

    return (tf.ragged.constant(atom_properties,dtype=tf.float32),
            tf.ragged.constant(bond_properties,dtype=tf.float32),
            tf.ragged.constant(adjacency_matrix,dtype=tf.float32)
            )