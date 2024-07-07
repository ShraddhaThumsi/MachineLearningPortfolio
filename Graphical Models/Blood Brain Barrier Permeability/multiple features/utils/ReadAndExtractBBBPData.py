from rdkit import Chem
import pandas as pd
import numpy as np
import tensorflow as tf
#data source credits "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"

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
    pair_indices=[]
    for id,atom in enumerate(molecule.GetAtoms()):
        atom_properties.append(get_properties_of_atom(atom)[1])
        pair_indices.append([atom.GetIdx(),atom.GetIdx()])
        #bond_properties.append(get_properties_of_bond(None))
        for n_id,neighbor in enumerate(atom.GetNeighbors()):
            bond=molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            bond_properties.append(get_properties_of_bond(bond)[1])
            atom_properties.append(get_properties_of_atom(neighbor)[1])

            pair_indices.append([float(atom.GetIdx()), float(neighbor.GetIdx())])
    return np.array(atom_properties).astype('float32'), np.array(bond_properties).astype('float32'),np.array(pair_indices).astype('float32')
def consruct_graph_from_smiles(list_of_smiles):
    all_molecules_atom_properties = []
    all_molecules_bond_properties = []
    all_molecules_pair_indices_list=[]
    for smiles in list_of_smiles:
        molecule = get_molecule_from_smiles(smiles)
        permol_atom_prop,permol_bond_prop,permol_pairs = construct_graph_from_molecule(molecule)
        all_molecules_atom_properties.append(permol_atom_prop)
        all_molecules_bond_properties.append(permol_bond_prop)
        all_molecules_pair_indices_list.append(permol_pairs)

    return (tf.ragged.constant(all_molecules_atom_properties,dtype=tf.float32),
            tf.ragged.constant(all_molecules_bond_properties,dtype=tf.float32),
            tf.ragged.constant(all_molecules_pair_indices_list,dtype=tf.float32)
            )