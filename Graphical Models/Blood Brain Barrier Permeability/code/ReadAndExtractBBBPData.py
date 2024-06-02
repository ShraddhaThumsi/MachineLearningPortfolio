from rdkit import Chem
import pandas as pd
import numpy as np
#data source credits "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
#python code credits https://www.kaggle.com/code/thumsishraddha/week11-12-graph-neural-network-gnn/edit
csv_path = '../data/BBBP.csv'
df = pd.read_csv(csv_path,usecols=[1,2,3])
print(df.shape)

def get_properties_of_atom(atom):
    return (['Atomic Degree',
             'Is Atom Aromatic',
             'Atom Hybridization',
             'Atomic Symbol',
             'Number of Hydrogens',
             'Total Valency',
             'Number of Radical Electrons',
             'Formal Charge',
             'Atomic Mass',
             'Is Atom in Ring'],
            [atom.GetDegree(),
            atom.GetIsAromatic(),
            atom.GetHybridization(),
            atom.GetSymbol(),
            atom.GetTotalNumHs(),
            atom.GetTotalValence(),
            atom.GetNumRadicalElectrons(),
            atom.GetFormalCharge(),
            atom.GetMass(),
            atom.GetIsInRing()])

def get_properties_of_bond(bond):
    if bond is None:
        return None
    else:
        return (['Bond Type',
            'Is Aromatic',
             'Is Conjugated',
             'Stereo configuration',
             'Bond is in ring'],
            [bond.GetBondType(),
             bond.GetIsAromatic(),
             bond.GetIsConjugated,
             bond.GetStereo(),
             bond.GetIsInRing()])

def get_molecule_from_smiles(smiles_string):
    return Chem.MolFromSmiles(smiles_string)

def construct_graph_from_molecule(molecule):

    atom_properties = []
    bond_properties = []

    num_atoms = molecule.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms))
    for id,atom in enumerate(molecule.GetAtoms()):
        atom_properties.append(get_properties_of_atom(atom))
        adjacency_matrix[id,id]=1
        bond_properties.append(get_properties_of_bond(None))
        for n_id,neighbor in enumerate(atom.GetNeighbors()):
            bond=molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            bond_properties.append(get_properties_of_bond(bond))
            atom_properties.append(get_properties_of_bond(neighbor))
            adjacency_matrix[id,n_id]=1
            adjacency_matrix[n_id,id]=1

    return np.array(atom_properties),np.array(bond_properties),np.array(adjacency_matrix)