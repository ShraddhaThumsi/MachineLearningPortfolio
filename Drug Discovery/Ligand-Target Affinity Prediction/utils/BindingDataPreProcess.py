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
    longest_sequence_length = max(df['amino_acid_numerical_representation'].apply(len))
    #padding shorter protein sequences to the same length as the longest sequence
    df['amino_acid_numerical_representation'] = df['amino_acid_numerical_representation'].apply(lambda x: x + [0] * (longest_sequence_length - len(x)))
    return df
def get_atom_properties(atom):

    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.Hybridization(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic()

    ]
def get_bond_properties(bond):
    return [
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.GetIsInRing(),
        bond.GetBondOrder()
    ]

def molecule_to_graph(molecule):
    num_of_atoms = molecule.GetNumAtoms()
    node_features = []
    edge_features = []
    for atom in molecule.GetAtoms():
        node_features.append(get_atom_properties(atom))
    node_features = np.array(node_features)
    adjacency_matrix = np.zeros((num_of_atoms, num_of_atoms))
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency_matrix[i,j]=1
        adjacency_matrix[j,i]=1
        edge_features.append(get_bond_properties(bond))
    edge_features = np.array(edge_features)
    return node_features, adjacency_matrix, edge_features

def create_graph_tensor(node_properties,adjacency_matrix,edge_properties):
    num_of_nodes = node_properties.shape[0]
    num_of_edges = edge_properties.shape[0]
    edge_indices = np.vstack(np.where(adjacency_matrix)).T
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={'atoms': tfgnn.NodeSet.from_fields(features={'features': tf.convert_to_tensor(node_properties)},
                                                      sizes=tf.constant([num_of_nodes]))},
        edge_sets={'bonds': tfgnn.EdgeSet.from_fields(
            features={'features': tf.convert_to_tensor(edge_properties)},
            sizes=tf.constant([num_of_edges]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=('atoms', tf.convert_to_tensor(edge_indices[:, 0], dtype=tf.int32)),
                target=('atoms', tf.convert_to_tensor(edge_indices[:, 1], dtype=tf.int32))
            )
        )}
    )
    return graph_tensor

def build_gnn_model():
    input_graph = tf.keras.layers.Input(type_spec=tfgnn.GraphTensorSpec.from_piece_specs(
        node_sets_spec={
            'atoms': tfgnn.NodeSetSpec.from_field_specs(
                features_spec={'features': tf.TensorSpec(shape=[None, 5], dtype=tf.float32)},
                sizes_spec=tf.TensorSpec(shape=[None], dtype=tf.int32)
            )
        },
        edge_sets_spec={
            'bonds': tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={'features': tf.TensorSpec(shape=[None, 3], dtype=tf.float32)},
                sizes_spec=tf.TensorSpec(shape=[None], dtype=tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_field_specs(
                    source_spec=tf.TensorSpec(shape=[None], dtype=tf.int32),
                    target_spec=tf.TensorSpec(shape=[None], dtype=tf.int32)
                )
            )
        }
    ))


    graph_update = tfgnn.keras.layers.GraphUpdate(
        edge_sets={
            'bonds': tfgnn.keras.layers.EdgeSetUpdate(
                edge_input_feature_size=3,
                next_state=tfgnn.keras.layers.SimpleConv(tf.constant([32]))
            )
        },
        node_sets={
            'atoms': tfgnn.keras.layers.NodeSetUpdate(
                node_input_feature_size=5,
                next_state=tfgnn.keras.layers.SimpleConv(tf.constant([32])))

        }
    )
    graph_tensor = input_graph
    graph_tensor = graph_update(graph_tensor)

    readout = tfgnn.keras.layers.Pool('mean')
    graph_tensor = readout(graph_tensor)

    output = tf.keras.layers.Dense(8)(graph_tensor)

    model = tf.keras.Model(inputs=input_graph, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
