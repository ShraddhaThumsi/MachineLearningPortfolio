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
        atom.GetHybridization(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic()

    ]
def get_bond_properties(bond):
    return [
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.IsInRing()
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

def create_graph_tensor(node_features,adjacency_matrix,edge_features):
    num_nodes = node_features.shape[0]
    num_edges = edge_features.shape[0]

    # Create edge indices
    edge_indices = np.vstack(np.where(adjacency_matrix)).T

    # Create GraphTensor
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={'atoms': tfgnn.NodeSet.from_fields(features={'features': tf.convert_to_tensor(node_features)},
                                                      sizes=tf.constant([num_nodes]))},
        edge_sets={'bonds': tfgnn.EdgeSet.from_fields(
            features={'features': tf.convert_to_tensor(edge_features)},
            sizes=tf.constant([num_edges]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=('atoms', tf.convert_to_tensor(edge_indices[:, 0], dtype=tf.int32)),
                target=('atoms', tf.convert_to_tensor(edge_indices[:, 1], dtype=tf.int32))
            )
        )}
    )

    return graph_tensor

def build_gnn_model(X):
    input_graph = tf.keras.layers.Input(type_spec=tfgnn.GraphTensorSpec.from_tensor(X[0]))
    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"atoms": tfgnn.keras.layers.NodeSetUpdate(
            {"bonds": tfgnn.keras.layers.SimpleConv(message_fn=tf.keras.layers.Dense(32), reduce_type="sum")})}
    )(input_graph)

    # Change to correct pooling layer
    pooled_graph = tfgnn.keras.layers.Pool(node_set_name="atoms", reduce_type="mean")(graph)
    output = tf.keras.layers.Dense(8)(pooled_graph)

    model = tf.keras.Model(inputs=[input_graph], outputs=[output])
    model.compile(optimizer='adam', loss='mse')
    return model
def prepare_dataset_from_tensor(df):
    X = []
    y = []
    for _, row in df.iterrows():
        molecule = row['molecule']
        if molecule is not None:
            node_features,adjacency_matrix,edge_features=molecule_to_graph(molecule)
            graph_tensor = create_graph_tensor(node_features,adjacency_matrix,edge_features)
            X.append(graph_tensor)
            target = np.array([row['Ki (nM)'], row['IC50 (nM)'], row['Kd (nM)'], row['EC50 (nM)'], row['kon (M-1-s-1)'],
                               row['koff (s-1)'], row['pH'], row['Temp (C)']])
            y.append(target)
    X = tf.ragged.constant(X)
    y= tf.convert_to_tensor(y)
    return X,y
def graph_tensor_dataset(X, y):
    def gen():
        for graph_tensor, target in zip(X, y):
            yield graph_tensor, target
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tfgnn.GraphTensorSpec.from_tensor(X[0]),
            tf.TensorSpec(shape=(8,), dtype=tf.float32)
        )
    )