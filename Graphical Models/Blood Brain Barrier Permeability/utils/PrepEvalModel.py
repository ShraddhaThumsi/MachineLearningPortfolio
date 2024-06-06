import tensorflow as tf
import numpy as np


def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """
    atom_features, bond_features, pair_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices. atom_partition_indices will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = tf.range(len(num_atoms))
    atom_partition_indices = tf.repeat(molecule_indices, num_atoms)
    bond_partition_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    increment = tf.cumsum(tf.cast(num_atoms[:-1],tf.float32))

    increment = tf.pad(
        tf.gather(increment, bond_partition_indices), [(num_bonds[0], 0)]
    )

    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    print('inside prepare_batch function, checking datatype of pair_indices')
    print(type(pair_indices))
    print('checking type of increment object')
    print(type(increment[:, tf.newaxis]))
    pair_indices = pair_indices + increment[:, tf.newaxis]
    print('converted pair indices to float successfully and also appended increment object')
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, atom_partition_indices), y_batch


def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))

    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(
        prepare_batch, num_parallel_calls=tf.data.AUTOTUNE
    )

class EdgeNetwork(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def build(self,input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim,self.atom_dim*self.atom_dim),
            trainable=True,
            initializer='glorot_uniform'
        )
        self.bias=self.add_weight(
            shape=(self.atom_dim*self.atom_dim),
            trainable=True,
            initializer='zeros'
        )
        self.built = True
    def call(self,inputs):
        atom_features,bond_features,pair_indices=inputs
        bond_features=tf.matmul(bond_features,self.kernel)+self.bias
        bond_features=tf.reshape(bond_features,(-1,self.atom_dim,self.atom_dim))
        atom_features_neighbors=tf.gather(atom_features,pair_indices[:,1])
        atom_features_neighbors=tf.expand_dims(atom_features_neighbors,axis=-1)
        transformed_features=tf.matmul(bond_features,atom_features_neighbors)
        transformed_features=tf.squeeze(transformed_features,axis=-1)
        aggregated_features=tf.math.segment_sum(transformed_features,pair_indices[:,0])
        return aggregated_features

class MessagePassing(tf.keras.layers.Layer):
    def __init__(self,units,steps=4,**kwargs):
        super().__init__(**kwargs)
        self.units=units
        self.steps=steps
    def build(self,input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step=EdgeNetwork()
        self.pad_length=max(0,self.units-self.atom_dim)
        self.update_step=tf.keras.layers.GRUCell(self.atom_dim+self.pad_dim)
        self.built=True
    def call(self,inputs):
        atom_features,bond_features,pair_indices=inputs
        atom_features_updated=tf.pad(atom_features,[(0,0),(0,self.pad_length)])
        for i in range(self.steps):
            atom_features_aggregated=self.message_step([atom_features_updated,bond_features,pair_indices])
            atom_features_updated,_=self.update_step(atom_features_aggregated,atom_features_updated)
        return atom_features_updated
