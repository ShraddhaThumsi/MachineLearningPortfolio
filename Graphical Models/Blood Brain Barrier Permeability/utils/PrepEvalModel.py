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
            shape=(self.atom_dim*self.atom_dim,),
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
        self.update_step=tf.keras.layers.GRUCell(self.atom_dim+self.pad_length)
        self.built=True
    def call(self,inputs):
        atom_features,bond_features,pair_indices=inputs
        atom_features_updated=tf.pad(atom_features,[(0,0),(0,self.pad_length)])
        for i in range(self.steps):
            atom_features_aggregated=self.message_step([atom_features_updated,bond_features,pair_indices])
            atom_features_updated,_=self.update_step(atom_features_aggregated,atom_features_updated)
        return atom_features_updated

class PartitionPadding(tf.keras.layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, atom_partition_indices = inputs

        # Obtain subgraphs
        atom_features = tf.dynamic_partition(
            atom_features, atom_partition_indices, self.batch_size
        )

        # Pad and stack subgraphs
        num_atoms = [tf.shape(f)[0] for f in atom_features]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_padded = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features, num_atoms)
            ],
            axis=0,
        )

        # Remove empty subgraphs (usually for last batch)
        nonempty_examples = tf.where(tf.reduce_sum(atom_features_padded, (1, 2)) != 0)
        nonempty_examples = tf.squeeze(nonempty_examples, axis=-1)

        return tf.gather(atom_features_padded, nonempty_examples, axis=0)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, embed_dim=64, dense_dim=512, **kwargs):
        super().__init__(**kwargs)

        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [tf.keras.layers.Dense(dense_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        attention_mask = mask[:, tf.newaxis, :] if mask is not None else None
        attention_output = self.attention(inputs, inputs, attention_mask=attention_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        return self.layernorm_2(proj_input + self.dense_proj(proj_input))

def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):
    print('inside MPNN Model function, atom dim is, ', atom_dim)
    print('inside MPNN Model function, bond dim is, ', bond_dim)
    atom_features = tf.keras.layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = tf.keras.layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = tf.keras.layers.Input((2,), dtype="int32", name="pair_indices")
    atom_partition_indices = tf.keras.layers.Input(
        (), dtype="int32", name="atom_partition_indices"
    )

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = PartitionPadding(batch_size)([x, atom_partition_indices])

    x = tf.keras.layers.Masking()(x)

    x = TransformerEncoder(num_attention_heads, message_units, dense_units)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(
        inputs=[atom_features, bond_features, pair_indices, atom_partition_indices],
        outputs=[x],
    )
    return model


