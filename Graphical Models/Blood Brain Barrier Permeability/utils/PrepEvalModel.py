import tensorflow as tf
def prepare_batch(X_batch,y_batch):
    atom_features,bond_features,pair_indices = X_batch
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()
    molecule_positions = tf.range(len(num_atoms))
    atom_partition_indices = tf.repeat(molecule_positions,num_atoms)
    bond_partition_indices = tf.repeat(molecule_positions[:-1],num_bonds[1:])
    #i want to find out how many atoms are there in each molecule so that i know how to pad the shortest molecule to the same size as the longest molecule
    #i want to merge the information from each subgraph or molecule so that i get a giant graph where each molecule is disjoint from the other, like a large forest of graphs

    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, bond_partition_indices),[(num_bonds[0],0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, atom_partition_indices), y_batch
def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    print('inside mpnn dataset function, checking the shape of the dataset after all the fluff', tf.data.experimental.cardinality(dataset))
    res = prepare_batch(X,y)
    print(res)
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(
        prepare_batch, num_parallel_calls=tf.data.AUTOTUNE
    )
