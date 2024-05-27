import utils.ReadBindingDBData as Read
import utils.BindingDataPreProcess as LigandData


df = Read.make_df()
df = LigandData.extract_ligand_data(df)
df['molecule'] = df['Ligand SMILES'].apply(LigandData.smiles_to_molecule)
print('shape of df after extracting smile data')


df = df.dropna(subset=['molecule'])
print(df.shape)
df = LigandData.protein_sequence_embeddings(df)
print('shape of df after extracting protein sequence embedding data')
print(df.shape)
