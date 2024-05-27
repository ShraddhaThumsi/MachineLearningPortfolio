import utils.ReadBindingDBData as Read
import utils.BindingDataPreProcess as LigandData


df = Read.make_df()
df = LigandData.extract_ligand_data(df)
df['molecule'] = df['Ligand SMILES'].apply(LigandData.smiles_to_molecule)
print(df.shape)

df = df.dropna(subset=['molecule'])
print(df.shape)