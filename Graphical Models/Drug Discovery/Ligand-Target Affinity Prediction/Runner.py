import utils.ReadBindingDBData as Read
import utils.BindingDataPreProcess as LigandData
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df = Read.make_df()
df = LigandData.extract_ligand_data(df)
df['molecule'] = df['Ligand SMILES'].apply(LigandData.smiles_to_molecule)
print('shape of df after extracting smile data')


df = df.dropna(subset=['molecule'])
print(df.shape)
df = LigandData.protein_sequence_embeddings(df)
print('shape of df after extracting protein sequence embedding data')
print(df.shape)
print(df.columns)
X,y=LigandData.prepare_dataset_from_tensor(df)
print(type(X))
print(type(y))
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=56)
train_dataset = LigandData.graph_tensor_dataset(X_train, y_train).batch(1)
test_dataset = LigandData.graph_tensor_dataset(X_test, y_test).batch(1)
model = LigandData.build_gnn_model(X_train)
#training loop
epochs=1
model.fit(train_dataset,epochs=epochs)
y_pred = model.predict(test_dataset).numpy()
mse = mean_squared_error(y_test, y_pred)
print(f'the rmse after training on {epochs} epochs is {mse}')

