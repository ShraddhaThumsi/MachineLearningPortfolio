import utils.ReadLiteratureData as read
import utils.EDA as EDA

train_data,test_data = read.read_and_clean_df()
print(train_data.shape)
EDA.analyze_gene_data(train_data)