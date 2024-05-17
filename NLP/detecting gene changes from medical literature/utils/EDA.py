import matplotlib.pyplot as plt
def analyze_gene_data(df):
    print(f'number of classes into which observations are bucketed: {len(set(df["Class"]))}')

    print('counting frequency by class')



    plt.hist(df["Class"], bins=9,rwidth=0.8,color='cyan')
    plt.show()

    print('counting frequency by gene')

    df_gene = df.groupby(by="Gene")
    df_gene_plot = df_gene.count().sort_values(by='ID',ascending=False).head(10)
    plt.plot(df_gene_plot,color='orange',linewidth=1)
    plt.xticks(rotation=45)
    plt.xlabel('Gene')
    plt.ylabel('Count')
    plt.show()

    return df
