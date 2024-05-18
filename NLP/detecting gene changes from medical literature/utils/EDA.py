import matplotlib.pyplot as plt
def analyze_gene_data(df):
    print(df['Variation'].value_counts())
    print(f'number of classes into which observations are bucketed: {len(set(df["Class"]))}')

    print('counting frequency by class')



    plt.hist(df["Class"], bins=9,rwidth=0.8,color='cyan')
    plt.savefig('./output/class_histogram.png')
    plt.show()

    print('counting frequency by gene')

    df_gene = df.groupby(by="Gene")
    df_gene_plot = df_gene.count().sort_values(by='ID',ascending=False).head(10)
    df_gene_plot.plot(kind='bar', y ='ID', ylabel = 'Frequency',xlabel='Variation',color='#A1CAD2')
    plt.plot(df_gene_plot,color='orange',linewidth=1)
    plt.xticks(rotation=45)
    plt.xlabel('Gene')
    plt.ylabel('Count')
    plt.savefig('./output/gene_counts.png')
    plt.show()

    print('counting frequency by variation')
    df_variation = df.groupby(by="Variation").count().sort_values(by='ID',ascending=False).head(10)
    df_variation.plot(kind='bar', y ='ID', ylabel = 'Frequency',xlabel='Variation',color='purple')
    plt.plot(df_variation,color='#D2A9A1',linewidth=1)
    plt.xticks(rotation=-10)
    plt.xlabel('variation in the gene')
    plt.ylabel('Count')
    plt.legend(loc='upper right')
    plt.savefig('./output/variations.png')
    plt.show()

    return df
