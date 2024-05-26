import pandas as pd
file = open('../data/BindingDB_BindingDB_Articles_202405.tsv')
lines = file.readlines()
file.close()
lines=[l.replace('\n',',') for l in lines]

individual_lines = [l.strip().split('\t') for l in lines]

def is_longer(line_item):
    return len(line_item) <= len(individual_lines[0])
long_lines = list(filter(is_longer, individual_lines))


df = pd.DataFrame(long_lines,columns=individual_lines[0])
print(df.shape)