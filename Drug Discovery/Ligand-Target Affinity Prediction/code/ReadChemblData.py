file = open('../data/BindingDB_BindingDB_Articles_202405.tsv')
lines = file.readlines()
file.close()
lines=[l.replace('\n',',') for l in lines]

print(len(lines))
print(lines[0])
individual_lines = [l.split('\t') for l in lines]
print(len(individual_lines))
print(len(individual_lines[1]))
print(individual_lines[1])
print(len(individual_lines[2]))
print(individual_lines[2])
# file=open('../data/BindingDB_BindingDB_Articles_202405.csv','w+')
# file.writelines(''.join(lines))
# file.close()
"""import re
file = open('../data/BindingDB_BindingDB_Articles_202405.csv','r')
lines=file.readlines()
file.close()
print(lines[0])
print(lines[1])
print('before replacing bad author column')
abc = lines[32]
print(abc)
reg_to_replace = r'[A-Za-z]*\, [A-Z]'
print(re.search(reg_to_replace,abc))
abc = re.sub(reg_to_replace,r'author name',abc)
lines= [re.sub(reg_to_replace,r'author name',l) for l in lines]
print('after replacing bad author column')
print(abc)
print('number of columns: ', len(lines[0].split(',')))
print('number of lines in the first sample: ', len(lines[1].split(',')))
print('number of lines in the second sample: ', len(lines[2].split(',')))
print('number of lines in the 32nd sample: ', len(lines[32].split(',')))
print('number of lines in the 32nd sample after cleanup: ', len(abc.split(',')))
print('number of lines in the 33rd sample: ', len(lines[33].split(',')))

# import pandas as pd
# path_to_file = '../data/BindingDB_BindingDB_Articles_202405.csv'
# df = pd.read_csv(path_to_file)"""

