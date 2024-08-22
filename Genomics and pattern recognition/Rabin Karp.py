import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
compounds = ['A','T','G','C']
index_dict={'A':0,'T':1,'G':2,'C':3}

base_string='CCAGTCAACTTTGGCCTGTGTGAACCTTATGTGTCATCAAATTTGGGCCTTTGGGGAAACCGGATGCGGAA'

candidate_string = 'ATGTGTCATC'
def hasher(cand):
    id_list = []
    hash_value=0
    for c in cand:
        index = index_dict[c]
        id_list.append(index)
    id = id_list[::-1]
    for pos,i in enumerate(id):
        hash_value+= i * (4 ** pos)
    return hash_value

def prepare_rolling_window(base,window_size):
    base_index_list = []
    print(base)
    for pos,_ in enumerate(base):

        base_index_list.append(pos)

    rolling_window = sliding_window_view(np.array(base_index_list), window_shape=window_size)
    rolling_window_as_str = []
    for window in rolling_window:
        letters = [base[i] for i in window]
        rolling_window_as_str.append(letters)
    print(f'rolling window: {rolling_window}')
    print(f'rolling window as string: {rolling_window_as_str}')
    return rolling_window_as_str

def rabin_karp(base,candidate):
    # this will return the list of indices where the small string appears
    print(f'i will search for {candidate} in {base}')
    hash_value = hasher(candidate)
    rolling_window_of_base = prepare_rolling_window(base,len(candidate))
    print(rolling_window_of_base)
    for pos,i in enumerate(rolling_window_of_base):

        hash_of_sub_part_of_base = hasher(i)
        if hash_of_sub_part_of_base == hash_value:
            return pos

    return -1
rabin_karp_index = rabin_karp(base_string,candidate_string)
if rabin_karp_index == -1:
    print(f'the string {candidate_string} is not found in {base_string}')
else:
    print(f'the string {candidate_string} is found in {base_string} in the position {rabin_karp_index}')


