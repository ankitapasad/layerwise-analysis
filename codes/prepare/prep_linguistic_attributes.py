"""
Prepare embedding files from semantic and syntactic attributes
https://github.com/ytsvetko/qvec/tree/master/oracles
"""

import ast
import fire
import numpy as np
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.insert(0, os.path.join(curr_dir, ".."))
from utils import save_dct, write_to_file, load_dct, read_lst

class PrepData():
    def __init__(self, fname):
        self.data = read_lst(fname)
        self.prop_to_idx, self.idx_to_prop = self.get_property_idx_map()
        self.wrd_embed_dct = self.get_embed_dct()

    def get_property_idx_map(self):
        prop_to_idx = {}
        idx_to_prop = {}
        idx = 0
        for line in self.data:
            _, dct_str = line.split("\t")
            prop_dct = ast.literal_eval(dct_str)
            for prop in prop_dct:
                if prop not in prop_to_idx:
                    prop_to_idx[prop] = idx
                    idx_to_prop[idx] = prop
                    idx += 1
        return prop_to_idx, idx_to_prop
    
    def get_embed_dct(self):
        wrd_embed_dct = {}
        num_props = len(self.prop_to_idx)
        for line in self.data:
            word, dct_str = line.split("\t")
            prop_dct = ast.literal_eval(dct_str)
            arr = np.zeros(num_props)
            for prop, value in prop_dct.items():
                arr[self.prop_to_idx[prop]] = value
            wrd_embed_dct[word] = arr
        return wrd_embed_dct

def main(fname, property_name, save_dir):
    data_obj = PrepData(fname)
    embed_fn = f'{property_name}_properties.pkl'
    save_dct(os.path.join(save_dir, embed_fn), data_obj.wrd_embed_dct)

if __name__ == "__main__":
    fire.Fire(main)