""" Functions for Mol Denoisng Autoencoder """
import numpy as np
from rdkit import Chem

class DAE_process:
    
    def __init__(self, max_smiles_len, vocab_unique_chars, 
                 index_to_char, char_to_index):
        """ 
        Args:
            max_smiles_len (int): maximum lengh of SMILES strings including padding characters ('')
            vocab_unique_chars (set): set of unique characters in SMILES strings dataset 
            index_to_char (numpy.ndarray): maps unique index/token to SMILES character 
            char_to_index (dict): map SMILES character to unique index/token
        """
        
        self.max_smiles_len = max_smiles_len
        self.vocab_unique_chars = vocab_unique_chars
        self.index_to_char = index_to_char
        self.char_to_index = char_to_index
        
    def one_hot_encode_SMILES(self, smile):
        """ one hot encode SMILES string
        
        Args:
            smile (str): SMILES string
        
        Returns:
            (numpy.ndarray): one hot encoded vector representing SMILES, including one hot codes 
            for padding character to equalize lengths up to maximum length
        """
        
        vec = np.empty(shape = [0,0])
        for char in smile:
            block = np.zeros(len(self.index_to_char))
            block[self.char_to_index[char]] = 1.0
            new = np.concatenate((vec, block), axis = None)
            vec = new
        return vec
    
    
    def one_hot_vector_to_smiles(self, out_vector):
        """ Convert DAE's output vector to SMILES by checking each vocab block in array 
        and finding SMILES characer corresponding to maximum index 
        
        Args:
            out_vector (numpy.ndarray): DAE output vector (array of sigmoid float32 values)
        
        Returns:
            (str): SMILES string mapped from DAE output vector
        """
        
        max_prob_indexes_each_block = list()
        for i in range(len(out_vector) - len(self.index_to_char)):
            if i % len(self.index_to_char) == 0:
                for j in range(len(self.index_to_char)):
                    if out_vector[i:i+len(self.index_to_char)][j] == max(out_vector[i:i+len(self.index_to_char)]):
                        max_prob_indexes_each_block.append(j)
        return ' '.join([self.index_to_char[ind] for ind in max_prob_indexes_each_block if self.index_to_char[ind] != ' ']).replace(' ', "")
      
   
    
    def check_SMILES(self, s):
        """ Check chemical validity of SMILES string
        
        Args:
            s(str): SMILES string
        
        Returns:
            (bool): chemical validity of string
        """
        
        return Chem.MolFromSmiles(s) != None
    
    
