""" Demo and test of Mol DAE model """
import numpy as np
import pandas as pd
import tensorflow as tf

from DAE_Chem_utils import DAE_process

""" Re-process noisy dataset and instantiate DAE_process """
data_noisy = pd.read_csv('noisy_train_set.csv')
smiles_noisy = data_noisy['noisy_SMILES']

# Define maximum SMILES length in terms of characters, 
# including padding character (spaces)
max_smiles_len = 110

# Ensure all SMILES are within character limit by filtering
smiles_noisy = list(filter(lambda smile : len(smile) <= max_smiles_len, smiles_noisy))

# Get vocabulary of unique characters set
vocab_unique_chars = set()
for smile in smiles_noisy:
    for char in smile:
        vocab_unique_chars.add(char)

# Map index identifier to SMILES character
index_to_char = np.array(sorted(vocab_unique_chars))

# Map SMILES character to index identifier
char_to_index = {y:x for (x, y) in enumerate(index_to_char)}

# Instantiate DAE_process class
dae_process = DAE_process(max_smiles_len, vocab_unique_chars, index_to_char, char_to_index)

# Convert noisy SMILES to one hot encoded vector input arrays
X_noisy = np.array([dae_process.one_hot_encode_SMILES(s) for s in smiles_noisy])

# Load model
DAE_model = tf.keras.models.load_model('Final_DAE_model')

# Define function that converts noisy input SMILES to denoised SMILES corresponding with DAE output vector
def DAE_Denoised_SMILES(noise_smiles):
    input_one_hot_SMILES_vec = dae_process.one_hot_encode_SMILES(noise_smiles)
    out_vec = DAE_model.predict(np.array([input_one_hot_SMILES_vec,]))[0]
    return dae_process.one_hot_vector_to_smiles(out_vec)


""" Test function on sample noisy and invalid SMILES molecules """
    
noisy_invalid_SMILES = smiles_noisy[1090] # '=C(OO)CC1CCC(CC1)c2ccc(cc2)c3ccc4nc(cn4c3)C(=Nc5cc(F)c(F)c(F)c5)O 

# Check chemical validity of SMILES using rdkit function
dae_process.check_SMILES(noisy_invalid_SMILES) # False

# Input noisy invalid SMILES sample to DAE
DAE_denoised_output_SMILES = DAE_Denoised_SMILES(noisy_invalid_SMILES)

# Check chemical validity of DAE output SMILES
dae_process.check_SMILES(DAE_denoised_output_SMILES) # True


