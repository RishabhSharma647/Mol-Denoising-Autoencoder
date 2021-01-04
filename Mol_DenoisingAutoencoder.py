""" Denoising Autoencoder for SMILES Molecules using one hot encoding featurization """

import numpy as np
import pandas as pd
import tensorflow as tf

from DAE_Chem_utils import DAE_process
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dropout

# Load and process datasets
data = pd.read_csv('train_set.csv')
data_noisy = pd.read_csv('noisy_train_set.csv')

smiles = data['SMILES']
smiles_noisy = data_noisy['noisy_SMILES']

# Define maximum SMILES length in terms of characters, 
# including padding character (spaces)
max_smiles_len = 110

# Ensure all SMILES are within character limit by filtering
smiles = list(filter(lambda smile : len(smile) <= max_smiles_len, smiles))
smiles_noisy = list(filter(lambda smile : len(smile) <= max_smiles_len, smiles_noisy))

# Get vocabulary of unique characters set
vocab_unique_chars = set()
for smile in smiles:
    for char in smile:
        vocab_unique_chars.add(char)

# Map index identifier to SMILES character
index_to_char = np.array(sorted(vocab_unique_chars))

# Map SMILES character to index identifier
char_to_index = {y:x for (x, y) in enumerate(index_to_char)}

# Instantiate DAE_process class
dae_process = DAE_process(max_smiles_len, vocab_unique_chars, index_to_char, char_to_index)

# One hot encode SMILES and create input and output arrays for original and noisy datasets
X = np.array([dae_process.one_hot_encode_SMILES(s) for s in smiles])
X_noisy = np.array([dae_process.one_hot_encode_SMILES(s) for s in smiles_noisy])

# Define Denoisong Autoencoder network hyper parameters
hidden_dim = 1900
compressed_dim = 1500
input_dim = max_smiles_len*len(index_to_char) # 3630


# Define Denoising Autoencoder
DAE = tf.keras.Sequential(
        [
           layers.Dense(hidden_dim, activation = 'sigmoid', name = 'hidden_1'),
           layers.Dense(compressed_dim, activation = 'sigmoid', name = 'hidden_2_bottleneck'),
           layers.Dense(hidden_dim, activation = 'sigmoid', name = 'hidden_3'),
           layers.Dense(input_dim, activation = 'sigmoid', name = 'output_denoised_reconstruction'),
        ]
)


# Compile model
DAE.compile(loss='categorical_crossentropy', optimizer='adam')

# Define batch size and number of epochs
batch_size = 100
num_epochs = 150
filepath = "saved_nets/DAE_weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Fit model
DAE.fit(X_noisy, X, 
          epochs = num_epochs,
          batch_size = batch_size,
          callbacks = [checkpoint],
          validation_split = .15,
          shuffle = True)


