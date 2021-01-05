# Mol-Denoising-Autoencoder

De-noising Autoencoder implementation in TensorFlow 2.0 for de-noising chemically invalid SMILES strings to valid analogs (For analog generation/Post-processing generative models data) using one hot encoding for SMILES featurization

### Description
Generative models such as Variational Autoencoders and Restricted Boltzmann Machines used for generating molecular analogs rarely achieve perfect results in terms of the chemical validity of the generated molecules. This project is a first iteration exploration in using a Denoising Autoencoder for denoising such chemically invalid SMILES molecules into valid analogs - with potential use cases being a post-processing step for generative models and as a SMILES analog generating model in itself. 

### Featurization
This iteration of the DAE uses one hot encoding to feauturize noisy input SMILES strings and decode output vectors to SMILES, as illustrated below. 
