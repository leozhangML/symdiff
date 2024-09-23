"""
This is a script to extract the EMA and normal trained DiT model from a Dit_Gaussian_Dynamics model.


The reason this is needed is because the normal saving of the model saves everything including the embeddings models
for the main DiT model.
"""


import torch
import argparse

parser = argparse.ArgumentParser(description='Saving models')

# Where is the original model saved?
parser.add_argument('--model_loc', type=str, default="Location of DiT Gaussian Dyanimics model")
