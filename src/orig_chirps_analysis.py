import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx

#terminal alias: alias python='/opt/anaconda3/envs/BatGPT/bin/python'

#Use Apple Silicon M3 chip
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

#TODO: add model = model.to(device)


# load CSV 
df = pd.read_feather("orig_chirps_2024-06-25T12_55_03.feather")


"""CATEGORIZATION BY SPECIES

To ensure accuracy in analysis, separate the bats into dataframes by species. 
Since Coto is the species with the most examples, I will use that species for analysis. 
Each bat appears in the dataset with the following frequencies:

Species count: {'Unkn': 270078,
'Myca': 429,
'Tabr': 8060,
'Myyu': 9656,
'Lano': 770,
'Tabr,Myca': 17,
'Coto': 116696,
'Laci,Tabr': 133,
'Coto,Laci': 9,
'Myyu,Myca': 122,
'Laci': 39829,
'Myev': 1587,
'Anpa': 16758,
'Myth': 1532,
'Epfu': 523,
'Eupe': 1794,
'Euma': 329,
'Myci': 49,
'Mylu': 28}

I used this code to make this determination:

unique_species = df['species'].unique()
species_dfs = {}
for species in unique_species:
    df_name = f"df_{species}"
    species_dfs[df_name] = df[df['species'] == species]
    # Create the dataframe as a variable in the global namespace
    globals()[df_name] = species_dfs[df_name]
"""

# Create separate dataframes for each species - use df_Coto

# Create Coto-specific dataframe and remove specified columns
df_Coto = df[df['species'] == 'Coto'].copy()
df_Coto = df_Coto.drop(['species', 'sin_year', 'cos_year'], axis=1)

#Sort by time
df_Coto['rec_datetime'] = pd.to_datetime(df_Coto['rec_datetime'])
df_Coto = df_Coto.sort_values('rec_datetime').reset_index(drop=True)
print("\nFirst few rows of sorted df_Coto (showing datetime):")
print(df_Coto[['rec_datetime']].head())
print("\nLast few rows of sorted df_Coto (showing datetime):")
print(df_Coto[['rec_datetime']].tail())


# Create sequential edges (each node connects to the next one temporally)
num_nodes = len(df_Coto)
edge_index = torch.tensor([range(num_nodes-1), range(1, num_nodes)], dtype=torch.long)

"""
# Create feature label dictionary from DataFrame columns
feature_label_dict = {i: col for i, col in enumerate(df.columns)}

# Convert all DataFrame values to a tensor for node features
node_features = torch.tensor(df.values, dtype=torch.float32)  # [num_nodes, num_features]

# Create the graph with the full feature vectors
data = Data(x=node_features, edge_index=edge_index)
data = data.to(device)

# Print feature dimension info
print(f"\nNode feature dimensions: {node_features.shape}")
print(f"First 5 feature labels as example:")
for i in range(min(5, len(feature_label_dict))):
    print(f"Feature {i}: {feature_label_dict[i]}")

# Optional: add temporal information as a node attribute
data.time = torch.tensor(df['TimeInFile'].values, dtype=torch.float)
"""