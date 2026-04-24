import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CHARACTER_DICT = 'ACDEFGHIKLMNPQRSTVWY'
character_to_index = {(character): i for i, character in enumerate(CHARACTER_DICT)}
index_to_character = {i: (character) for i, character in enumerate(CHARACTER_DICT)}

num_chars=len(character_to_index)
MAX_MIC=4
MAX_SEQUENCE_LENGTH=50

config = { "font.family":'Arial','axes.labelsize':31, 'axes.titlesize':37, 'legend.fontsize':28, 'xtick.labelsize': 31, 'ytick.labelsize': 31, 'axes.linewidth':1.5}

def encode(seqs):
    if isinstance(seqs, str):
        seqs = [seqs]
    encoded = []
    for seq in seqs:
        vector = np.zeros([MAX_SEQUENCE_LENGTH, num_chars])
        for i, character in enumerate(seq[:MAX_SEQUENCE_LENGTH]):
            vector[i][character_to_index[character]] = 1
        encoded.append(vector)
    return np.array(encoded)
    
def split_data(data_path):
    data=pd.read_csv(data_path,index_col=False)
    sequences=df['sequence']
    X=encode(sequences)
    y=df['value'].to_numpy()
    
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    
    return X_train, x_test, y_train, y_test