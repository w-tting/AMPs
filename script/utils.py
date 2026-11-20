import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

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

def AMP_test_performance(model):
    df=pd.read_csv('../data/split_data/test_final.csv')
    df=df[(df['value'] <4)]
    sequences=df['sequence']
    x_test=encode(sequences)
    y_test=df['value'].to_numpy()
    y_pred = model.predict(x_test)
    spear=spearmanr(y_test, y_pred)    
    pear=pearsonr(y_test, y_pred)

    print(f'Spearman = {spear[0]:.3f}, Pearson ={pear[0][0]:.3f}')
    
    return spear[0], pear[0], y_test, y_pred
        
def rank_percentile(values):
    n = len(values)
    rank = values.rank(method='average', ascending=True)
    percentile= 100 * (n - rank) / (n- 1)
    return percentile
