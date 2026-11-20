import numpy as np
import pandas as pd
from utils import encode, AMP_test_performance
from fine_tune import finetune_model

def subsampling(input_csv, sizes, runs=20):
    df = pd.read_csv(input_csv)
    for i in range(runs):
        sampled_df = df.sample(n=sizes)
        sequences=sampled_df['sequence']
        X_tune=encode(sequences)
        y_tune=sampled_df['value'].to_numpy()
        model=finetune_model('../model/Pretrained.h5', X_tune, y_tune)
        AMP_test_performance(model)
    print('Sbsampling Finished.')
    
def label_permutation(input_csv, runs=50):
    sampled_df=pd.read_csv(input_csv)
    sequences=sampled_df['sequence']
    X_tune=encode(sequences)
    for i in range(runs):
        y_tune= np.random.default_rng().permutation(sampled_df['value'].to_numpy())
        model=finetune_model('../model/Pretrained.h5', X_tune, y_tune)
        AMP_test_performance(model)
    print('Label shuffled Finished.')
    
if __name__ == '__main__':
    #RM72
    input_csv='../data/RM.csv'
    subsampling(input_csv, runs=20, sizes=72)
    #SM35
    input_csv='../data/SM.csv'
    subsampling(input_csv, runs=20, sizes=35)
    #Shuffled labels
    input_csv='../data/SM.csv'
    label_permutation(input_csv, runs=50)