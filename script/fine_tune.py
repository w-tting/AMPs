import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from utils import encode

def finetune_model(pretrained_model_path, X_new, y_new):
    print(f"Loading pretrained model: {pretrained_model_path} ...")
    model = load_model(pretrained_model_path)
 
    for layer in model.layers[:-1]:
        layer.trainable = False        
    model.layers[-1].trainable = True   
    
    fine_tune_lr = 1e-6
    optimizer = Adam(learning_rate=fine_tune_lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    print(f"Starting fine tuning...")
    model.fit(X_new, y_new, batch_size=16, epochs=50, validation_split=0.1, verbose=1)
    model.save(model_path)
    
    return model
    
if __name__ == '__main__':
    #SM or RM
    data_name='SM'
    data_path=f'../data/{data_name}.csv'
    data=pd.read_csv(data_path)
    sequences=df['sequence']
    X=encode(sequences)
    y=df['value'].to_numpy()
    
    pretrained_model_path='../model/Base.h5'
    tuned_model_path=f'../model/{data_name}.h5'
    tuned_model=finetune_model(pretrained_model_path, X, y)
    tuned_model.save(tuned_model_path)