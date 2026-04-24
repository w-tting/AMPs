import numpy as np
import pandas as pd
from utils import MAX_SEQUENCE_LENGTH, num_chars, split_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, ZeroPadding1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from model import cnn_model

def Hyperparameters_search(param_grid, X_train, y_train, build_model):
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    best_loss = float('inf') 
    best_params = {}
    
    all_results = []
    
    print(f"Start the search, {len(param_combinations)} combinations in total...")
    
    for params in param_combinations:
        print()
        print(f"The parameters are currently being tested: {params}")
        
        batch_size = params.pop('batch_size')
        epochs = params.pop('epochs')
        
        fold_histories = []
        fold_losses = []
        
        for train_index, val_index in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]     
            
            model = build_model(**params)
            history=model.fit(X_fold_train, y_fold_train, validation_data=(X_fold_val, y_fold_val), batch_size=batch_size, epochs=epochs, verbose=0)      
            val_loss= model.evaluate(X_fold_val, y_fold_val, verbose=0)
            fold_histories.append(history.history)
            fold_losses.append(val_loss)
         
        avg_loss = np.mean(fold_losses)   
        std_loss= np.std(fold_losses)
    
        print(f"Validation avg loss/std loss: {avg_loss:.4f} (+/- {std_loss:.4f})")
  
        params['batch_size'] = batch_size
        params['epochs'] = epochs
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = params
    print(f"\Best parameter combination: {best_params}")
    print(f"Best cross validation loss: {best_loss:.4f}")
    
    return best_params
param_grid = {
    'batch_size': [16, 20, 32],
    'epochs': [10, 30, 50, 100], 
    'learning_rate': [0.01, 0.001, 0.0001],
    'kernel_size': [3, 5, 7, 10],  
    'dropout_rate': [0, 0.1, 0.5, 0.9],    
    'filters': [32, 64, 128],
    'dense_units': [64, 100, 256],
    }

if __name__ == '__main__':
    data_path='../data/ecoli.csv'
    X_train, x_test, y_train, y_test=split_data(data_path)
    best_params=Hyperparameters_search(param_grid, X_train, y_train, cnn_model)
    