import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from utils import config, encode, AMP_test_performance

plt.rcParams.update(config)

def Actual_Predicted(spear, pear, y_test, y_pred, model_name):
    y_test_1d = np.asarray(y_test).ravel()
    y_pred_1d = np.asarray(y_pred).ravel()
    k, b = np.polyfit(y_test_1d, y_pred_1d, 1)
    
    fig, ax=plt.subplots(figsize=(8,8))
    plt.scatter(y_test,y_pred,color='pink')
    x_line = np.linspace(np.min(y_test_1d), np.max(y_test_1d), 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='black', )
    plt.xlabel('Reference activity label')
    plt.ylabel('Model output')
    plt.yticks(font="Arial")
    plt.text(-1.5, 3.5, f'{model_name}')
    plt.text(-1.5,3,f"Spearman $\it{{ρ}}$ = {spear:.3f}")
    plt.text(-1.5,2.5,f"Pearson $\it{{r}}$ = {pear[0]:.3f}")
    plt.ylim(-2,4)
    plt.xlim(-2,4)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':    
    model_name='Pretrained'
    model_path = f'../model/{model_name}.h5'
    model = load_model(model_path, compile=False)
    spear, pear, y_test, y_pred = AMP_test_performance(model)

    Actual_Predicted(spear, pear, y_test, y_pred, model_name)