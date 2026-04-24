import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from utils import config, split_data

plt.rcParams.update(config)

def Actual_Predicted(y_test, y_pred, model_name):
    R2=r2_score(y_test, y_pred)
    fig, ax=plt.subplots(figsize=(10,10))
    plt.scatter(y_test,y_pred,color='pink')
    plt.plot([-2,5],[-2,5],'k--')
    plt.plot([-2,4],[-1,5],'k--')
    plt.plot([-1,5],[-2,4],'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0,4.5,f"R$^2$={R2:.3f}")
    plt.ylim(-2,5)
    plt.xlim(-2,5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    plt.tight_layout()
    #plt.savefig(f'Actual_Predicted_{model_name}.svg', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':    
    X_train, x_test, y_train, y_test=split_data('../data/ecoli.csv')
    model_name='Base'
    model_path = f'../model/{model_name}.h5'
    model = load_model(model_path, compile=False)
    y_pred=model.predict(x_test)
    Actual_Predicted(y_test, y_pred, model_name)