import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.svm import SVR
from tensorflow.keras.models import load_model
import joblib
from scipy.stats import spearmanr, pearsonr
from properties import calc_charge, calc_hydrophobicity, calc_hmoment
from fine_tune import finetune_model
from utils import encode

#features
AA_LIST=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

def aa_composition(sequences):
    X=[]
    for seq in sequences:
        length=len(seq)
        X.append(
            [  seq.count(aa)/length for aa in AA_LIST   ] )
    return np.array(X)

def phy_features(sequences):
    charge = calc_charge(sequences)
    hydrophobicity = calc_hydrophobicity(sequences)
    hmoment = calc_hmoment(sequences)

    return np.column_stack([charge, hydrophobicity, hmoment])   

#CV params
def spearman_score( y_true,y_pred):
    return spearmanr(y_true, y_pred)[0]
    
spearman_scorer = make_scorer(spearman_score,greater_is_better=True)

def train_ridge(X,y):
    pipe=Pipeline([('scale',StandardScaler()),('model',Ridge()) ])
    params={'model__alpha': [0.001,0.01, 0.1, 1,10, 100 ] }
    grid=GridSearchCV(pipe, params,cv=5,scoring=spearman_scorer)
    grid.fit(X,y)
    print('Best params:', grid.best_params_)
    return grid.best_estimator_

def train_rbf_svr(X,y):
    pipe=Pipeline([('scale',StandardScaler()),('model',SVR(kernel='rbf'))])
    params={'model__C':[0.1,1,10,100],
            'model__gamma':['scale',0.01,0.1,1],
            'model__epsilon':[0.01,0.05,0.1]}
    grid=GridSearchCV(pipe,params,cv=5,scoring=spearman_scorer,n_jobs=1)
    grid.fit(X,y)
    print('Best params:',grid.best_params_)
    return grid.best_estimator_
 
def calculate_metrics(y_true,y_pred):
    spearman = spearmanr( y_true,y_pred )[0]
    pearson = pearsonr(y_true,y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred) )
    return {'Spearman': spearman,'Pearson': pearson,'RMSE': rmse}

def evaluate_models(y_true,predictions):
    results=[]
    for model_name,pred in predictions.items():
        metric=calculate_metrics(y_true,pred )
        metric['Model']=model_name
        results.append(metric)
    return pd.DataFrame(results)

def run_cv(sequences, y, pretrained):
    kf=KFold(n_splits=5,shuffle=True,)
    predictions={'AA_Ridge':np.zeros(len(y)), 'PHY_rbf':np.zeros(len(y)), 'CNN': np.zeros(len(y)),}
    
    Xaa=aa_composition(sequences)
    Xphy=phy_features(sequences)
    Xcnn=encode(sequences)
    
    fold_results = []
    fold_no = 1
    for train,test in kf.split(y):
        print(f'\n========== Fold {fold_no} ==========' )
        # AA Ridge
        ridge=train_ridge( Xaa[train],  y[train] )
        predictions['AA_Ridge'][test]=ridge.predict(Xaa[test] )
        # PHY RBF-SVR
        PHY_rbf=train_rbf_svr(Xphy[train],y[train])
        predictions['PHY_rbf'][test]=PHY_rbf.predict(Xphy[test] ) 
        # CNN
        cnn=finetune_model(pretrained,Xcnn[train],y[train])
        predictions['CNN'][test]=cnn.predict(Xcnn[test], verbose=0 ).flatten()

        current_fold_predictions = {
        'AA_Ridge': predictions['AA_Ridge'][test], 
        'PHY_rbf': predictions['PHY_rbf'][test], 
        'CNN': predictions['CNN'][test],}

        current_fold_results = []
        for model_name, y_pred in current_fold_predictions.items():
            metrics = calculate_metrics( y[test],y_pred)
            row = {'Fold': fold_no, 'Model': model_name,**metrics}
            fold_results.append(row)
            current_fold_results.append(row)
            fold_df = pd.DataFrame(current_fold_results )
        print( fold_df[ [ 'Model','Spearman',   'Pearson', 'RMSE' ] ].round(4) .to_string(index=False) )
        fold_no += 1
     # 5-fold Mean ± SD
    fold_results_df = pd.DataFrame(fold_results)
    metric_cols = ['Spearman', 'Pearson','RMSE' ]
    summary_stats = ( fold_results_df .groupby('Model')[metric_cols].agg(['mean', 'std']) )
    summary_mean_sd = pd.DataFrame(index=summary_stats.index )
    for metric in metric_cols:
        summary_mean_sd[metric] = (summary_stats[(metric, 'mean')] .map(lambda x: f'{x:.4f}')+ ' ± '+ summary_stats[(metric, 'std')].map(lambda x: f'{x:.4f}'))
    print('\n========== 5-Fold Performance (Mean ± SD) ==========' )
    print(summary_mean_sd .reset_index().to_string(index=False) )       
    return predictions   
    
#Final model
def AA_ridge(train_sequences,y):
    X_train=aa_composition(train_sequences)
    model=Pipeline([('scale',StandardScaler()), ('model',  Ridge(alpha=100)) ] )
    model.fit(X_train,y )
    return model

def PHY_rbf_svr(train_sequences,y):
    X_train=phy_features(train_sequences )
    model=Pipeline([('scale',StandardScaler()),('model',SVR(kernel='rbf', C=0.1, epsilon=0.01, gamma=0.01))])
    model.fit(X_train,y )
    return model

if __name__ == '__main__':
    #data
    train=pd.read_csv('../data/SM.csv')
    sequences=train['sequence'].tolist()
    y=train['value'].values
    
    #CV for best params
    pretrained_model_path='../model/Pretrained.h5'
    predictions=run_cv(sequences, y, pretrained_model_path)
    cv_results=evaluate_models(y, predictions)
    print('\n========== Overall OOF Performance ==========')
    print(cv_results.round(3).to_string(index=False))
    
    #train
    aa_model=AA_ridge(sequences,y)
    phy_model=PHY_rbf_svr(sequences,y)
    models={'AA_ridge': aa_model,'PHY_rbf': phy_model,}
    #joblib.dump(models, 'models.joblib')
    
    #predict
    #models=joblib.load('models.joblib')
    candidate_df=pd.read_csv('../data/output/SM_gen.csv')
    candidate_df=candidate_df.drop_duplicates(subset=['sequence'], keep='first')
    candidate_seqs=candidate_df['sequence'].tolist() 
    
    cnn_model = load_model('../model/SM.h5', compile=False)
    X_candidate_cnn=encode(candidate_seqs)
    candidate_df['SM_score']=cnn_model.predict(X_candidate_cnn).flatten()
    
    X_candidate_AA=aa_composition(candidate_seqs)
    candidate_df['AA_score']=aa_model.predict(X_candidate_AA)
    
    X_candidate_PHY=phy_features(candidate_seqs)
    candidate_df['PHY_score']=phy_model.predict(X_candidate_PHY)
    candidate_df.to_csv('../data/output/SM_gen_scores.csv', index=False)