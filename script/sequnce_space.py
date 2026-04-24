import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import config, split_data
from Bio import Align
from tqdm import tqdm

plt.rcParams.update(config)

def analyze_model(model_path, X_data, y_data, model_name, layer_index=-2):
    print(f"\n Model loading: {model_path} ...")
    try:
        original_model = load_model(model_path, compile=False)
    except Exception as e:
        print(f"Model load failed: {e}")
        return
        
#     original_model.summary()
    target_layer = original_model.layers[layer_index]
    print(f"Selected feature layer: {target_layer.name} (Output Shape: {target_layer.output_shape})")
    feature_extractor = Model(inputs=original_model.input, outputs=target_layer.output)
    features = feature_extractor.predict(X_data, batch_size=32, verbose=1)
    print(f"Features shape: {features.shape}")
    print("Dimensionality reduction is underway (this might take some time)...")
    reducer = TSNE(n_components=2, perplexity=30, learning_rate='auto')
    features_2d = reducer.fit_transform(features)
        
    plt.figure(figsize=(10, 8))
    scatter=plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y_data, vmin=1.5，vmax=4, cmap='bwr',s=20, alpha=0.5, linewidth=0)
    # cbar = plt.colorbar(scatter)
    # cbar.ax.tick_params(labelsize=28) 
    # cbar.set_label('Log MIC', rotation=90, labelpad=20, fontsize=32)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()
    #plt.savefig(f'seq_space_{model_name}.svg', dpi=300, bbox_inches='tight')
    plt.show()
    
def visualize_similarity(sequences, labels):
    """
    Calculate the normalized similarity matrix based on Smith-Waterman.
    """
    n_samples = len(sequences)
    
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.1

    self_scores = np.array([aligner.score(s, s) for s in sequences])

    similarity_matrix = np.zeros((n_samples, n_samples))

    for i in tqdm(range(n_samples), desc="Calculating Matrix"):
        for j in range(i, n_samples):
            score = aligner.score(sequences[i], sequences[j])         
            # normalized similarity matrix = Score / sqrt(Self_i * Self_j)
            denom = np.sqrt(self_scores[i] * self_scores[j])
            norm_score = score / denom if denom > 0 else 0
            
            similarity_matrix[i, j] = norm_score
            similarity_matrix[j, i] = norm_score          

    reducer = TSNE(n_components=2, perplexity=30, metric='euclidean', init='pca')  
    embedding = reducer.fit_transform(similarity_matrix)
    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        mask = labels == label#True, Fakse
        plt.scatter(embedding[mask, 0], embedding[mask, 1], c=[color_map[label]], label=label, s=20, alpha=0.5, linewidth=0)
    plt.legend(loc='lower left', edgecolor='white', framealpha=0, labelspacing=0.1, markerscale=2, handletextpad=0.1, handlelength=0.8)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tick_params(axis='both',width=1.5, labelsize=28)
    plt.show()
    
if __name__ == '__main__':
    #model network layer
    X_train, x_test, y_train, y_test = split_data('../data/ecoli.csv')
    model_name='Base'
    model_path = f'../model/{model_name}.h5'
    model = load_model(model_path, compile=False)
    y_pred=model.predict(x_test)
    analyze_model(model_path, x_test, y_pred, model_name, layer_index=-2)
    
    #sequence space visualization using similarity matrix
    AMP= pd.read_csv('ecoli_df.csv',index_col=False)
    nonAMP= pd.read_csv('nonAMP.csv',index_col=False)
    AMP['label'] = 'Natural AMP'
    nonAMP['label'] = 'nonAMP'
    df = pd.concat([AMP, nonAMP], ignore_index=True)
    labels = df['label'].values
    sequences = df['sequence'].tolist()
    visualize_similarity(sequences, labels)