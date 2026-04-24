## Requirements
biopython>=1.78
matplotlib==3.5.1
modlamp==4.3.2
numpy==1.21.2
pandas==1.3.4
python==3.7.11
scikit-learn==1.0.1
scipy==1.7.1
tensorflow==2.5.0

## Environment
Use the provided file to set up a new virtual environment with conda.
Run pip install -r requirements.txt

## Data
The `data` folder contains the data used for model training. In each file, the "sequence" column represents the peptide sequence composed of standard amino acids, and the "value" column represents the log10 transformation of its MIC antibacterial activity.

The GRAMPA.csv is a database of peptides and their antimicrobial activity against various bacteria. The "TRUE" or "FALSE" in the "has_unusual_modification" column indicate whether the sequence has any modifications other than the C-terminal amidation.

The ecoli.csv is extracted from grampa.csv and contains the antimicrobial peptide sequences against E. coli with only C-terminal amidation, along with their activities.

The nonAMP.csv is obtained from the UniProt database. The sequences have no antibacterial activity and labeled a maximum activity value of 4.

Merge the two datasets of ecoli.csv and nonAMP.csv to create train.csv,which will be used for model training. 

The SM.csv and RM.csv are the variants of CecA produced by Saturated Mutation or Random Mutation that are used for model fine-tuning.

## Script
`utils.py` defines variables and constants, and provides sequences encoding and data processing functions that can be input into the model. These are imported and used in other scripts.

`model.py` defines the architecture of the neural network model and provides an example for running. The 'hyperparameters.py' script outputs the optimal parameter combination, which can be used to replace the default parameters.

`hyperparameters.py` uses cross-validation to search for the best parameter combination. The parameters to be optimized can be easily changed in the 'param_grid' dictionary.

`fine_tune.py` uses SM.csv and RM.csv to fine-tune the base model and saves the final model in the 'model' folder. Simply change the name of the csv file being read, and you can use different data to fine-tune the model.

`sequence_generation.py` generates new AMPs using simulated annealing and utilizes the saved model in the 'model' folder to make MIC predictions. The model's sensitivity to low MICs will affect the convergence area of the new sequence. Run 'python script/sequence_generation.py' to output the new sequences and their predicted log MIC.     

`R2score.py` plots scatter plots of the actual vs. predicted activity of AMP sequences for different models, calculates and labels the R2 value.

`sequence_space.py` uses t-SNE to reduce the dimensionality, visualizes the neural network of the model and plots the latent sequence space based on the similarity matrix of sequences. To accelerate the sequence similarity calculation, the Bio.Align module needs to be loaded. Therefore, make sure that the version of biopython is >=1.78.

`properties.py` uses lists of sequences, and with the help of modlamp and biopython to calculate and plotthe the physicochemical properties graphs.  

## Model
To retrain models, run 'python script/fine_tune.py' from the root directory.
The `model` dir provides a pre-trained model that can be used to predict MICs and generate new AMPs by run 'python script/sequence_generation.py'. 
