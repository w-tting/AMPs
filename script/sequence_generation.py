import pandas as pd
import numpy as np
import random
from copy import deepcopy
from tensorflow.keras.models import load_model
from utils import *

def generate_random_sequence(alphabet, length_of_sequence_min=0, length_of_sequence_max=MAX_SEQUENCE_LENGTH, fixed_length=-10):
        sequence = ''
        choices =list(alphabet)
        counter = 0
        length_to_use = -10
        if fixed_length<0:
            while counter<20 and (length_to_use < length_of_sequence_min or 
                                  length_to_use > length_of_sequence_max):
                length_to_use = random.choice(seqs_lengths)
                counter += 1
        else:
            length_to_use = fixed_length
        for _ in range(length_to_use):
            sequence += random.choice(choices)
        return sequence
    
def sequence_to_vector(sequence):
    #sequence='XXX', not ['XXX']
    vector = np.zeros([MAX_SEQUENCE_LENGTH, len(character_to_index) ])
    for i, character in enumerate(sequence[:MAX_SEQUENCE_LENGTH]):
        vector[i][character_to_index[character]] = 1
    return vector

def is_acceptable(sequence_with_padding):
    at_underscore=False
    for char in sequence_with_padding:
        if char == '_':
            at_underscore=True
        elif at_underscore:
            return False
    return True

def generate_move(old_vector,min_length=10,max_length=25):
    vector=deepcopy(old_vector)
    asc=random.random()
    peptide_length=0
    #actual length
    for i in range(len(vector)):
        if np.sum(vector[i])>0.5:
            peptide_length=i+1
            
    if asc<0.025 and peptide_length>=min_length:
        #Remove from the front 2.5% of the time
        for i in range(len(vector)-1):
            vector[i]=[k for k in vector[i+1]]
        vector[len(vector)-1]=np.zeros(num_chars)
        
    elif asc < 0.05 and peptide_length>=min_length:
        #Remove from the back 2.5% of the time
        vector[peptide_length - 1] = np.zeros(num_chars)
        
    elif asc < .075 and peptide_length<max_length:
        #Add to the front 2.5% of the time
        which = random.randint(0, num_chars - 1)
        for i in range(1,len(vector)):
            vector[-i]=[k for k in vector[-i-1]]
        vector[0]=np.zeros(num_chars)
        vector[0][which]=1
        
    elif asc < 0.1 and peptide_length<max_length:
        #Add to the back 2.5% of the time
        which = random.randint(0, num_chars - 1)
        if peptide_length < len(vector):
            vector[peptide_length] = np.zeros(num_chars)
            vector[peptide_length][which] = 1
            
    else:
        #Swap something in the middle
        which_index=random.randint(0,peptide_length-1)
        which_residue=random.randint(0,num_chars - 1)
        blah=0
 
        try:
            vector[which_index]=np.zeros(num_chars)
            vector[which_index][which_residue]=1
        except:
            print ('Trying to change index '+repr(which_index)+' but array is only of length '+repr(len(vector)))

    temp_seq = ''
    for v in vector:
        if np.sum(v) > 0.5: 
            temp_seq += 'X'
        else: temp_seq += '_'
    if not is_acceptable(temp_seq):
        return old_vector
    
    return vector

def accept_move(mic_old,mic_new,temp):
    if mic_new<mic_old:
        return True
    return random.random()<np.exp((mic_old-mic_new)/temp)

def vector_to_sequence(vector):
    sequence = ''
    for v in vector:
        nonzeros = np.argwhere(v[:len(character_to_index)])
        if len(nonzeros) > 1:
            print("?????")
        elif len(nonzeros) == 0:
            sequence += '_'
        else:
            sequence += index_to_character[np.argwhere(v)[0][0]]
    return sequence

def generate_sequence_by_simulated_annealing(model,cdict=CHARACTER_DICT, min_seq_length=10, max_seq_length=25, cooling_schedule = 'Power', nsteps=100000, t0=MAX_MIC/np.log(2), tf=0.00001/np.log(2)):
    s = generate_random_sequence(cdict, length_of_sequence_min=min_seq_length, length_of_sequence_max=max_seq_length)
    v = sequence_to_vector(s)

    print ('Starting sequence: '+repr(s))

    temp = t0
    scale=np.power(tf/t0,1./nsteps)
    for i in range(nsteps):
        move = generate_move(v, min_length=min_seq_length, max_length=max_seq_length)
        old_and_new = model.predict(np.array([v,move]))
        if accept_move(old_and_new[0],old_and_new[1],temp):
            v = move
        if cooling_schedule =='Power':
            temp=temp*scale
        elif cooling_schedule == 'Linear':
            temp = temp + (tf - t0)/nsteps
        else:
            print ('Cooling schedule not recognized: '+cooling_schedule)
            break
    return vector_to_sequence(v), model.predict(np.array([v]))[0]

def run_to_row(sequence_with_underscores, pred_log_mic,):
    if '_' in sequence_with_underscores:
        sequence = sequence_with_underscores[:sequence_with_underscores.find('_')]
    else:
        sequence = sequence_with_underscores

    return {'sequence':sequence, 'value':pred_log_mic,}
    
if __name__ == '__main__':
    df=pd.read_csv('../data/ecoli.csv')
    seqs = df['sequence']
    seqs_lengths = [min(MAX_SEQUENCE_LENGTH,len(seq)) for seq in seqs]

    num_seqs=50 #The total number of the new sequence
    max_seq_length=25 #The maximum length of the new sequence
    cooling_schedule='Power'   
    model_name='Base'
    model_path = f'../model/{model_name}.h5'
    model = load_model(model_path, compile=False)
    start = time.time()
    rows=[]
    for i in range(num_seqs):
        seq, plm=generate_sequence_by_simulated_annealing(model, max_seq_length=max_seq_length, cooling_schedule=cooling_schedule)
        rows.append(run_to_row(seq, plm))  
    end=time.time()
    print ('runtime in seconds: '+repr(end-start))   
    new_df=pd.DataFrame(rows)
    new_df.sort_values(by='value', ascending=True, inplace=True)
    new_df.to_csv(f'../data/output/{model_name}_gen.csv',index=False)