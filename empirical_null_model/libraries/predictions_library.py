import pandas as pd
import numpy as np
import re,os
import matplotlib.pyplot as plt


#####################################  MISC. FUNCTIONS ##################################
# list of one-letter aa
aa = ['R','H','K','D','E','S','T','N','Q','A','V','L','I','M','F' ,'Y', 'W', 'C','G','P']

'''Split data into Train, Test and Validation sets with ratio 8:1:1'''
def split_data(idx=None, data=None, selection=aa, column_label_score_NAME='scores', seed=42):
  
    train, test = (np.array([0.8, 0.1]) * data.shape[0]).astype(int)
    
    X, y = data[selection].values, data[column_label_score_NAME].values    
    X, y = X[idx], y[idx]
    
    Xtrain, ytrain = X[:train], y[:train]
    Xtest, ytest = X[train:-test], y[train:-test]
    Xvalid, yvalid = X[-test:], y[-test:] 
   
    return(Xtrain, ytrain, Xtest, ytest, Xvalid, yvalid)

'''given X and y it outputs an array of thresholds, False positive rate and True positive rate'''
def roc_curve(X,y):
    # make sure X and y are numpy arrays
    X,y = np.array(X), np.array(y)
    # thresholds to use for defining fpr and tpr
    scale = np.linspace(X.min(), X.max(), 50)
    # define fpr and tpr functions to use with different thresholds
    def tpr(y_hat, y):
        a,b = set(np.hstack(np.where(y==1))), set(np.hstack(np.where(y_hat==1)))
        return(len(a & b) / len(a))
    def fpr(y_hat, y):
        a,b = set(np.hstack(np.where(y==0))), set(np.hstack(np.where(y_hat==1)))
        return(len(a & b) / len(a))
    # Core of the function where TPR and FPR are calculated
    TPR, FPR = np.zeros(50), np.zeros(50)
    for n,s in enumerate(scale):
        # threshold of variable defines what is + and -
        y_hat = np.array([1 if x>=s else 0 for x in X])
        TPR[n] = tpr(y_hat, y)
        FPR[n] = fpr(y_hat, y)
    # the resulting array will contain thresholds, fpr and tpr
    roc = np.vstack([scale, FPR, TPR])
    return(roc)

# I have this function to compare upon any doubt but I will be using the sklearn implementation 
from sklearn.metrics import roc_auc_score

''' Open the new lukasz's files with sequences as index and two columns, scores and normalized scores'''
def open_fastas(ff):
    vals, keys = [], []
    fasta = open(ff)
    while True:
        try:
            header, sequence = next(fasta).strip(), next(fasta).strip()
            _, bg, b1, b2, b3, b4, score, norm_score = header.split("_")
            vals.append([score, norm_score])
            keys.append(sequence)
        except StopIteration: break
    df = pd.DataFrame(vals, index=keys, columns=['scores','norm_scores'])
    df = df.astype(float)
    return df

'''one hot encooding of data'''
def one_hot_encode(seq):
    X = np.zeros(shape=(len(seq),len(aa)))
    try:
        for n,i in enumerate(seq): X[n, aa.index(i)] = 1
        return X
    except exception as e:
        print('exception {} encountered'.format(e))
        
#################################### NEWER FUNCTIONS ########################################

''' Load psipred and iupred predictions '''
def load_predicted_properties(fileName):
    k, v = [],[]
    with open(fileName,"r") as f:
        while 1:
            try:
                _1, sequence, prediction, _2 = next(f), next(f), next(f), next(f)
                k.append(sequence.strip())
                v.append([i for i in prediction.strip()])
            except StopIteration:
                print(fileName+' loaded succesfully!')
                break
    df = pd.DataFrame(v, index=k)
    return(df)

# load netsurf data
def load_netsurf(fileName):
    k,v1,v2 = [],[],[]
    with open(fileName, "r") as f:
        while 1:
            try:
                _1, seq, acc, _2, ss, _3 = next(f), next(f), next(f), next(f), next(f), next(f)
                k.append(seq.strip())
                v1.append([i for i in acc.strip()])
                v2.append([i for i in ss.strip()])
            except StopIteration:
                print(fileName+'load succesfully!')
                break
    acc = pd.DataFrame(v1, index=k)
    ss =pd.DataFrame(v2, index=k)    
    return ss, acc

# one hot encode the data
def ohe(seq, property_dict):
    tmp = np.zeros(shape=(len(seq), len(property_dict)))
    for n,i in enumerate(seq):
        tmp[n, property_dict.index(i)] = 1
    return tmp

################################### PREDICTOR FUNCTIONS #######################################

def predict(model, sequence):
    # apend the 1st 15 residues to the "N" terminal and the last 15 residues to the "C" terminal
    sequence = sequence[:15] + sequence + sequence[-15:]
    # declare array of probabilities of being TAD per aminoacid
    probas = []
    # loop sliding over each position and taking the 30 aa (position is the 15t)
    for i in range(len(sequence)-30):
        test_window = sequence[i:i+30]
        ohe_test_window = ohe(test_window, aa).reshape(1,600)
        probas.append( model.predict_proba(ohe_test_window)[0][0] )
    return np.array(probas)

def print_prediction(probabilities, sequence, wide=120):
    prediction = ''.join(['*' if i>0.8 else ' ' for i in probabilities])

    while len(sequence)>0:
        print(prediction[:wide]+'\n'+sequence[:wide])
        prediction = prediction[wide:]
        sequence = sequence[wide:]

def plot_prediction(probabilities):
    filtered = np.array([i if i >0.8 else 0 for i in probabilities])
    filtered = np.convolve(filtered, np.ones(15)/15, "same")
    #plt.subplot(121); 
    plt.plot(filtered); plt.title('filtered'); plt.ylim(0.1)
    plt.xlabel('position'); plt.ylabel('TAD probability'); 
    #plt.subplot(122); plt.plot(probabilities); plt.title('original'); plt.ylim(0.1)
    
def protein_from_sgd(name): 
    sequence = []
    found = False
    for i in open('../from_scratch/orf_trans.fasta'):
        if i[0]=='>':
            if found and i[0]=='>': break
            if re.search(name,i[:20]) and not found: found = True
        if found and i[0] != '>': sequence.append(i.strip().strip('*'))
    return ''.join(sequence)