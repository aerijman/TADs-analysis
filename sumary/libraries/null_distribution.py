import os,sys,re
import numpy as np
import pandas as pd
import warnings
#warnings.filterwarnings('ignore')
#sys.path.append(os.path.abspath('../../new/'))
from summary_utils import *
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

myHOME = os.path.abspath('..')

start, end = "\033[1m", "\033[0;0m" # this is to print bold

def generate_split_index(data):
    '''
        function generate indices for train, test, validation split
        input: data
        output: _train, _test, _valid are indices of data, splitted 80,10,10 
    '''
    idx = np.arange(data.shape[0])
    np.random.seed=0
    np.random.shuffle(idx)

    _10 = len(idx)
    _8, _9 = int(0.8*_10), int(0.9*_10)

    _train = data.index[idx[:_8]]
    _test = data.index[idx[_8:_9]]
    _valid = data.index[idx[_9:]]
    
    return _train, _test, _valid


def calculate_charge(seq):
    '''
        calculates the charge/residue of the sequence
    '''
    charge = seq.count('K') + seq.count('R') + seq.count('H') * 0.5 \
             - seq.count('D') - seq.count('E')
            
    return (charge/len(seq))


def ohe_single(sequence, ss, **kwargs): #kwargs are aa, ss_psipred_set):
    '''
        ohe a single protein
        input: protein sequence and secondary structure (psipred format). list of aa and set of ss
               AS USED WITH THE TRAINING DATA!!!
        output: one hot encoded data of the protein as input for the neural network
    '''
    # if aa and ss_psipred lists where not provided, here are defined  
    if 'aa_list' not in kwargs.keys(): 
        print('using local aa_list')
        aa_list = ['R','H','K','D','E','S','T','N','Q','A','V','L','I','M','F' ,'Y', 'W', 'C','G','P'] # (or just aa from predictions_library)
    else:
        aa_list = kwargs['aa_list']
    
    if 'ss_list' not in kwargs.keys():
        print('using local ss_list')
        ss_list = ['E', 'H', '-']
    else:
        ss_list = kwargs['ss_list']
    
    # translate ss into ohe
    categorical_ss = np.zeros(shape=(len(ss),3))
    # go over each position
    for n,i in enumerate(ss):
        # fill 1 at the position that matches the index in ss_psipred_set
        position = ss_list.index(i)
        categorical_ss[n, position] = 1
    
    # translate sequence into ohe
    categorical_seq = np.zeros(shape=(len(sequence),20))
    # go over each positoin
    for n,i in enumerate(sequence):
        # fill 1 at position that matches index in aa
        position = aa.index(i)
        categorical_seq[n, position] = 1
    
    # return merged matrix  
    return np.hstack([categorical_seq, categorical_ss]).reshape(1, len(ss), 23, 1)


def translate_dna2aa(dna):
    '''
        function translate nucleotide sequence into aminoacid sequence
        input: dna sequence as a string
        output: aa sequence as a string
    '''
    # codons list
    gene_code = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'}
    
    codon_seq = [dna[i:i+3] for i in np.arange(0,len(dna),3)]
    aa_seq = [gene_code[i] for i in codon_seq]
    
    return ''.join(aa_seq)


def read_wig(wig):
    '''
        function read wiggle file
        input: wiggle file
        output: tuple with position as key and counts as values 
    '''
    positions, counts = [],[]
    f = open(wig)
    try:
        while True:
            line = next(f)
            _, position, count = line.strip('\n').split('\t')
            positions.append(position)
            counts.append(count)

    except StopIteration:
        print('done! ' + wig + ' file loaded')
    
    return pd.DataFrame(np.array(counts, 'float'), index=np.array(positions, 'int'), columns=['counts'])


def make_m_dict(Stark_data_annot, gfp_p, gfp_m, FOLD_CHANGE_THRESHOLD=2, MINIMUM_READS=100):
    '''
        function generate a table of measurements from Stark's (or other) data
        INPUT: Stark_data_annot (including gene name, start and end positions in gfp_p and gfp_m)
               gfp_p: wig file with positions and counts for sorting GFP positive
               gfp_m: wig file with positions and counts for sorting GFP negative
               FOLD_CHANGE_THRESHOLD: minimal fold change gfp_p/gfp_m accepted to generate data
               MINIMUM_READS: minimum absolute counts in gfp_p accepted to generate data
        OUTPUT: table (key: gene name 
                       values: counts over the length of each protein)
    '''
    k,v=[],[]
    for TF_name in Stark_data_annot.index:

        # extract gene values from plus and minus sets 
        start, end = Stark_data_annot.loc[TF_name, ['start','end']]
        plus = gfp_p[start:end]
        minus = gfp_m[start:end]

        # averaged values every 3 nt to plot together with protein 
        plus = np.array([np.mean(plus[i:i+3]) for i in np.arange(0, len(plus), 3)])
        minus = np.array([np.mean(minus[i:i+3]) for i in np.arange(0, len(minus), 3)]) 
        minus += 10e-4 # pseudocount to avoid dividing by zero

        # take values of tADs whose plus/minus >fold_change_threshold and plus >minimum_reads counts
        tAD = np.nan_to_num(plus/minus)
        tAD = np.array([k if k>FOLD_CHANGE_THRESHOLD and i>MINIMUM_READS else 0 for i,k in zip(plus, tAD)])
        tAD = np.nan_to_num((tAD - np.min(tAD)) / (np.max(tAD) - np.min(tAD) + 10e-4)) # added 10e-4 to avoid division by 0

        k.append(TF_name)
        v.append(tAD)

    # finally define m_dict, a dictionary for m
    m_dict = dict(zip(k,v))

    return m_dict


def make_p_dict(Stark_data_annot, deep_model, fastas_folder='../data/Stark_data/prediction_files/', 
                horiz_folder='../data/Stark_data/prediction_files/'):
    ''' 
    function generate table (dictionary) of key=f, values=predictions from best NN model
    INPUT: Stark_data_annot (I could use f instead, but I keep this to make it equal to make_m_dict)
           fastas_folder directory  
           horiz_folder directory
    OUTPUT:table (keys=gene names, values=prediction scores over the length of each protein)
    '''
    k,v = [],[]
    for TF_name in Stark_data_annot.index:

        # if ss file is not here yet, skip
        if not os.path.exists(horiz_folder+ TF_name + '.horiz'): continue

        # open fasta and horiz files and generate ohe
        seq = read_fasta(fastas_folder+ TF_name + '.fasta')
        ss = read_horiz(horiz_folder+ TF_name + '.horiz')
        single = prepare_ohe([seq,ss]) #ohe_single(seq,ss, aa_list=aa, ss_list=ss_psipred_set)

        # predict using deep_model
        predictions = []
        for i in range(0, len(ss)-30):
            region = deep_model.predict(single[i:i+30].reshape(1,30,23,1))[0][0]
            predictions.append(region)    

        k.append(TF_name)
        v.append(np.array(predictions))

    # finally define p_dict, a dictionary for p
    p_dict = dict(zip(k,v))
    
    return p_dict


def build_distribution(f, p_dict, m, **kwargs): #n_iter=10000, corr_type='spearman'):
    '''
        function build distribution of correlation coefficients of two vectors
        as one of them is permutated in each iteration.
        INPUT: f (index of genes from table)
               p_dict (table. keys=gene names (as in f), values=(counts over the length of the protein))
               m = vector of concatenated counts from m_dict
               n_iter = number of iterations (each iteration shuffles the values and computes correlation)
               corr_type = type of correlation to compute
        OUTPUT: list of correlation coefficients.
    '''    
    # obtain arguments
    if 'n_iter' in kwargs.keys(): n_iter=kwargs['n_iter']
    else: n_iter = 10000
    
    if 'corr_type' in kwargs.keys(): 
        if kwargs['corr_type'] in ['pearson', 'R', 2]: corr_type=2
        else: 
            corr_type=1 # which corresponds to spearmanr
    else: 
        corr_type=1        
        
    if corr_type==1:
        # what are the boundaries of the windows to compute spearman correlation?
        windows = [len(i) for i in p_dict.values()]
        windows = np.insert(np.cumsum(windows),0,0)
        windows = [(windows[i],windows[i+1]) for i in range(len(windows)-1)]
        
        # To speed-up the process, I will rank the m vector and keep the index of the rank
        # this way I don't have to compute the spearman on each loop and can make the method 
        # x10 faster
        m_index = np.concatenate([np.argsort(m[i:j])+i for i,j in windows])
    
    # build distribution of correlations
    corr_distrib = np.zeros(n_iter)
    for n in range(n_iter):

        # permutation of p
        k = np.random.permutation(f[:])
        p_permut = np.concatenate([p_dict[i] for i in k])

        # compute correlation
        if corr_type==2:
            corr_distrib[n] = np.corrcoef(m, p_permut)[0][1]
        else:
            # compute spearmanr per window
            #corrs = np.nan_to_num([spearmanr(m[i:j], p_permut[i:j])[0] for i,j in windows])
            p_index = np.concatenate([np.argsort(p_permut[i:j])+i for i,j in windows])
            #corr = np.corrcoef(m_index, p_index)[0][1]   
            #corr_distrib.append(spearmanr(m, p_permut)[0])
            corrs = np.nan_to_num([np.corrcoef(m[i:j], p_permut[i:j])[0][1] for i,j in windows])
            corr_distrib[n] = np.mean(corrs)
            
    return corr_distrib




def calc_typeI_error(corr_values, point, **kwargs):
    '''
        function makes distribution out of correlation values (corr_values)
        and calculate the area under "normal" curve as extreme or more extreme than point,
        that is the probability that a number falls on the right of the point in that curve(
        if point is positive, else to the left of the point).
        INPUT = corr_values (list of values of correlation) and point (single sample to test)
        optional = bins (bins=100) and plot (plot=True)
        OUTPUT = type I error or area under the curve from point to the right or 
                 point to the left if point is negative value.
    '''
    # allow user to input bins number
    if 'bins' in kwargs.keys(): bins=kwargs['bins']
    else: bins=100
    
    # make histogram of corr_values
    y,x = np.histogram(corr_values, bins=bins)
    
    # have to make x.shape=y.shape
    x = x[:-1]
    
    # allow user to plot the distribution
    if 'plot' in kwargs.keys():
        if kwargs['plot']==True:
            yc = np.convolve(y, np.ones(10)/10, "same")
            if point>0:
                _right = np.where(x>=point)
                _left = np.where(x<point)
                plt.plot(x, yc, c="k")
                plt.fill_between(x[_right], yc[_right], color='g', alpha=0.4)
                plt.fill_between(x[_left], yc[_left], color='b', alpha=0.3)
            else:
                _left = np.where(x<=point)
                _right = np.where(x>point)                
                plt.plot(x, yc, c="k")
                plt.fill_between(x[_right], yc[_right], color='b', alpha=0.3)
                plt.fill_between(x[_left], yc[_left], color='g', alpha=0.4)
    
    # get coordinates of "values as or more extreme than point"
    if point<0:
        idx = np.where(x<=point)
    else:
        idx = np.where(x>=point)

    # measure total area and area from point to it's right
    total_area = np.sum(y)
    area_type_I_error = np.sum(y[idx])
    
    # typeI error
    probaTypeI = area_type_I_error *1. / total_area            
            
    return probaTypeI


# create matrix to store results. This matrix should contain for each position, all possible 20 aa with their 
# corresponding tAD probabilities. Dims = [seq_position, aa.index] = predictions
def compute_mutagenesis(ohe_data, refId, deep_model):
    '''
        function prodices predictions on original sequence and all posible 20 mutations/position
        for all positions
        INPUT: ohe_data. data corresponds to the sequence of the protein and it's secondary structure
               refId. index reference to look for a particular sequence within an array of sequences (ohe)
               deep_model. deep_learning model
        OUTPUT: prediction for the original sequence (sequence must be 30aa long).
                results. matrix containing TAD probabilities of all differente mutants --> arr[seq_position, aa.index] = predictions                         
     '''
    all_samples = ohe_data.shape[0]
    #results = np.zeros(shape=(all_samples,30,20))
    results = np.zeros(shape=(30,20))

    # go over all samples and mutate every position in a ohe setting.
    #sample = ohe_data[bestIdx[0]]#ohe_data[10].copy()
    sample = ohe_data[refId]

    # first of all measure tAD probaboility in the original sequence
    prediction = deep_model.predict(sample.reshape(np.append(1, sample.shape)))
    print('original_prediction: {}'.format(prediction[0][0]))
    # list of amino acids in ohe format
    original_seq = np.where(sample[:,:20,0]==1)[1] #ohe_data[0,:,:20,0]==1)[1]

    # start filling results with original_seq
    for n in range(30):
        results[n,original_seq[n]]=prediction[0][0]

    # go over all positions in the sequence
    for position in range(30):  #len(original_seq)
        # ohe_aminoacid in current position
        original_position = original_seq[position]
        # list all possible ohe aminoacids
        ohe_positions = list(np.arange(20))
        # remove the original aa from the list
        ohe_positions.remove(original_position)

        # copy into new instance to avoid overwriting and to restore all other positions to their 
        # original values
        this_sample = sample.copy()
        # start mutation in that position
        for mutation in ohe_positions:
            # reset all aa in position to 0
            this_sample[position,:20,0]=0
            # make mutation
            this_sample[position,mutation,0]=1
            # predict the mutant tAD probability
            tmp = deep_model.predict(this_sample.reshape(np.append(1, sample.shape)))
            
            # If there is a radical change, let me know
            if prediction > 0.5 and tmp[0][0] < 0.5 or prediction < 0.5 and tmp[0][0] > 0.5:
                # print the original sequence
                SEQUENCE = ''.join([aa[S] for S in original_seq])
                print(SEQUENCE)
                # print which mutation at which position
                print('{}/{} at position {} -> score {} into {}'.format(aa[original_position], aa[mutation], position, prediction, tmp[0][0]))
                #print(position, aa.index(mutation))
                
            #results[sample_number, position, mutation] = prediction
            #result.append([aa[mutation], aa[original_position]])
            results[position,mutation]=tmp[0][0]
            
    return prediction, results


def read_bed(bed, length):
    '''
        function read bed or stack wiggle file
        input: bed file
               length of the single chromosomal genome.
        output: numpy array with position as key and counts as values 
    '''
    # this array will contain all data 
    genome = np.zeros(length, dtype='float')
    f = open(bed)
    try:
        while True:
            line = next(f)
            
            # avoid header if exists
            if line[0]=='#': continue
            
            # read and fill genome array
            _, position1, position2, count = line.strip('\n').split('\t')
            x1, x2 = int(position1), int(position2)
            genome[x1:x2] = float(count)
            
    except StopIteration:
        print('done! ' + bed + ' file loaded')
    
    return genome