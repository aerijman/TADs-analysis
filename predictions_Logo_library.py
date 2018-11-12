import os,sys,re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#sys.path.append(os.path.abspath('../../new/'))
from predictions_library import *

myHOME = os.path.abspath('..')

start, end = "\033[1m", "\033[0;0m" # this is to print bold

def initialize_notebook_and_load_datasets():
    '''
        function initialize notebook and load datasets.
        input: nothing
        output: data
    '''
    # load datasets
    positives = open_fastas(myHOME + '/data/ARG3_O1_O2_clustered_rescored.fasta_bin14_threshold10.0_37923positives.fasta')
    negatives = open_fastas(myHOME + '/data/ARG3_O1_O2_clustered_rescored.fasta_bg_37923negatives.fasta')
    positives['TAD']=1
    negatives['TAD']=0 
    data = pd.concat([pd.DataFrame(i) for i in [positives, negatives]], axis=0)

    # get aa frequencies
    for i in aa:
        data[i] = np.array([seq.count(i)/30. for seq in data.index])

    # function to extract distribution of sequence in bins
    def fasta_header_extract(fileName):
        headers, sequences = [], []
        for i in open(fileName):
            if i[0]=='>':
                tmp = i.strip().split('_')
                headers.append(tmp)
                continue
            else:
                sequences.append(i.strip())
        values = np.array(headers)[:,1:6].astype(int)
        return pd.DataFrame(values, index=sequences, columns = ['bg','bin1','bin2','bin3','bin4'])
    
    # from data, take out sequences with no reads in bins2,3 and 4 (bin distribution is extracted from the fasta headers)
    positives_bins = fasta_header_extract(myHOME + '/data/ARG3_O1_O2_clustered_rescored.fasta_bin14_threshold10.0_37923positives.fasta')
    negatives = open_fastas(myHOME + '/data/ARG3_O1_O2_clustered_rescored.fasta_bg_37923negatives.fasta')
    
    positives_out = positives_bins[(positives_bins['bin2']==0) & (positives_bins['bin3']==0) & (positives_bins['bin4']==0)].index
    data = data.loc[ set(data.index) - set(positives_out)]

    return data

    
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


def open_fasta(fasta):
    '''
        function open fasta file
        input: fasta file
        output: sequence
    '''
    sequence = []
    for i in open(fasta):
        # obviate header
        if i[0]=='>': continue
        # else take line and later join
        sequence.append(i.strip('\n'))
        
    return ''.join(sequence)

def open_ss(ss):
    '''
        function open .horiz files
        input: .horiz file
        output: sequence of ss elements
    '''
    sequence = []
    for i in open(ss):
        # jump lines that are not relevant
        if i[:4]!='Pred': continue
        sequence.append(i[6:].strip('\n'))
        
    return ''.join(sequence).replace('C','-')


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
        plus = gfp_p.iloc[start:end].counts.values
        minus = gfp_m.iloc[start:end].counts.values

        # averaged values every 3 nt to plot together with protein 
        plus = np.array([np.mean(plus[i:i+3]) for i in np.arange(0, len(plus), 3)])
        minus = np.array([np.mean(minus[i:i+3]) for i in np.arange(0, len(minus), 3)])

        # take values of tADs whose plus/minus >fold_change_threshold and plus >minimum_reads counts
        tAD = np.nan_to_num(plus/minus)
        tAD = np.array([k if k>FOLD_CHANGE_THRESHOLD and i>MINIMUM_READS else 0 for i,k in zip(plus, tAD)])
        tAD = np.nan_to_num((tAD - np.min(tAD)) / (np.max(tAD) - np.min(tAD)))

        k.append(TF_name)
        v.append(tAD)

    # finally define m_dict, a dictionary for m
    m_dict = dict(zip(k,v))

    return m_dict

def make_p_dict(Stark_data_annot, deep_model, fastas_folder='../data/Stark_tADs/fastas/', horiz_folder='../data/Stark_tADs/horiz/'):
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
        seq = open_fasta(fastas_folder+ TF_name + '.fasta')
        ss = open_ss(horiz_folder+ TF_name + '.horiz')
        single = ohe_single(seq,ss, aa_list=aa, ss_list=ss_psipred_set)

        # predict using deep_model
        predictions = []
        for i in range(0, len(ss)-30):
            region = deep_model.predict(single[0, i:i+30, :, 0].reshape(1,30,23,1))[0][0]
            predictions.append(region)    

        k.append(TF_name)
        v.append(np.array(predictions))

    # finally define p_dict, a dictionary for p
    p_dict = dict(zip(k,v))
    
    return p_dict


def build_distribution(f, p_dict, n_iter=10000):
    '''
        function build distribution of correlation coefficients of two vectors
        as one of them is permutated in each iteration.
        INPUT: f (index of genes from table)
               p_dict (table. keys=gene names (as in f), values=(counts over the length of the protein))
               m = vector of concatenated counts from m_dict
        OUTPUT: list of correlation coefficients.
    '''
    # build distribution of correlations
    corr_distrib = []
    for n in range(n_iter):

        # permutation of p
        k = f[:]
        np.random.shuffle(k)
        p_permut = np.concatenate([p_dict[i] for i in k])

        # compute correlation
        corr_distrib.append(np.corrcoef(m, p_permut)[0][1])

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
