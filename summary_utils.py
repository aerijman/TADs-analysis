import os,sys,re
import numpy as np
import pandas as pd
import sqlite3
import uuid
from functools import reduce
import itertools


# many data files are still located at data folder (actually symlink to other dir)
DATA = os.path.abspath('../data')

# list of one-letter aa and physical properties
aa = ['R','H','K','D','E','S','T','N','Q','A','V','L','I','M','F' ,'Y', 'W', 'C','G','P']
dipeptides = np.array([''.join(i) for i in itertools.product(aa, repeat=2)])
physical_props = pd.read_csv(DATA + '/physical_props.tsv', sep='\t', index_col=0)
ss = ['E','H','-'] # list of secondary structure elements


def load_data_from_sqlite(database, table):
    '''
        function extract data from a database file.
        INPUT: database filename
               table= 'positives' or 'negatives'. This argument is a string!
        OUTPUT: data array with seq and ss and labels
                length of the table
    '''
    # create connection to database
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    
    # extract positives and negatives into different variables
    cursor.execute('SELECT seq,ss FROM ' + table)
    samples = np.array( cursor.fetchall() )

    # get length of the table 
    cursor.execute('SELECT COUNT(*) FROM ' + table)
    length = cursor.fetchone()

    # return data in a table with named columns and replace the indices assigned by sqlite
    return samples, length


def store_data_numpy(arr, TYPE='int8', path='/fh/scratch/delete90/hahn_s/aerijman/temporary/'):
    '''
        function stores temporarily the data in the disk and keep a map to it in the memory
        so the data can be accessed very fasta without having to store the whole data in the memory.
        It might be more efficient to store the data into more than one file. 
        INPUT: arr (array, ideally the one hot encoding arrays)
               TYPE (indicates which type of data is to be handled (int8 default for ohe data) no char or string allowed)
        OUTPUT: temporary array to be used as if it was the original array.
                filename. A '.dat' file which is indexed and mapped by numpy. Filename is provided so the file is 
                          deleted before finishing the work.
    '''  
    # asign a random name to the file that will be stored in the disk
    PATH = os.path.abspath(path)
    filename = PATH + '/npy_tmp_' + uuid.uuid4().hex + '.dat'

    # create a file with a numpy index/pointer
    fp = np.memmap(filename, dtype=TYPE, mode='w+', shape=arr.shape)

    # fill file with data
    fp[:] = arr[:]
    
    # copy a short segment to check later if loaded data is equal to stored data 
    test1 = np.concatenate(fp[:20])
    
    # check that file was created
    if fp.filename != filename: 
        print('filename wrong...')
        return -1

    # Deletion flushes memory changes to disk before removing the object
    del fp

    # Load the memmap and verify data was stored
    newfp = np.memmap(filename, dtype=TYPE, mode='r', shape=arr.shape)
    
    # check that loaded data is equal to data prior to storing it
    test2 = np.concatenate(newfp[:20])    
    _test1 = test1.shape==test2.shape
    _test2 = np.sum([i==j for i,j in zip(test1, test2)]) / (reduce(lambda x,y:x*y, test1.shape))
    if not _test1 or not _test2:
        print('stored and loaded data are not the same...')
        return -1
        
    return newfp, filename


def subsample_negatives(list_of_arrs, sampleSize):
    '''
        Function randomly takes a sample of the size "sampleSize" (which, ideally is the size of positives)
        from all arrays (whose names are listed in name_of_arrs). These arrays are actually mapped from 
        files in disk (see function store_data_numpy).
        INPUT:  name_of_arrs = name of all arrays which are maps of files in the disk (where obtained as second output in store_data_numpy)
                sampleSize = ideally should be the size of the positive training/testing set to make an unbalanced training/testing set.
        OUTPUT: list of arrays from all different numpy arrays (mapped files) that could later be stacked outside of this function.
    '''
    # check that any of the arrays is shorter than sampleSize!
    for arr in list_of_arrs:
        if arr.shape[0] < sampleSize:
            print('some array is shorter than sampleSize (' + samplezise + ')')
            return -1

    # Calculate how many samples to extract from each array
    N_arrays = len(list_of_arrs)
    N_of_each_arr = int(sampleSize / N_arrays)

    # select a randomly chosen "N_on_each_arr" long sample of each array  
    idx = np.arange(sampleSize)
    np.random.shuffle(idx)
    idx = idx[:N_of_each_arr]

    return [arr[idx] for arr in list_of_arrs]


def get_aa_frequencies(seq):
    '''
        function retrieves frequencies of aminoacids given a sequence of aminoacids
        INPUT: aminoacid sequence
        OUTPUT: array of aminoacid frequencies indexed according to global aa list
    '''
    # length of the sequence
    k = len(seq)
    
    # return the array of sequences
    return np.vstack( [seq.count(i)/k for i in aa] )

'''make get_aa_frequencies a ufunc'''
get_aa_frequencies = np.frompyfunc(get_aa_frequencies, 1,1)


def get_dipeptide_frequencies(seq):
    '''
        function retrieves frequencies of dipeptides given a sequence of aminoacids
        INPUT: aminoacid sequence
        OUTPUT: array of dipeptide frequencies indexed according to dipeptides global list
    '''
    # length of sequence
    k = len(seq)

    # return an array of the frequencies of dipeptides 
    return np.array([seq.count(i)/k for i in dipeptides])

'''make get_dipeptide_frequencies a ufunc'''
get_dipeptide_frequencies = np.frompyfunc(get_dipeptide_frequencies, 1,1)


def get_n_divisors_mmap(n=15):
    '''
        function calculate how many files to split the data into.
        data is composed by positives and negatives.
        INPUT = n. Potential maximum number of files to store data in mapped disk
        OUTPUT = 
    '''
    # Select indices for the validation set and by elimination set the training set 
    for SET in ['positives','negatives']:
        name = SET + '_validation'
        length = vars()['length_' + SET][0]
        tmp = np.arange(length)
        np.random.seed(0)
        np.random.shuffle(tmp)
        vars()[name] = tmp[:int(length_positives[0]/10)]
        vars()[SET + '_train'] = list(set(np.arange(length)) - set( eval(name) )) 

    # In how many files should I store the mapped negatives
    potential_divisors = np.arange(1,n)
    divisors = np.where(length_negatives%potential_divisors==0)[0] + 1
    print('The negative set can be devided into N arrays to make the computation faster.\n' 
        'These are good options to choose: {}'.format(divisors.astype(str)))

    # update the lengths of positives and negatives sets
    length_positives, length_negatives = len(positives_train), len(negatives_train)


def initialize_mmap_single_frequencies():
    '''
        function creates ~20  
    '''

    # create one numpy_map array for positives and 12 for negatives
    idx = positives_train
    p = get_aa_frequencies(positives[idx,0])
    p_train, p_filename = store_data_numpy(np.hstack(p).T, float)

    # set the positive validation array
    idx = positives_validation
    p_valid = get_aa_frequencies(positives[idx,0])
    p_valid = np.hstack(p_valid).T

    # negatives. SQL indexes start with 1 and not 0
    N = divisors[-1]
    idxs = np.array(negatives_train)
    idxs = np.vstack(np.split(idxs, N)) 

    n_filenames = np.empty(N, dtype='O')
    n_train_shape = tuple(np.insert(idxs.shape, 2, 20))
    n_train = np.zeros(shape=n_train_shape, dtype=np.float)
    for i in range(N):
        n = get_aa_frequencies(negatives[idxs[i],0])
        n_train[i], n_filenames[i] = store_data_numpy(np.hstack(n).T, float)

    # set the negative validation array 
    idx = negatives_validation
    n_valid = get_aa_frequencies(negatives[idx,0])
    n_valid = np.hstack(n_valid).T

    # set a proper validation set with negatives and positives
    X_valid = np.vstack([n_valid, p_valid])
    y_valid = np.hstack([np.zeros(n_valid.shape[0]), np.ones(p_valid.shape[0])])


def ohe(sequence, lexicon):
    '''
        function returns the data in ohe shape. The columns correspond to the lexicon.
        INPUT: sequence. Sequence of amino acids or secondary structure (ss) elements.
               lexicon. Ordered list of all 20 amino acids or ss elements.
        OUTPUT: ohe_data (shape = (1, len(lexicon))
        e.g. of lexicon for ss: ["E","H","-"] --> beta, alpha, coil

        NOTE: This function can be vectorized since it will constitute a ufunc 
              and the result matrix should have a shape = (len(sequences), len(lexicon))
    '''
    # define the ohe_data 
    N,n = len(sequence), len(lexicon)
    ohe_data = np.zeros(shape=(N,n))

    # fill ohe_data
    for k, i in enumerate(sequence):
        aa = lexicon.index(i)
        ohe_data[k,aa] = 1

    return ohe_data

#'''make ohe a ufunc'''
#ohe_data = np.frompyfunc(ohe, 1,1)


def prepare_ohe(INPUT):
    '''
        function align "ohe" sequence and secondary structure data into a single ohe matrix.
        INPUT: list. list[0]=aa, list[1]=ss
        OUTPUT: ohe including aa and ss
    '''
    aa_ohe = ohe(INPUT[0], aa)
    ss_ohe = ohe(INPUT[1], ss)
    
    return np.hstack([aa_ohe, ss_ohe])


def not_epoch_generator(n_train, p_train, batch_size, **kwargs):
    '''
        function generate data and labels to fit one epoch in keras.
        INPUT: n_train. Array of memory map of the ohe negative training samples. Shape:(#files, #samples/file, 30, 23)
               p_train. Array of memory map of the ohe positive training samples. Shape:(#samples, 30, 23)
               batch_size. Number of samples to include in each batch
               **kwargs. dictionary including seed for reproducibility
        OUTPUT: generator of (X_train_batch, y_train_batch) 
    '''
    # decide what to do with seed
    if 'seed' in kwargs.keys(): 
        seed = int(kwargs['seed'])
    else:
        seed = np.random.randint(500)
    
    # dimensions of n_dim correspond to (#npy-files, #samples/file) of negative set
    negat = np.concatenate(n_train)
    n_total = negat.shape[0]
    
    # prepare indexes of negatives to subsample 
    np.random.seed(seed)
    n_idx = np.random.permutation( np.arange(n_total) )
    
    # Divide the negative set into parts that are equivalent to the number of samples of the positive
    shape = n_total/p_train.shape[0]
    negat = np.array_split(negat, shape)
    
    # for each subsample from negative set 
    for idxs in negat:  
        
        # join positive and negatives
        X = np.vstack([idxs, p_train])
        X = X.reshape(np.insert(X.shape[:],3,1))
        y = np.hstack([np.zeros(idxs.shape[0]), np.ones(p_train.shape[0])])
        idx = np.random.permutation(np.arange(y.shape[0]))
        
        # shuffled samples 
        X,y = X[idx], y[idx]
        
        # now retrieve samples in mini-batches
        idx = len(X)
        batches = int(idx/batch_size)
        batches = np.array_split(np.arange(idx), batch_size)

        for batch in batches:
            yield X[batch], y[batch]

            
def epoch_generator(n_train, p_train, batch_size=100):
    '''
        function generate data and labels to fit one epoch in keras.
        INPUT: n_train. Array of memory map of the ohe negative training samples. Shape:(#files, #samples/file, 30, 23)
               p_train. Array of memory map of the ohe positive training samples. Shape:(#samples, 30, 23)
               batch_size. How many samples per batch
        OUTPUT: generator of (X_train_batch, y_train_batch) 
    '''
    # dimensions of n_dim correspond to (#npy-files, #samples/file) of negative set
    negat = np.concatenate(n_train)
    n_total = negat.shape[0]
    
    # prepare indexes of negatives to subsample 
    np.random.seed(0)
    n_idx = np.random.permutation( np.arange(n_total) )
    
    # Divide the negative set into parts that are equivalent to the number of samples of the positive
    shape = n_total/p_train.shape[0]
    negat = np.array_split(negat, shape)
    
    # join positive and negatives
    idxs = negat[0] # take the same number of positives and negatives
    X = np.vstack([idxs, p_train])
    X = X.reshape(np.insert(X.shape[:],3,1))
    y = np.hstack([np.zeros(idxs.shape[0]), np.ones(p_train.shape[0])])
    idx = np.random.permutation(np.arange(y.shape[0]))

    # shuffled samples 
    X,y = X[idx], y[idx]
    batches = len(idx)/batch_size
    
    while True:
        for i in np.array_split(idx, batches):
            yield X[i],y[i]

            
def read_fasta(fasta):
    '''
        function reads fasta
        INPUT: fasta filename
        OUTPUT: The sequence as a string
    '''
    lista = []
    for line in open(fasta):
        if line[0]!='>': 
            lista.append(line.strip('\n'))
        else: continue

    return ''.join(lista)
    

def read_horiz(horiz):
    '''
        functio opens a secondary structure (psipred) file
        INPUT: horiz filename
        OUTPUT: The secondary structure as a string. ('C' = '-')
    '''
    lista = []
    for line in open(horiz):
        if line[:4]=='Pred':
            lista.append(line[6:].strip('\n'))
        else:
            continue
            
    return ''.join(lista).replace('C','-')


def predict_TAD(ohe_data, model):
    '''
        function outputs probability of TAD per position
        INPUT: ohe_data including fasta and ss
        OUTPUT: predictions array (1D) of probabilities over the length of the protein
        NOTE: This functions slides along 30aa long windows and predict the center of the protein.
              The first and last 15 residues are repeated on purpose.
    '''
    # exit if data is not correctly shaped.
    if ohe_data.shape[1] != 23:
        print('shape should be (L,23) where L is the length of the protein...')
        return

    # repeat the first and last 15 residues.
    data = np.concatenate([ohe_data[:15], ohe_data, ohe_data[-15:]])
    
    # numpy arr with results
    predictions = np.zeros(ohe_data.shape[0])
    
    # go over 30 aa long windows 
    for pos in range(data.shape[0]-30):
        seq = data[pos:pos+30].reshape(1,30,23,1)
        predictions[pos] = model.predict(seq)
    
    return predictions


def mutate_protein(sequence, **kwargs):
    '''
        function mutate allowed residues from a sequence and returns a generator
        INPUT: sequence. Sequence of the protein.
               residues. Dictionary with diff. aminoacids allowed on each position.  
        OUTPUT: mutants. Array of strings containing the mutants.
    '''
    # get length of sequence
    length = len(sequence)
    
    # initialize the new sequence (will include mutations) with mutation genes still WT
    new_sequence = np.array([i for i in sequence])

    # if user don't provide dictionary of residues, mutate all residues to all 20 possible aa
    if 'residues' not in kwargs.keys():
        MUT_residues = np.arange(length)
        MUT_list = [aa]*length
    else:
        # which residues should be mutated
        MUT_residues = kwargs['residues'].keys()    
        MUT_list = list( kwargs['residues'].values() )

    # generate combinations in numpy arrays
    for mutations in itertools.product(*MUT_list):
        for position, mutation in zip(MUT_residues, mutations):
            new_sequence[position] = mutation
        yield ''.join(new_sequence)


def predict_motif_statistics(predictions, cutoff):
    '''
        function predict a motif and it's length
        INPUT: array of scores/residue
        OUTPUT: length of longest TAD, 
                Starting position of the longest TAD
                average prediction score on 30mer 
                average prediction score on longest TAD
    '''
    # find the largest contiguous region above cutoff
    regions = []
    flag=False
    for n,i in enumerate(predictions):
        if i > cutoff:              # if catoff 
            if not flag:
                tmp = [n,n]
                flag=True
            else:
                tmp[1] = n
        else:                       # if NOT cutoff 
            if flag:
                regions.append( tmp ) # THEN keep length of that region
                flag=False
                del tmp
        
        if n==len(predictions)-1 and flag: 
            tmp[1] = n
            regions.append( tmp )

    if regions == []: return 0,0,0,0    # if no regions, return zeros

    if len(regions)>1:              # if more than 1 fragment in regions...
        # find the largest region
        l = 0 
        for i in regions:
            if i[1]-i[0] > l:
                l = i[1]-i[0]
                longest = i
    
    else: 
        longest = regions[0]
        l = longest[1]-longest[0]
    
    if l==0: return 0,0,0,0 # in case of single residues above the cutoff
    
    length = l
    average_30mer = np.mean(predictions)
    average_longest = np.mean(predictions[longest[0]:longest[1]])
    
    return length, longest[0], average_30mer, average_longest


# helper functions for predict_on_disorder function
def find_disorder_regions(dis, cutoff):
    '''
        function finds the boundaries of the whole disorder region
        INPUT: list of 'D' or '-'
               cutoff. Number of resudues that have to be "D" to define the region as disordered
        OUTPUT: list or tuples, each tuple=(A,B), where A and B are the start and end of the disorder region
    '''
    flag = False
    results = []
    
    for i in range(len(dis)):
        
        if dis[i:i+30].count('D') >=cutoff: # enough disordered
            if not flag:
                tmp = [i,i+30]
                flag=True
            elif flag:
                tmp[1] = i+30
        
        elif dis[i:i+30].count('D') <cutoff:
            if flag:
                results.append(tmp)
                flag=False
                del tmp
                
        if i==len(dis)-1 and flag:
            results.append(tmp)
                
    return results


def fit_model_non_bias(model, train, test, iterations=10):
    '''
        function fit a sklearn model when dataset is imbalanced.
        In this case there are way more negatives
        INPUT: model, train, test and iterations (default=10)
        OUTPUT: mean and std of 20 predictions randomly resampling the negative set. 
    '''
    