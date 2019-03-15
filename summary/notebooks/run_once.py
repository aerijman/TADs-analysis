import os,sys
sys.path.append(os.path.abspath('../libraries/'))
from summary_utils import *
import matplotlib.pyplot as plt

db = '../data/data_complete.db'

# load complete dataset and 
positives, length_positives = load_data_from_sqlite(db, 'positives')
negatives, length_negatives = load_data_from_sqlite(db, 'negatives')
print('positives: {} samples\nnegatives: {} samples'.format(length_positives[0], length_negatives[0]))


# Add this to try eliminating positive sequences without reads in bins 2 to 4
# These sequences might be noise
conn = sqlite3.connect(db)
cursor = conn.cursor()

cursor.execute('SELECT seq,ss FROM positives WHERE bin2>0 OR bin3>0 OR bin4>0')
positives = cursor.fetchall()
length_positives = [len(positives)]


# indices for positives and negatives
idx_positives = np.arange(length_positives[0])
idx_negatives = np.arange(length_negatives[0])

# shuffle indices
[np.random.shuffle(i) for i in [idx_positives, idx_negatives]]

# TEST set is 10% of positives and same number from negatives
# VALIDATION set is 10% of each (can subsample validation_negatives later)
_10p = int(length_positives[0]/10)
_10n = int(length_negatives[0]/10)

idx_test_p = idx_positives[:_10p]
idx_valid_p = idx_positives[_10p:_10p*2]
idx_train_p = idx_positives[_10p*2:]

idx_test_n = idx_negatives[:_10p]
idx_valid_n= idx_negatives[_10p:_10p+_10n]
idx_train_n = idx_negatives[_10p+_10n:]


# define the positive and negative sets
ohe_positives = np.array([prepare_ohe(i) for i in positives])
ohe_negatives = np.array([prepare_ohe(i) for i in negatives])


# create one numpy_map array for positives and 12 for negatives
p = ohe_positives[idx_train_p]
p_train, p_filename = store_data_numpy(p)

# set the positive validation array
p_valid = ohe_positives[idx_valid_p]
p_test = ohe_positives[idx_test_p]

# negatives. SQL indexes start with 1 and not 0
N = 10 #divisors[-1]
idxs = np.array(idx_train_n)
idxs = np.array_split(idxs, N) 

n_filenames = np.empty(N, dtype='O')
n_train = [np.zeros(shape=(i.shape[0],30,23), dtype=np.int8) for i in idxs]

for i in range(N):
    n = ohe_negatives[idxs[i]]
    n_train[i], n_filenames[i] = store_data_numpy(n)

# set the negative validation array 
n_valid = ohe_negatives[idx_valid_n]
n_test = ohe_negatives[idx_test_n]

# set a proper validation and test set with negatives and positives
X_valid = np.vstack([n_valid, p_valid])
y_valid = np.hstack([np.zeros(n_valid.shape[0]), np.ones(p_valid.shape[0])])
idx = np.random.permutation( np.arange(y_valid.shape[0]) )
X_valid, y_valid = X_valid[idx], y_valid[idx]

X_test = np.vstack([n_test, p_test])
y_test = np.hstack([np.zeros(n_test.shape[0]), np.ones(p_test.shape[0])])
idx = np.random.permutation( np.arange(y_test.shape[0]) )
X_test, y_test = X_test[idx], y_test[idx]


import keras.backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, GlobalMaxPooling2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model, model_from_json
from keras.activations import softmax, softplus, softsign, relu
from keras.callbacks import EarlyStopping
from keras import regularizers
import tensorflow as tf
from sklearn.metrics import roc_auc_score

### load model ###

# read model file
json_file = open("../models/deep_model.json", "r")
json_model = json_file.read()
json_file.close()

# load model into keras
model = model_from_json(json_model)

# load weights into model
model.load_weights("../models/deep_model.h5")

# Test the model
y_hat = model.predict(X_valid.reshape( np.insert(X_valid.shape, 3, 1) ))
print('ROC_AUC validation (including more negatives than positives): {:.2f}'.format(roc_auc_score(y_valid, y_hat)))


from null_distribution import *

# load genome sequence of 180 TF from Stark's publication
Stark_data_genome = read_fasta('../data/Stark_data/GSE114387_TFbb_list180_linear.fa')

# get length of the genome
genome_length = len(Stark_data_genome)

# load bed files 
gfp_p = read_bed('../data/Stark_data/GSM3140923_short-library_GFP+_1+.wig', genome_length)
gfp_m = read_bed('../data/Stark_data/GSM3140921_short-library_GFP-_1+.wig', genome_length)

# Load annotations from stark data
Stark_data_annotations = pd.read_csv('../data/Stark_data/TFbb_list180_linear_anno.bed', delimiter='\t', 
                                     header=None, names=['_','start','end']+['_']*8, index_col=3).iloc[:,1:3]

# exclude a problematic gene
Stark_data_annotations = Stark_data_annotations.drop('Su(var)2-10_FBtr0088576', axis=0)

# original factor and position indexes
f = list(Stark_data_annotations.index)  ####################################################################--> I should ONLY use genes that show enrichment over the background

windows = Stark_data_annotations.values

corrs = np.array([spearmanr(gfp_p[w[0]:w[1]], gfp_m[w[0]:w[1]])[0] for w in windows])
Rs = np.array([np.corrcoef(gfp_p[w[0]:w[1]], gfp_m[w[0]:w[1]])[0][1] for w in windows])

plt.figure(figsize=(15,15))

MINIMUM_READS = 200
#cutoffs = np.linspace(1,20,20)
cutoffs = np.linspace(1,7,20)
for n,FOLD_CHANGE_THRESHOLD in enumerate(cutoffs):
    
    # generate tables of measurements and predictions for genes in Stark data
    m_dict = make_m_dict(Stark_data_annotations, gfp_p, gfp_m, MINIMUM_READS=MINIMUM_READS, FOLD_CHANGE_THRESHOLD=FOLD_CHANGE_THRESHOLD)
    p_dict = make_p_dict(Stark_data_annotations, model)

    # join values into vectors for measurements and predictions.
    m = np.concatenate([m_dict[i][:-30] for i in f]) # [:-30] because I didn't predict score for these residues 
    p = np.concatenate([p_dict[i] for i in f])


    # compute original correlation --> IT SHOULD CORRESPOND WITH THE CORRELATION USED IN build_distribution!!
    corr = spearmanr(m,p)[0]

    ### Assuming that p value is the probability of type-I error, this area should be the equivalent
    corr_distrib = build_distribution(f,p_dict, m, n_iter=1000, corr_type='pearson') #, corr_type='spearman')
    
    # make it as subplots
    plt.subplot(5,4,n+1)
    
    p_value = calc_typeI_error(corr_distrib, corr, plot=True, bins=100, corr_type='spearman')
    plt.title("cut_off={} and nreads={}\np_val={} ".format(FOLD_CHANGE_THRESHOLD, MINIMUM_READS, p_value));
    
plt.tight_layout()
plt.savefig('figure.jpg', dpi=300)
plt.close()
