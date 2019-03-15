import os,sys
sys.path.append(os.path.abspath('../libraries/'))
from summary_utils import *
from null_distribution import *


# load genome sequence of 180 TF from Stark's publication
Stark_data_genome = read_fasta('../data/Stark_data/GSE114387_TFbb_list180_linear.fa')

# get length of the genome
genome_length = len(Stark_data_genome)

# Load annotations from stark data
Stark_data_annotations = pd.read_csv('../data/Stark_data/TFbb_list180_linear_anno.bed', delimiter='\t', 
                                     header=None, names=['_','start','end']+['_']*8, index_col=3).iloc[:,1:3]


# this is while preparing the fasta files to predict ss with psipred
for i in Stark_data_annotations.index:
    filename = i + '.fasta'
    start, end = Stark_data_annotations.loc[i, ['start', 'end']]
    dna_sequence = Stark_data_genome[start:end]
    protein_seq = translate_dna2aa(dna_sequence)
    f = open('../data/Stark_data/' + filename, 'w')
    f.write('>'+i+'\n'+protein_seq)
    f.close()
    
# Then run psipred and All Done!