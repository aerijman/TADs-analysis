#!/bin/python
'''
	This script creates a database in "../data/" folder named data_complete.db in sqlite
	The database includes two tables 'positives' and 'negatives'
	Each table includes sequence, reads in bg and bins 1 to 4 and ss (secondary structure)
	The script can be executed with "python database.py" without arguments. 
'''

import os,sys
sys.path.append(os.path.abspath('../libraries/'))
from utils import *

# get path and check it exists
PATH = os.path.abspath('../data/')
if not os.path.isdir(PATH):
	sys.exit('There is a problem with the data folder')

############################
# LOAD AND TRANSFORM DATA  #
############################

# load data
bg = os.path.join(PATH, 'ARG3_O1_O2_clustered_rescored.fasta_bg.fasta')
bins = os.path.join(PATH, 'ARG3_O1_O2_clustered_rescored.fasta_bin14_threshold10.0_37923positives.fasta')
psipred = os.path.join(PATH, '30mers_ALL.PSIPRED_ss.fasta')

def main():

	# load negatives and positives sequences with reads in bins 
	positives = load_data(bins, 1)
	negatives = load_data(bg, 0)

	# load psipred secondary structure predictions with sequences as indices
	ss_psipred_df = load_psipred(os.path.join(PATH, psipred))

	# The index for the combined matrix should be identical for ss_psipred and sequences
	set_p, set_n, set_s = set(positives.index), set(negatives.index), set(ss_psipred_df.index)

	# I don't need sequences for which I don't have part of the data in my datasets
	p_idx, n_idx = set_p & set_s, set_n & set_s 

	# join columns, reshape and cast for sql
	columns = ['b0','b1','b2','b3','b4']
	def generate_df(df, idx): 
		new_df = pd.concat([df.loc[idx, columns], ss_psipred_df.loc[idx]], axis=1)
		return np.hstack([np.array(new_df.index).reshape(-1,1), new_df.values]) 

	positives = generate_df(positives, p_idx)
	negatives = generate_df(negatives, n_idx)

	###########################
	# CREATE DATABASE OF TADS #
 	###########################

	import sqlite3


	# sql command to create the table
	sql_create_table = lambda tableName: ''' CREATE TABLE IF NOT EXISTS {} ( 
										id integer NOT NULL PRIMARY KEY,
										seq text NOT NULL,
										bg integer NOT NULL,
										bin1 integer NOT NULL,
										bin2 integer NOT NULL,
										bin3 integer NOT NULL,
										bin4 integer NOT NULL,
										ss text NOT NULL);
										'''.format(tableName)

	# sql command to import samples into table
	import_samples = lambda table:'INSERT INTO {}(seq, bg, bin1, bin2, bin3, bin4, ss) VALUES(?,?,?,?,?,?,?)'.format(table)

	# connect to db
	conn = sqlite3.connect(os.path.join(PATH, 'data_complete.db'))
	cursor = conn.cursor()

	# create positive and negative tables
	cursor.execute( sql_create_table('positives') )
	cursor.execute( sql_create_table('negatives') )

	# import samples into tables
	cursor.executemany(import_samples('negatives'), negatives)
	cursor.executemany(import_samples('positives'), positives)

	# commit changes
	conn.commit()

	# close connection to db
	conn.close()



def load_data(dataset_fasta, label):
	'''
		Function load the data into a list suitable to export to SQL
		INPUT: dataset_fasta (a fasta file that contains info of bins in header and sequences)
				label (1 for TAD and 0 for non-TADs)
		OUTPUT: A pandas dataframe of lists composed of (sequence, reads in each bin and label)
	'''
	# fill data matrix
	f, n, data = open(dataset_fasta, 'r'), 0, []
	while True:
		try:
			header, sequence = next(f), next(f)	
			sequence = sequence.strip('\n')
			_, b0, b1, b2, b3, b4, _, _ = header.strip('\n').split('_')
			
			# avoid sequences with Stop codons
			if 'X' in sequence: continue
			data.append([sequence,b0,b1,b2,b3,b4,label])
			n+=1

		except StopIteration:
			print(dataset_fasta+' succesfully loaded!')
			break

	data = np.array(data)
	return pd.DataFrame(data[:,1:], index= data[:,0], columns=(['b0','b1','b2','b3','b4','label']))


def load_psipred(filename):
	'''
		function opens psipred results fasta file and retrieves the data in a table to store in a database
		INPUT:  filename of the fasta file
		OUTPUT: table (pandas df) with secondary structures, where index=sequence  
	'''
	# open file and read content filling output list
	output = []
	f = open(filename, 'r')
	while True:
		try:
			header, seq, ss, _ = next(f), next(f), next(f), next(f)
			output.append([i.strip('\n') for i in (seq,ss)])
		except StopIteration:
			print(filename + 'successfuly loaded!')
			break

	output = np.array(output)

	return pd.DataFrame(output[:,1], index=output[:,0])


if __name__=='__main__': main()