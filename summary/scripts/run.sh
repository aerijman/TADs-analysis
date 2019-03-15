#!/bin/bash

module load Python/3.6.5-foss-2016b-fh3

# create the database to avoid memory issuses during the reaining of the neural network.
#python create_database.py

# function to sum 2 numbers
p () { echo $(echo $1 + $2 | bc); }	

# cutoff for submissions to slurm
cutoff=100

# routine to go over all models built by the python script and train them in parallel with slurm
k=0
m=""
for i in $(python build_model.py)
do 
	# every 4 lines we complete the parameters needed to run the script
	# so keep counting on very line until the 5th line is reached
	k=$(p $k 1)
	
	# while <5th line, add to parameters variable $m
	if [ $k -le 5 ]
	then
		m="$m $i"
	# if 5th line reached, complete $m, use it to run the model and set it back to ""
	else
		sbatch -p campus -c 2 --job-name=${m:1:10} --wrap="python run_model_v3.py $m"
		m=$i
		k=1
	fi



	# count the number of submissions 
	subm=$(squeue -u aerijman | wc -l)	

	# avoid overloading the queue set cutoff=50
	while [ ${subm} -ge ${cutoff} ]
	do
		sleep 5
    	subm=$(squeue -u aerijman | wc -l)
	done

done

