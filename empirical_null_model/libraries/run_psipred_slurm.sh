#!/bin/bash

################################################################################################################
# can run this script with bash script.sh& or tmux or nohup and then run
# while sleep 4; do clear; queue=$(squeue -u aerijman | wc -l); results=($(wc -l horiz.results)); printf "queued= $queue\ndone= ${results[0]}\n"; done
# to track the process
################################################################################################################

# load psipred module
ml psipred/4.01-foss-2016b

# set maximum number of processes to run in SLURM
MAX_QUEUE=200

HSF1='MNNAANTGTTNESNVSDAPRIEPLPSLNDDDIEKILQPNDIFTTDRTDASTTSSTAIEDIINPSLDPQSAASPVPSSSFFHDSRKPSTSTHLVRRGTPLGIYQTNLYGHNSRENTNPNSTLLSSKLLAHPPVPYGQNPDLLQHAVYRAQPSSGTTNAQPRQTTRRYQSHKSRPAFVNKLWSMLNDDSNTKLIQWAEDGKSFIVTNREEFVHQILPKYFKHSNFASFVRQLNMYGWHKVQ
DVKSGSIQSSSDDKWQFENENFIRGREDLLEKIIRQKGSSNNHNSPSGNGNPANGSNIPLDNAAGSNNSNNNISSSNSFFNNGHLLQGKTLRLMNEANLGDKNDVTAILGELEQIKYNQIAISKDLLRINKDNELLWQENMMARERHRTQQQALEKMFRFLTSIVPHLDPKMIMDGLGDPKVNNEKLNSANNIGLNRDNTGTIDELKSNDSFINDDRNSFTNATTNARNNMSPNNDDNSIDTAST
NTTNRKKNIDENIKNNNDIINDIIFNTNLANNLSNYNSNNNAGSPIRPYKQRYLLKNRANSSTSSENPSLTPFDIESNNDRKISEIPFDDEEEEETDFRPFTSRDPNNQTSENTFDPNRFTMLSDDDLKKDSHTNDNKHNESDLFWDNVHRNIDEQDARLQNLENMVHILSPGYPNKSFNNKTSSTNTNSNMESAVNVNSPGFNLQDYLTGESNSPNSVHSVPSNGSGSTPLPMPNDNDTEHAST
SVNQGENGSGLTPFLTVDDHTLNDNNTSEGSTRVSPDIKFSATENTKVSDNLPSFNDHSYSTQADTAPENAKKRFVEEIPEPAIVEIQDPTEYNDHRLPKRAKK'

# 5' primer to add at "N" terminal (left of the sequence)
p5=${HSF1:463:30}

header=true # file has header and I have to skip it

# open file containing the sequence fused at the right of p5
for insert in `cat $1 | awk 'BEGIN{FS=","}{print $2}'`
do
	# if header, then continue with next iteration and flag header as false
	if [ $header = true ] 
	then
		header=false	
	else 
		printf ">${insert}\n${p5}${insert}" > ${insert}.fasta # write fasta file (this is the input of psipred)
		
		# check how many processes are in the queue
		queue=$(squeue -u aerijman | wc -l)
		queue=$(echo $queue -1 | bc)
		
		# if few processes queued, proceed, else wait.
		if [ $queue -lt $MAX_QUEUE ]
		then
			sbatch -p campus -c 1 --job-name=${insert} --wrap="runpsipred ${insert}.fasta"		
		else
			# take the chance to find *horiz files which contain the result
			for prefix in `ls *horiz`
			do
				# extract the resulting sequence of 2ry structure elements and append it to a ingle file with all esults
				horiz=$(while read line; do if [ "${line:0:4}" == Pred ]; then echo ${line:6:${#line}} | tr -d "\n"; fi; done < $prefix)
				printf ">${p5}${prefix:0:-6}\n${horiz}\n" >> horiz.results
				# rm all side files (from psipred-blast)
				rm ${prefix:0:-5}*
			done
			
			# This  loop is tracking if any process has finished (so a new processes can ve queued)
			while [ $queue -ge $MAX_QUEUE ]
			do
				queue=$(squeue -u aerijman | wc -l)
		        queue=$(echo $queue -1 | bc)

			done
		fi
	fi
done

# finish with the last files.
while [ $queue -gt 0 ] 
do
	queue=$(squeue -u aerijman | wc -l)
	queue=$(echo $queue -1 | bc)
		
	# take the chance to find *horiz files which contain the result
	for prefix in `ls *horiz`
	do
		# extract the resulting sequence of 2ry structure elements and append it to a ingle file with all esults
		horiz=$(while read line; do if [ "${line:0:4}" == Pred ]; then echo ${line:6:${#line}} | tr -d "\n"; fi; done < $prefix)
		printf ">${p5}${prefix:0:-6}\n${horiz}\n" >> horiz.results
		# rm all side files (from psipred-blast)
		rm ${prefix:0:-5}*
	done
	
	# remove junk files
	rm *blast *out *pn *mn *chk *aux *fasta *horiz
done
