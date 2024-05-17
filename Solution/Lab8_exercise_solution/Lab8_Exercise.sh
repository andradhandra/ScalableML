#!/bin/bash
#$ -l h_rt=1:00:00  #time needed
#$ -pe smp 4 #number of cores
#$ -l rmem=10G #number of memery
#$ -P rse-com6012 # require a com6012-reserved node
#$ -q rse-com6012.q # specify com6012 queue
#$ -o ../Output/Lab6_exercise.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M youremail@shef.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

#SBATCH --time=1:00:00  # Time allocated for the job to complete (1 hour)
#SBATCH --ntasks=1  # This script will only launch a single SLURM task
#SBATCH --cpus-per-task=4  # Allocate 4 CPUs to this job
#SBATCH --mem=10G  # Allocate 8 gigabytes of memory to this job
#SBATCH --output=../Output/Lab8_exercise.txt  # This is where your output and errors are logged.

module load Java/17.0.4

module load Anaconda3/2022.05

source activate myspark

spark-submit ../Code/Lab_8_Exercise_Solution.py
