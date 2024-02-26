#!/bin/bash

#SBATCH --cpus-per-task=4  # Specify a number of cores per task
#SBATCH --mem=4G  # Request 5 gigabytes of real memory (mem)
#SBATCH --output=./Output/COM6012_Lab2.txt  # This is where your output and errors are logged

module load Java/17.0.4
module load Anaconda3/2022.05

source activate myspark

spark-submit ./Code/Lab_2_Exercise_Solution_1.py