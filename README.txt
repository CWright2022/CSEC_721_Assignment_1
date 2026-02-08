CSEC-721 Lab 1: Location Privacy
Cayden Wright

To run the code, simply run main.py:

python3 main.py

This will take all the .csv files in INPUT, pertubate them based on a Polar Laplace distribution, and output the corresponding file to OUTPUT.

PARAMETERS:
INPUT: input directory, relative to current directory
OUTPUT: output directory, relative to current directory
PRIVACY_BUDGET: Epsilon value - smaller value = bigger pertubations

ARGUMENTS:
--plot_only: skips pertubations and just plots CSVs in output directory
--pertrub_only: skips plotting