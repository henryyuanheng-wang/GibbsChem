#! /usr/bin/python

## CODE TO RUN HOGG SCORES FOR VARIOUS ELEMENTS
## INCORRECT - THIS DOES NOT USE NEURAL NETWORKS!!! 
import fileinput
import os 
import sys
os.system('rm -rf BatchScoresNoSc/')
os.system('mkdir BatchScoresNoSc/')
os.system('rm -rf Scores/')
os.system('mkdir Scores')

print("1: Default Hogg no Sc Score")

for line in fileinput.input("Chempy/parameter.py", inplace=True):
	if "\tyield_table_name_sn2_index" in line:
		print("\tyield_table_name_sn2_index = 5")
	elif "\tyield_table_name_agb_index" in line:
		print("\tyield_table_name_agb_index = 2")
	elif "\tyield_table_name_1a_index" in line:
		print("\tyield_table_name_1a_index = 2") 
	else:
		print(line,end='')
fileinput.close()		

os.system('Chempy/Hogg_run.sh')
os.system('mkdir BatchScoresNoSc/Default')
os.system('scp Scores/* BatchScoresNoSc/Default/')
os.system('rm -rf Scores/')
os.system('mkdir Scores')

print("2: Chieffi Hogg no Sc Score")

for line in fileinput.input("Chempy/parameter.py", inplace=True):
	if "\tyield_table_name_sn2_index" in line:
		print("\tyield_table_name_sn2_index = 4")
	elif "\tyield_table_name_agb_index" in line:
		print("\tyield_table_name_agb_index = 2")
	elif "\tyield_table_name_1a_index" in line:
		print("\tyield_table_name_1a_index = 2") 
	else:
		print(line,end='')
fileinput.close()	
os.system('Chempy/Hogg_run.sh')
os.system('mkdir BatchScoresNoSc/Chieffi')
os.system('scp Scores/* BatchScoresNoSc/Chieffi/')
os.system('rm -rf Scores/')
os.system('mkdir Scores')
	

print('All processes complete. Outputs are in BatchScoresNoSc/ folder')