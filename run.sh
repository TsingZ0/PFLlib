#!/bin/bash
# cd system/

# ===============================================================horizontal(cifar10 10client)======================================================================
# # dir fedProx cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedProx -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 > cifar10_fedProx_dir.out 2>&1 &

# # dir fedFomo cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedFomo -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 > cifar10_fedFomo_dir.out 2>&1 &

# # dir ditto cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo Ditto -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_ditto_dir.out 2>&1 &

# # # dir FedALA cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedALA -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedALA_dir.out 2>&1 &

# # dir GPFL cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo GPFL -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_GPFL_dir.out 2>&1 &

# # dir FedPAC cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedPAC -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedPAC_dir.out 2>&1 &

# # dir local cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo Local -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_Local_dir.out 2>&1 &

# # dir MOON cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo MOON -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_MOON_dir.out 2>&1 &

# # # # dir FedAvg cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedAvg -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedAvg_dir.out 2>&1 &

# ## pat
# # pat pFedMe cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo pFedMe -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -bt 1 -lam 15 -ls 5 --partition pat > cifar10_pFedMe_pat.out 2>&1 &

# # pat fedProx cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedProx -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 --partition pat > cifar10_fedProx_pat.out 2>&1 &

# # pat fedFomo cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedFomo -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 --partition pat > cifar10_fedFomo_pat.out 2>&1 &

# # pat ditto cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo Ditto -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_ditto_pat.out 2>&1 &

# # pat FedALA cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedALA -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedALA_pat.out 2>&1 &

# # # pat GPFL cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo GPFL -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat  > cifar10_GPFL_pat.out 2>&1 &

# # # pat FedPAC cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedPAC -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedPAC_pat.out 2>&1 &

# # pat MOON cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo MOON -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_MOON_pat.out 2>&1 &

# # # pat FedAvg cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedAvg -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedAvg_pat.out 2>&1 &


# ===============================================================horizontal(cifar10 100client)======================================================================
# dir pfedMe cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo pFedMe -gr 100 -did 0 -go cnn -lbs 8 -nc 100 -jr 0.1 -nb 10 -bt 1 -lam 15 -ls 5 > cifar10_pfedMe_dir_100.out 2>&1 &

# dir fedProx cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedProx -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 -mu 0.001 > cifar10_fedProx_dir_100.out 2>&1 &

# # dir fedFomo cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedFomo -gr 100 -did 0 -go cnn -lbs 8 -nc 100 -jr 0.1 -nb 10 -ls 5 -M 5 > cifar10_fedFomo_dir_100.out 2>&1 &

# # dir ditto cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo Ditto -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 > cifar10_ditto_dir_100.out 2>&1 &

# # dir FedALA cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedALA -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 > cifar10_FedALA_dir_100.out 2>&1 &

# # dir GPFL cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo GPFL -gr 100 -did 0 -go cnn -lbs 8 -nc 100 -jr 0.1 -nb 10 -ls 5 > cifar10_GPFL_dir_100.out 2>&1 &

# # dir FedPAC cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedPAC -gr 100 -did 0 -go cnn -lbs 8 -nc 100 -jr 0.1 -nb 10 -ls 5 > cifar10_FedPAC_dir_100.out 2>&1 &

# # dir MOON cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo MOON -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 > cifar10_MOON_dir_100.out 2>&1 &

# # dir FedAvg cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedAvg -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 > cifar10_FedAvg_dir_100.out 2>&1 &


# pat
# pat pFedMe cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo pFedMe -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -bt 1 -lam 15 -ls 5 --partition pat > cifar10_pFedMe_pat_100.out 2>&1 &

# pat fedProx cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedProx -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 -mu 0.001 --partition pat > cifar10_fedProx_pat_100.out 2>&1 &

# # pat fedFomo cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedFomo -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 -M 5 --partition pat > cifar10_fedFomo_pat_100.out 2>&1 &

# # pat ditto cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo Ditto -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat > cifar10_ditto_pat_100.out 2>&1 &

# # pat FedALA cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedALA -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat > cifar10_FedALA_pat_100.out 2>&1 &

# # pat GPFL cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo GPFL -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat  > cifar10_GPFL_pat_100.out 2>&1 &

# # # pat FedPAC cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedPAC -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat > cifar10_FedPAC_pat_100.out 2>&1 &

# # pat MOON cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo MOON -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat > cifar10_MOON_pat_100.out 2>&1 &

# # pat FedAvg cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedAvg -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat > cifar10_FedAvg_pat_100.out 2>&1 &







# ===============================================================horizontal(cifar100 10client)======================================================================

# dir pfedMe cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo pFedMe -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -bt 1 -lam 15 -ls 5 > cifar100_pfedMe_dir.out 2>&1 &

# # dir fedProx cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedProx -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 -mu 0.001 > cifar100_fedProx_dir.out 2>&1 &

# # dir fedFomo cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedFomo -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 -M 5 > cifar100_fedFomo_dir.out 2>&1 &

# # dir ditto cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo Ditto -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 > cifar100_ditto_dir.out 2>&1 &

# # dir FedALA cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedALA -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 > cifar100_FedALA_dir.out 2>&1 &

# # dir GPFL cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo GPFL -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 > cifar100_GPFL_dir.out 2>&1 &

# # dir FedPAC cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedPAC -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 > cifar100_FedPAC_dir.out 2>&1 &

# # dir Local cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo Local -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 > cifar100_Local_dir.out 2>&1 &

# # dir MOON cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo MOON -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 > cifar100_MOON_dir.out 2>&1 &

# # # dir FedAvg cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedAvg -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 > cifar100_FedAvg_dir.out 2>&1 &



# ## pat
# pat pFedMe cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo pFedMe -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -bt 1 -lam 15 -ls 5 --partition pat > cifar100_pFedMe_pat.out 2>&1 &

# # pat fedProx cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedProx -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 -mu 0.001 --partition pat > cifar100_fedProx_pat.out 2>&1 &

# # pat fedFomo cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedFomo -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 -M 5 --partition pat > cifar100_fedFomo_pat.out 2>&1 &

# # pat ditto cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo Ditto -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 --partition pat > cifar100_ditto_pat.out 2>&1 &

# # pat FedALA cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedALA -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 --partition pat > cifar100_FedALA_pat.out 2>&1 &

# # pat GPFL cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo GPFL -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 --partition pat  > cifar100_GPFL_pat.out 2>&1 &

# pat FedPAC cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedPAC -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 --partition pat > cifar100_FedPAC_pat.out 2>&1 &

# # # pat MOON cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo MOON -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 --partition pat > cifar100_MOON_pat.out 2>&1 &

# # # pat FedAvg cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedAvg -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 100 -ls 5 --partition pat > cifar100_FedAvg_pat.out 2>&1 &


# ===============================================================horizontal(cifar100 100client)======================================================================
# dir pfedMe cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo pFedMe -gr 100 -did 0 -go cnn -lbs 8 -nc 100 -jr 0.1 -nb 100 -bt 1 -lam 15 -ls 5 > cifar100_pfedMe_dir_100.out 2>&1 &

# # dir fedProx cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedProx -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 -mu 0.001 > cifar100_fedProx_dir_100.out 2>&1 &

# dir fedFomo cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedFomo -gr 100 -did 0 -go cnn -lbs 8 -nc 100 -jr 0.1 -nb 100 -ls 5 -M 5 > cifar100_fedFomo_dir_100.out 2>&1 &

# # dir ditto cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo Ditto -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 > cifar100_ditto_dir_100.out 2>&1 &

# # dir FedALA cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedALA -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 > cifar100_FedALA_dir_100.out 2>&1 &

# # dir GPFL cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo GPFL -gr 100 -did 0 -go cnn -lbs 8 -nc 100 -jr 0.1 -nb 100 -ls 5 > cifar100_GPFL_dir_100.out 2>&1 &

# # dir FedPAC cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedPAC -gr 100 -did 0 -go cnn -lbs 8 -nc 100 -jr 0.1 -nb 100 -ls 5 > cifar100_FedPAC_dir_100.out 2>&1 &

# # dir MOON cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo MOON -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 > cifar100_MOON_dir_100.out 2>&1 &

# # dir FedAvg cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedAvg -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 > cifar100_FedAvg_dir_100.out 2>&1 &


# pat
# pat pFedMe cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo pFedMe -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -bt 1 -lam 15 -ls 5 --partition pat > cifar100_pFedMe_pat_100.out 2>&1 &

# # pat fedProx cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedProx -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 -mu 0.001 --partition pat > cifar100_fedProx_pat_100.out 2>&1 &

# # pat fedFomo cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedFomo -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 -M 5 --partition pat > cifar100_fedFomo_pat_100.out 2>&1 &

# # pat ditto cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo Ditto -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 --partition pat > cifar100_ditto_pat_100.out 2>&1 &

# # pat FedALA cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedALA -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 --partition pat > cifar100_FedALA_pat_100.out 2>&1 &

# # pat GPFL cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo GPFL -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 --partition pat  > cifar100_GPFL_pat_100.out 2>&1 &

# # # pat FedPAC cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedPAC -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 --partition pat > cifar100_FedPAC_pat_100.out 2>&1 &

# # pat MOON cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo MOON -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 --partition pat > cifar100_MOON_pat_100.out 2>&1 &

# # pat FedAvg cifar10
# nohup python -u main.py -data Cifar100 -m cnn -algo FedAvg -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 100 -ls 5 --partition pat > cifar100_FedAvg_pat_100.out 2>&1 &


# ===============================================================horizontal(fmnist 10client)======================================================================

# dir pfedMe fmnist
# nohup python -u main.py -data fmnist -m cnn -algo pFedMe -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -bt 1 -lam 15 -ls 5 > fmnist_pfedMe_dir.out 2>&1 &

# # dir fedProx fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedProx -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 > fmnist_fedProx_dir.out 2>&1 &

# # dir fedFomo fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedFomo -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 > fmnist_fedFomo_dir.out 2>&1 &

# # dir ditto fmnist
# nohup python -u main.py -data fmnist -m cnn -algo Ditto -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > fmnist_ditto_dir.out 2>&1 &

# # dir FedALA fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedALA -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > fmnist_FedALA_dir.out 2>&1 &

# # dir GPFL fmnist
# nohup python -u main.py -data fmnist -m cnn -algo GPFL -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > fmnist_GPFL_dir.out 2>&1 &

# dir FedPAC fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedPAC -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > fmnist_FedPAC_dir.out 2>&1 &

# dir Local fmnist
# nohup python -u main.py -data fmnist -m cnn -algo Local -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > fmnist_Local_dir.out 2>&1 &

# dir MOON fmnist
# nohup python -u main.py -data fmnist -m cnn -algo MOON -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > fmnist_MOON_dir.out 2>&1 &

# # dir FedAvg fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedAvg -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > fmnist_FedAvg_dir.out 2>&1 &


# ## pat
# pat pFedMe fmnist
# nohup python -u main.py -data fmnist -m cnn -algo pFedMe -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -bt 1 -lam 15 -ls 5 --partition pat > fmnist_pFedMe_pat.out 2>&1 &

# # pat fedProx fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedProx -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 --partition pat > fmnist_fedProx_pat.out 2>&1 &

# pat fedFomo fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedFomo -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 --partition pat > fmnist_fedFomo_pat.out 2>&1 &

# pat ditto fmnist
# nohup python -u main.py -data fmnist -m cnn -algo Ditto -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > fmnist_ditto_pat.out 2>&1 &

# pat FedALA fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedALA -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > fmnist_FedALA_pat.out 2>&1 &

# # pat GPFL fmnist
# nohup python -u main.py -data fmnist -m cnn -algo GPFL -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat  > fmnist_GPFL_pat.out 2>&1 &

# pat FedPAC fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedPAC -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > fmnist_FedPAC_pat.out 2>&1 &

# # pat MOON fmnist
# nohup python -u main.py -data fmnist -m cnn -algo MOON -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > fmnist_MOON_pat.out 2>&1 &

# # pat FedAvg fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedAvg -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > fmnist_FedAvg_pat.out 2>&1 &


# ===============================================================horizontal(fmnist 100client)======================================================================

# # dir fedProx fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedProx -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 -mu 0.001 > fmnist_fedProx_dir_100.out 2>&1 &

# dir fedFomo fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedFomo -gr 100 -did 0 -go cnn -lbs 4 -nc 100 -jr 0.1 -nb 10 -ls 5 -M 5 > fmnist_fedFomo_dir_100.out 2>&1 &

# # dir ditto fmnist
# nohup python -u main.py -data fmnist -m cnn -algo Ditto -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 > fmnist_ditto_dir_100.out 2>&1 &

# # dir FedALA fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedALA -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 > fmnist_FedALA_dir_100.out 2>&1 &

# # dir GPFL fmnist
# nohup python -u main.py -data fmnist -m cnn -algo GPFL -gr 100 -did 0 -go cnn -lbs 8 -nc 100 -jr 0.1 -nb 10 -ls 5 > fmnist_GPFL_dir_100.out 2>&1 &

# # dir FedPAC fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedPAC -gr 100 -did 0 -go cnn -lbs 8 -nc 100 -jr 0.1 -nb 10 -ls 5 > fmnist_FedPAC_dir_100.out 2>&1 &

# dir MOON fmnist
# nohup python -u main.py -data fmnist -m cnn -algo MOON -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 > fmnist_MOON_dir_100.out 2>&1 &

# dir FedAvg fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedAvg -gr 100 -did 0 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 > fmnist_FedAvg_dir_100.out 2>&1 &



# pat

# # pat fedProx fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedProx -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 -mu 0.001 --partition pat > fmnist_fedProx_pat_100.out 2>&1 &

# # pat fedFomo fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedFomo -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 -M 5 --partition pat > fmnist_fedFomo_pat_100.out 2>&1 &

# # pat ditto fmnist
# nohup python -u main.py -data fmnist -m cnn -algo Ditto -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat > fmnist_ditto_pat_100.out 2>&1 &

# # pat FedALA fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedALA -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat > fmnist_FedALA_pat_100.out 2>&1 &

# # pat GPFL fmnist
# nohup python -u main.py -data fmnist -m cnn -algo GPFL -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat  > fmnist_GPFL_pat_100.out 2>&1 &

# # # pat FedPAC fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedPAC -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat > fmnist_FedPAC_pat_100.out 2>&1 &

# pat MOON fmnist
# nohup python -u main.py -data fmnist -m cnn -algo MOON -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat > fmnist_MOON_pat_100.out 2>&1 &

# pat FedAvg fmnist
# nohup python -u main.py -data fmnist -m cnn -algo FedAvg -gr 100 -did 1 -go cnn -lbs 64 -nc 100 -jr 0.1 -nb 10 -ls 5 --partition pat > fmnist_FedAvg_pat_100.out 2>&1 &



# ===============================================================horizontal(cifar10 10client dir=0.01)======================================================================
# # dir fedProx cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedProx -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 --niid_alpha 0.01 > cifar10_fedProx_dir_0.01.out 2>&1 &

# dir fedFomo cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedFomo -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 --niid_alpha 0.01 > cifar10_fedFomo_dir_0.01.out 2>&1 &

# # dir ditto cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo Ditto -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5  --niid_alpha 0.01 > cifar10_ditto_dir_0.01.out 2>&1 &

# # dir FedALA cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedALA -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5  --niid_alpha 0.01 > cifar10_FedALA_dir_0.01.out 2>&1 &

# # dir GPFL cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo GPFL -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5  --niid_alpha 0.01 > cifar10_GPFL_dir_0.01.out 2>&1 &

# # dir FedPAC cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedPAC -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5  --niid_alpha 0.01 > cifar10_FedPAC_dir_0.01.out 2>&1 &

# # dir MOON cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo MOON -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --niid_alpha 0.01 > cifar10_MOON_dir_0.01.out 2>&1 &

# # # dir FedAvg cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedAvg -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --niid_alpha 0.01 > cifar10_FedAvg_dir_0.01.out 2>&1 &


# ===============================================================horizontal(cifar10 10client dir=1)================================================================
# # dir fedProx cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedProx -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 --niid_alpha 1 > cifar10_fedProx_dir_1.out 2>&1 &

# dir fedFomo cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedFomo -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 --niid_alpha 1 > cifar10_fedFomo_dir_1.out 2>&1 &

# # dir ditto cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo Ditto -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5  --niid_alpha 1 > cifar10_ditto_dir_1.out 2>&1 &

# # dir FedALA cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedALA -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5  --niid_alpha 1 > cifar10_FedALA_dir_1.out 2>&1 &

# # dir GPFL cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo GPFL -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5  --niid_alpha 1 > cifar10_GPFL_dir_1.out 2>&1 &

# # dir FedPAC cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedPAC -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5  --niid_alpha 1 > cifar10_FedPAC_dir_1.out 2>&1 &

# # dir MOON cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo MOON -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --niid_alpha 1 > cifar10_MOON_dir_1.out 2>&1 &

# # # dir FedAvg cifar10
# nohup python -u main.py -data Cifar10 -m cnn -algo FedAvg -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --niid_alpha 1 > cifar10_FedAvg_dir_1.out 2>&1 &



