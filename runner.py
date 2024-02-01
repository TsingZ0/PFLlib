strings = [
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedProx -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 > cifar10_fedProx_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedFomo -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 > cifar10_fedFomo_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo Ditto -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_ditto_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedALA -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedALA_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo GPFL -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_GPFL_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedPAC -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedPAC_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo MOON -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_MOON_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedAvg -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedAvg_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo pFedMe -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -bt 1 -lam 15 -ls 5 --partition pat > cifar10_pFedMe_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedProx -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 --partition pat > cifar10_fedProx_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedFomo -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 --partition pat > cifar10_fedFomo_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo Ditto -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_ditto_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedALA -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedALA_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo GPFL -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat  > cifar10_GPFL_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedPAC -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedPAC_pat.out 2>&1 &"
]

