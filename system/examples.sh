#!/bin/bash

# ===============================================================horizontal(mnist)======================================================================


# rm ../dataset/mnist/config.json
# cd ../dataset/
# nohup python -u generate_mnist.py noniid - dir > mnist_dataset.out 2>&1
# cd ../system/

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo FedAvg -gr 2000 -did 0 -go dnn > mnist_fedavg.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m cnn -algo FedAvg -gr 2000 -did 0 -go cnn > mnist_fedavg1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m resnet -algo FedAvg -gr 2000 -did 0 -go resnet > mnist_fedavg2.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo FedAvg -gr 2000 -did 0 -cdr 0.5 -go unstable > mnist_fedavg1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo FedProx -gr 2000 -did 0 -mu 0.001 -go dnn > mnist_fedprox.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m cnn -algo FedProx -gr 2000 -did 0 -mu 0.001 -go cnn > mnist_fedprox1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m resnet -algo FedProx -gr 2000 -did 0 -mu 0.001 -go resnet > mnist_fedprox2.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo pFedMe -gr 2000 -did 0 -lrp 0.09 -bt 1 -lam 15 -go dnn > mnist_pfedme.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m cnn -algo pFedMe -gr 2000 -did 0 -lrp 0.1 -bt 1 -lam 15 -go cnn > mnist_pfedme1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m resnet -algo pFedMe -gr 2000 -did 0 -lrp 0.1 -bt 1 -lam 15 -go resnet > mnist_pfedme2.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo pFedMe -gr 2000 -did 0 -lrp 0.09 -bt 1 -lam 15 -cdr 0.5 -go unstable > mnist_pfedme1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go dnn > mnist_peravg.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m cnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go cnn > mnist_peravg1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m resnet -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go resnet > mnist_peravg2.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -cdr 0.5 -go unstable > mnist_peravg1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo FedFomo -gr 2000 -M 5 -did 1 -go dnn > mnist_fedfomo.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m cnn -algo FedFomo -gr 2000 -M 5 -did 1 -go cnn > mnist_fedfomo1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m resnet -algo FedFomo -gr 2000 -M 5 -did 1 -go resnet > mnist_fedfomo2.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo FedMTL -gr 2000 -itk 4000 -did 1 -go dnn > mnist_FedMTL.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m cnn -algo FedMTL -gr 2000 -itk 4000 -did 1 -go cnn > mnist_FedMTL1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m resnet -algo FedMTL -gr 2000 -itk 4000 -did 1 -go resnet > mnist_FedMTL2.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m dnn -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 0 -go dnn > mnist_fedamp.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m cnn -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 0 -go cnn > mnist_fedamp1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data mnist -m resnet -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 0 -go resnet > mnist_fedamp2.out 2>&1 &


# ===============================================================horizontal(Cifar10)======================================================================


# rm ../dataset/Cifar10/config.json
# cd ../dataset/
# nohup python -u generate_cifar10.py noniid - dir > cifar10_dataset.out 2>&1
# cd ../system/

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo FedAvg -gr 2000 -did 0 -go dnn > cifar10_fedavg.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo FedAvg -gr 2000 -did 0 -go cnn > cifar10_fedavg1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m resnet -algo FedAvg -gr 2000 -did 0 -go resnet > cifar10_fedavg2.out 2>&1 &
# # nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo FedAvg -gr 2000 -did 0 -cdr 0.5 -go unstable > cifar10_fedavg1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo FedProx -gr 2000 -did 0 -mu 0.001 -go dnn > cifar10_fedprox.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo FedProx -gr 2000 -did 0 -mu 0.001 -go cnn > cifar10_fedprox1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m resnet -algo FedProx -gr 2000 -did 0 -mu 0.001 -go resnet > cifar10_fedprox2.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo pFedMe -gr 2000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go dnn > cifar10_pfedme.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo pFedMe -gr 2000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go cnn > cifar10_pfedme1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m resnet -algo pFedMe -gr 2000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go resnet > cifar10_pfedme2.out 2>&1 &
# # nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo pFedMe -gr 2000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -cdr 0.5 -go unstable > cifar10_pfedme1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go dnn > cifar10_peravg.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go cnn > cifar10_peravg1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m resnet -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go resnet > cifar10_peravg2.out 2>&1 &
# # nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -cdr 0.5 -go unstable > cifar10_peravg1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo FedFomo -gr 2000 -M 5 -did 1 -go dnn > cifar10_fedfomo.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo FedFomo -gr 2000 -M 5 -did 1 -go cnn > cifar10_fedfomo1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m resnet -algo FedFomo -gr 2000 -M 5 -did 1 -go resnet > cifar10_fedfomo2.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo FedMTL -gr 2000 -itk 4000 -did 1 -go dnn > cifar10_FedMTL.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo FedMTL -gr 2000 -itk 4000 -did 1 -go cnn > cifar10_FedMTL1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m resnet -algo FedMTL -gr 2000 -itk 4000 -did 1 -go resnet > cifar10_FedMTL2.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 0 -go dnn > cifar10_fedamp.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 0 -go cnn > cifar10_fedamp1.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m resnet -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 0 -go resnet > cifar10_fedamp2.out 2>&1 &


# ===============================================================horizontal(Cifar100)======================================================================


# rm ../dataset/Cifar100/config.json
# cd ../dataset/
# nohup python -u generate_cifar100.py noniid - dir > cifar100_dataset.out 2>&1
# cd ../system/

# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo FedAvg -gr 2000 -did 0 -go dnn > cifar100_fedavg.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m cnn -algo FedAvg -gr 2000 -did 0 -go cnn > cifar100_fedavg1.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m resnet -algo FedAvg -gr 2000 -did 0 -go resnet > cifar100_fedavg2.out 2>&1 &

# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo FedProx -gr 2000 -did 0 -mu 0.001 -go dnn > cifar100_fedprox.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m cnn -algo FedProx -gr 2000 -did 0 -mu 0.001 -go cnn > cifar100_fedprox1.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m resnet -algo FedProx -gr 2000 -did 0 -mu 0.001 -go resnet > cifar100_fedprox2.out 2>&1 &

# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo pFedMe -gr 2000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go dnn > cifar100_pfedme.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m cnn -algo pFedMe -gr 2000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go cnn > cifar100_pfedme1.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m resnet -algo pFedMe -gr 2000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go resnet > cifar100_pfedme2.out 2>&1 &

# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go dnn > cifar100_peravg.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m cnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go cnn > cifar100_peravg1.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m resnet -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go resnet > cifar100_peravg2.out 2>&1 &

# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo FedFomo -gr 2000 -M 5 -did 1 -go dnn > cifar100_fedfomo.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m cnn -algo FedFomo -gr 2000 -M 5 -did 1 -go cnn > cifar100_fedfomo1.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m resnet -algo FedFomo -gr 2000 -M 5 -did 1 -go resnet > cifar100_fedfomo2.out 2>&1 &

# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo FedMTL -gr 2000 -itk 4000 -did 1 -go dnn > cifar100_FedMTL.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m cnn -algo FedMTL -gr 2000 -itk 4000 -did 1 -go cnn > cifar100_FedMTL1.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m resnet -algo FedMTL -gr 2000 -itk 4000 -did 1 -go resnet > cifar100_FedMTL2.out 2>&1 &

# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 0 -go dnn > cifar100_fedamp.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m cnn -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 0 -go cnn > cifar100_fedamp1.out 2>&1 &
# nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m resnet -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 0 -go resnet > cifar100_fedamp2.out 2>&1 &


# ===============================================================horizontal(fmnist)======================================================================  

# rm ../dataset/fmnist/config.json
# cd ../dataset/
# nohup python -u generate_fmnist.py noniid - dir > fmnist_dataset.out 2>&1
# cd ../system/

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m dnn -algo FedAvg -gr 2000 -did 0 -go dnn > fmnist_fedavg.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m cnn -algo FedAvg -gr 2000 -did 0 -go cnn > fmnist_fedavg1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m dnn -algo FedProx -gr 2000 -did 0 -mu 0.001 -go dnn > fmnist_fedprox.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m cnn -algo FedProx -gr 2000 -did 0 -mu 0.001 -go cnn > fmnist_fedprox1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m dnn -algo pFedMe -gr 2000 -did 0 -lrp 0.09 -bt 1 -lam 15 -go dnn > fmnist_pfedme.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m cnn -algo pFedMe -gr 2000 -did 0 -lrp 0.1 -bt 1 -lam 15 -go cnn > fmnist_pfedme1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m dnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go dnn > fmnist_peravg.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m cnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go cnn > fmnist_peravg1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m dnn -algo FedFomo -gr 2000 -M 5 -did 1 -go dnn > fmnist_fedfomo.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m cnn -algo FedFomo -gr 2000 -M 5 -did 1 -go cnn > fmnist_fedfomo1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m dnn -algo FedMTL -gr 2000 -itk 4000 -did 1 -go dnn > fmnist_FedMTL.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m cnn -algo FedMTL -gr 2000 -itk 4000 -did 1 -go cnn > fmnist_FedMTL1.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m dnn -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 1 -go dnn > fmnist_fedamp.out 2>&1 &
# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data fmnist -m cnn -algo FedAMP -gr 2000 -alk 1e3 -lam 1 -sg 1e-1 -did 1 -go cnn > fmnist_fedamp1.out 2>&1 &


# ===============================================================horizontal(agnews)======================================================================  

# rm ../dataset/agnews/config.json
# cd ../dataset/
# nohup python -u generate_agnews.py noniid - dir > agnews_dataset.out 2>&1
# cd ../system/

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 4 -data agnews -m lstm -algo FedAvg -gr 500 -did 0 -go lstm > agnews_fedavg.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 4 -data agnews -m lstm -algo FedProx -gr 500 -did 0 -mu 0.001 -go lstm > agnews_fedprox.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 4 -data agnews -m lstm -algo pFedMe -gr 500 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go lstm > agnews_pfedme.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 4 -data agnews -m lstm -algo PerAvg -gr 500 -did 0 -bt 0.001 -go lstm > agnews_peravg.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 4 -data agnews -m lstm -algo FedFomo -gr 500 -M 5 -did 1 -go lstm > agnews_fedfomo.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 4 -data agnews -m lstm -algo FedMTL -gr 500 -itk 4000 -did 1 -go lstm > agnews_FedMTL.out 2>&1 &

# nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 4 -data agnews -m lstm -algo FedAMP -gr 500 -alk 1e3 -lam 1 -sg 1e-1 -did 0 -go lstm > agnews_fedamp.out 2>&1 &
