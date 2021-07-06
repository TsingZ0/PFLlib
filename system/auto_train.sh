#!/bin/bash

# rm -r ../figures
# rm -r ../results
# rm -r ./models


# ===============================================================(mnist)======================================================================


# rm ../dataset/mnist/config.json
# cd ../dataset/
# nohup python generate_mnist.py noniid realworld > mnist_dataset.out 2>&1
# cd ../system/

# nohup python main.py -data mnist -m dnn -algo FedAvg -gr 5000 -did 0 -go dnn > mnist_fedavg.out 2>&1 &
# nohup python main.py -data mnist -m cnn -algo FedAvg -gr 5000 -did 0 -go cnn > mnist_fedavg1.out 2>&1 &
# nohup python main.py -data mnist -m resnet18 -algo FedAvg -gr 2000 -did 0 -go resnet > mnist_fedavg2.out 2>&1 &
# nohup python main.py -data mnist -m dnn -algo FedAvg -gr 5000 -did 0 -cdr 0.5 -go unstable > mnist_fedavg1.out 2>&1 &

# nohup python main.py -data mnist -m dnn -algo FedProx -gr 5000 -did 0 -mu 1 -go dnn > mnist_fedprox.out 2>&1 &
# nohup python main.py -data mnist -m cnn -algo FedProx -gr 5000 -did 0 -mu 1 -go cnn > mnist_fedprox1.out 2>&1 &
# nohup python main.py -data mnist -m resnet18 -algo FedProx -gr 2000 -did 0 -mu 1 -go resnet > mnist_fedprox2.out 2>&1 &

# nohup python main.py -data mnist -m dnn -algo pFedMe -gr 5000 -did 0 -lrp 0.09 -bt 1 -lam 15 -go dnn > mnist_pfedme.out 2>&1 &
# nohup python main.py -data mnist -m cnn -algo pFedMe -gr 5000 -did 0 -lrp 0.1 -bt 1 -lam 15 -go cnn > mnist_pfedme1.out 2>&1 &
# nohup python main.py -data mnist -m resnet18 -algo pFedMe -gr 2000 -did 0 -lrp 0.1 -bt 1 -lam 15 -go resnet > mnist_pfedme2.out 2>&1 &
# nohup python main.py -data mnist -m dnn -algo pFedMe -gr 5000 -did 0 -lrp 0.09 -bt 1 -lam 15 -cdr 0.5 -go unstable > mnist_pfedme1.out 2>&1 &

# nohup python main.py -data mnist -m dnn -algo PerAvg -gr 5000 -did 0 -bt 0.001 -go dnn > mnist_peravg.out 2>&1 &
# nohup python main.py -data mnist -m cnn -algo PerAvg -gr 5000 -did 0 -bt 0.001 -go cnn > mnist_peravg1.out 2>&1 &
# nohup python main.py -data mnist -m resnet18 -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go resnet > mnist_peravg2.out 2>&1 &
# nohup python main.py -data mnist -m dnn -algo PerAvg -gr 5000 -did 0 -bt 0.001 -cdr 0.5 -go unstable > mnist_peravg1.out 2>&1 &

# nohup python main.py -data mnist -m dnn -algo FedFomo -gr 5000 -M 5 -did 1 -go dnn > mnist_fedfomo.out 2>&1 &
# nohup python main.py -data mnist -m cnn -algo FedFomo -gr 5000 -M 5 -did 1 -go cnn > mnist_fedfomo1.out 2>&1 &
# nohup python main.py -data mnist -m resnet18 -algo FedFomo -gr 2000 -M 5 -did 1 -go resnet > mnist_fedfomo2.out 2>&1 &

# nohup python main.py -data mnist -m dnn -algo MOCHA -gr 5000 -itk 4000 -did 1 -go dnn > mnist_mocha.out 2>&1 &
# nohup python main.py -data mnist -m cnn -algo MOCHA -gr 5000 -itk 4000 -did 1 -go cnn > mnist_mocha1.out 2>&1 &
# nohup python main.py -data mnist -m resnet18 -algo MOCHA -gr 2000 -itk 4000 -did 1 -go resnet > mnist_mocha2.out 2>&1 &

# nohup python main.py -data mnist -m sep_dnn -algo FedPlayer -gr 5000 -did 1 -go dnn > mnist_fedplayer.out 2>&1 &
# nohup python main.py -data mnist -m sep_cnn -algo FedPlayer -gr 5000 -did 1 -go cnn > mnist_fedplayer1.out 2>&1 &
# nohup python main.py -data mnist -m sep_resnet18 -algo FedPlayer -gr 2000 -did 1 -go resnet > mnist_fedplayer2.out 2>&1 &

# nohup python main.py -data mnist -m dnn -algo FedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.01 -did 0 -go dnn > mnist_fedamp.out 2>&1 &
# nohup python main.py -data mnist -m cnn -algo FedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.01 -did 0 -go cnn > mnist_fedamp1.out 2>&1 &
# nohup python main.py -data mnist -m resnet18 -algo FedAMP -gr 2000 -alk 0.002 -lam 1.0 -sg 0.01 -did 0 -go resnet > mnist_fedamp2.out 2>&1 &

# nohup python main.py -data mnist -m dnn -algo HeurFedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.1 -xi 0.1 -did 1 -go dnn > mnist_heurfedamp.out 2>&1 &
# nohup python main.py -data mnist -m cnn -algo HeurFedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.1 -xi 0.1 -did 1 -go cnn > mnist_heurfedamp1.out 2>&1 &
# nohup python main.py -data mnist -m resnet18 -algo HeurFedAMP -gr 2000 -alk 0.002 -lam 1.0 -sg 0.1 -xi 0.1 -did 1 -go resnet > mnist_heurfedamp2.out 2>&1 &


# ===============================================================(Cifar10)======================================================================


# rm ../dataset/Cifar10/config.json
# cd ../dataset/
# nohup python generate_cifar10.py noniid realworld > cifar10_dataset.out 2>&1
# cd ../system/

# nohup python main.py -data Cifar10 -m dnn -algo FedAvg -gr 5000 -did 0 -go dnn > cifar10_fedavg.out 2>&1 &
# nohup python main.py -data Cifar10 -m cnn -algo FedAvg -gr 5000 -did 0 -go cnn > cifar10_fedavg1.out 2>&1 &
# nohup python main.py -data Cifar10 -m resnet18 -algo FedAvg -gr 2000 -did 0 -go resnet > cifar10_fedavg2.out 2>&1 &
# # nohup python main.py -data Cifar10 -m dnn -algo FedAvg -gr 5000 -did 0 -cdr 0.5 -go unstable > cifar10_fedavg1.out 2>&1 &

# nohup python main.py -data Cifar10 -m dnn -algo FedProx -gr 5000 -did 0 -mu 1 -go dnn > cifar10_fedprox.out 2>&1 &
# nohup python main.py -data Cifar10 -m cnn -algo FedProx -gr 5000 -did 0 -mu 1 -go cnn > cifar10_fedprox1.out 2>&1 &
# nohup python main.py -data Cifar10 -m resnet18 -algo FedProx -gr 2000 -did 0 -mu 1 -go resnet > cifar10_fedprox2.out 2>&1 &

# nohup python main.py -data Cifar10 -m dnn -algo pFedMe -gr 5000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go dnn > cifar10_pfedme.out 2>&1 &
# nohup python main.py -data Cifar10 -m cnn -algo pFedMe -gr 5000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go cnn > cifar10_pfedme1.out 2>&1 &
# nohup python main.py -data Cifar10 -m resnet18 -algo pFedMe -gr 2000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go resnet > cifar10_pfedme2.out 2>&1 &
# # nohup python main.py -data Cifar10 -m dnn -algo pFedMe -gr 5000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -cdr 0.5 -go unstable > cifar10_pfedme1.out 2>&1 &

# nohup python main.py -data Cifar10 -m dnn -algo PerAvg -gr 5000 -did 0 -bt 0.001 -go dnn > cifar10_peravg.out 2>&1 &
# nohup python main.py -data Cifar10 -m cnn -algo PerAvg -gr 5000 -did 0 -bt 0.001 -go cnn > cifar10_peravg1.out 2>&1 &
# nohup python main.py -data Cifar10 -m resnet18 -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go resnet > cifar10_peravg2.out 2>&1 &
# # nohup python main.py -data Cifar10 -m dnn -algo PerAvg -gr 5000 -did 0 -bt 0.001 -cdr 0.5 -go unstable > cifar10_peravg1.out 2>&1 &

# nohup python main.py -data Cifar10 -m dnn -algo FedFomo -gr 5000 -M 5 -did 1 -go dnn > cifar10_fedfomo.out 2>&1 &
# nohup python main.py -data Cifar10 -m cnn -algo FedFomo -gr 5000 -M 5 -did 1 -go cnn > cifar10_fedfomo1.out 2>&1 &
# nohup python main.py -data Cifar10 -m resnet18 -algo FedFomo -gr 2000 -M 5 -did 1 -go resnet > cifar10_fedfomo2.out 2>&1 &

# nohup python main.py -data Cifar10 -m dnn -algo MOCHA -gr 5000 -itk 4000 -did 1 -go dnn > cifar10_mocha.out 2>&1 &
# nohup python main.py -data Cifar10 -m cnn -algo MOCHA -gr 5000 -itk 4000 -did 1 -go cnn > cifar10_mocha1.out 2>&1 &
# nohup python main.py -data Cifar10 -m resnet18 -algo MOCHA -gr 2000 -itk 4000 -did 1 -go resnet > cifar10_mocha2.out 2>&1 &

# nohup python main.py -data Cifar10 -m sep_dnn -algo FedPlayer -gr 5000 -did 1 -go dnn > cifar10_fedplayer.out 2>&1 &
# nohup python main.py -data Cifar10 -m sep_cnn -algo FedPlayer -gr 5000 -did 1 -go cnn > cifar10_fedplayer1.out 2>&1 &
# nohup python main.py -data Cifar10 -m sep_resnet18 -algo FedPlayer -gr 2000 -did 1 -go resnet > cifar10_fedplayer2.out 2>&1 &

# nohup python main.py -data Cifar10 -m dnn -algo FedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.01 -did 0 -go dnn > cifar10_fedamp.out 2>&1 &
# nohup python main.py -data Cifar10 -m cnn -algo FedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.01 -did 0 -go cnn > cifar10_fedamp1.out 2>&1 &
# nohup python main.py -data Cifar10 -m resnet18 -algo FedAMP -gr 2000 -alk 0.002 -lam 1.0 -sg 0.01 -did 0 -go resnet > cifar10_fedamp2.out 2>&1 &

# nohup python main.py -data Cifar10 -m dnn -algo HeurFedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.1 -xi 0.1 -did 1 -go dnn > cifar10_heurfedamp.out 2>&1 &
# nohup python main.py -data Cifar10 -m cnn -algo HeurFedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.1 -xi 0.1 -did 1 -go cnn > cifar10_heurfedamp1.out 2>&1 &
# nohup python main.py -data Cifar10 -m resnet18 -algo HeurFedAMP -gr 2000 -alk 0.002 -lam 1.0 -sg 0.1 -xi 0.1 -did 1 -go resnet > cifar10_heurfedamp2.out 2>&1 &


# ===============================================================(fmnist)======================================================================  

# rm ../dataset/fmnist/config.json
# cd ../dataset/
# nohup python generate_fmnist.py noniid realworld > fmnist_dataset.out 2>&1
# cd ../system/

# nohup python main.py -data fmnist -m dnn -algo FedAvg -gr 5000 -did 0 -go dnn > fmnist_fedavg.out 2>&1 &
# nohup python main.py -data fmnist -m cnn -algo FedAvg -gr 5000 -did 0 -go cnn > fmnist_fedavg1.out 2>&1 &

# nohup python main.py -data fmnist -m dnn -algo FedProx -gr 5000 -did 0 -mu 1 -go dnn > fmnist_fedprox.out 2>&1 &
# nohup python main.py -data fmnist -m cnn -algo FedProx -gr 5000 -did 0 -mu 1 -go cnn > fmnist_fedprox1.out 2>&1 &

# nohup python main.py -data fmnist -m dnn -algo pFedMe -gr 5000 -did 0 -lrp 0.09 -bt 1 -lam 15 -go dnn > fmnist_pfedme.out 2>&1 &
# nohup python main.py -data fmnist -m cnn -algo pFedMe -gr 5000 -did 0 -lrp 0.1 -bt 1 -lam 15 -go cnn > fmnist_pfedme1.out 2>&1 &

# nohup python main.py -data fmnist -m dnn -algo PerAvg -gr 5000 -did 0 -bt 0.001 -go dnn > fmnist_peravg.out 2>&1 &
# nohup python main.py -data fmnist -m cnn -algo PerAvg -gr 5000 -did 0 -bt 0.001 -go cnn > fmnist_peravg1.out 2>&1 &

# nohup python main.py -data fmnist -m dnn -algo FedFomo -gr 5000 -M 5 -did 1 -go dnn > fmnist_fedfomo.out 2>&1 &
# nohup python main.py -data fmnist -m cnn -algo FedFomo -gr 5000 -M 5 -did 1 -go cnn > fmnist_fedfomo1.out 2>&1 &

# nohup python main.py -data fmnist -m dnn -algo MOCHA -gr 5000 -itk 4000 -did 1 -go dnn > fmnist_mocha.out 2>&1 &
# nohup python main.py -data fmnist -m cnn -algo MOCHA -gr 5000 -itk 4000 -did 1 -go cnn > fmnist_mocha1.out 2>&1 &

# nohup python main.py -data fmnist -m sep_dnn -algo FedPlayer -gr 5000 -did 1 -go dnn > fmnist_fedplayer.out 2>&1 &
# nohup python main.py -data fmnist -m sep_cnn -algo FedPlayer -gr 5000 -did 1 -go cnn > fmnist_fedplayer1.out 2>&1 &

# nohup python main.py -data fmnist -m dnn -algo FedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.01 -did 1 -go dnn > fmnist_fedamp.out 2>&1 &
# nohup python main.py -data fmnist -m cnn -algo FedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.01 -did 1 -go cnn > fmnist_fedamp1.out 2>&1 &

# nohup python main.py -data fmnist -m dnn -algo HeurFedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.1 -xi 0.1 -did 1 -go dnn > fmnist_heurfedamp.out 2>&1 &
# nohup python main.py -data fmnist -m cnn -algo HeurFedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.1 -xi 0.1 -did 1 -go cnn > fmnist_heurfedamp1.out 2>&1 &

# ===============================================================(agnews)======================================================================  

# rm ../dataset/agnews/config.json
# cd ../dataset/
# nohup python generate_agnews.py noniid realworld > agnews_dataset.out 2>&1
# cd ../system/

# nohup python main.py -data agnews -m lstm -algo FedAvg -gr 5000 -did 0 -go lstm > agnews_fedavg.out 2>&1 &

# nohup python main.py -data agnews -m lstm -algo FedProx -gr 5000 -did 0 -mu 1 -go lstm > agnews_fedprox.out 2>&1 &

# nohup python main.py -data agnews -m lstm -algo pFedMe -gr 5000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go lstm > agnews_pfedme.out 2>&1 &

# nohup python main.py -data agnews -m lstm -algo PerAvg -gr 5000 -did 0 -bt 0.001 -go lstm > agnews_peravg.out 2>&1 &

# nohup python main.py -data agnews -m lstm -algo FedFomo -gr 5000 -M 5 -did 1 -go lstm > agnews_fedfomo.out 2>&1 &

# nohup python main.py -data agnews -m lstm -algo MOCHA -gr 5000 -itk 4000 -did 1 -go lstm > agnews_mocha.out 2>&1 &

# nohup python main.py -data agnews -m sep_lstm -algo FedPlayer -gr 5000 -did 1 -go lstm > agnews_fedplayer.out 2>&1 &

# nohup python main.py -data agnews -m lstm -algo FedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.01 -did 0 -go lstm > agnews_fedamp.out 2>&1 &

# nohup python main.py -data agnews -m lstm -algo HeurFedAMP -gr 5000 -alk 0.002 -lam 1.0 -sg 0.1 -xi 0.1 -did 1 -go lstm > agnews_heurfedamp.out 2>&1 &
