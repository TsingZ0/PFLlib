nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 500 -did 0 -go cnn -cstart 10 -crestart 3 -camou 1 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)_Cifar10_cnn_Camouflaged.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 500 -did 1 -go cnn -cstart 10 -crestart 3 -camou 1 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)_Cifar10_cnn_Camouflaged_10images.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 500 -did 1 -go cnn -cstart 10 -crestart 3 -camou 1 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)_Cifar10_cnn_Camouflaged_10images.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 500 -did 1 -go cnn -cstart 10 -crestart 3 -camou 1 -cimages 5 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)_Cifar10_cnn_Camouflaged_5images.txt 2>&1 &

nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 500 -did 2 -go cnn -cstart 10 -crestart 3 -camou 0 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-cnn-PoisonOnly-10images.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 500 -did 2 -go cnn -cstart 10 -crestart 3 -camou 0 -cimages 5 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-cnn-PoisonOnly-5images.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 500 -did 2 -go cnn -cstart 10 -crestart 3 -camou 0 -cimages 1 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-cnn-PoisonOnly-1images.txt 2>&1 &


nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 200 -did 3 -go cnn -cstart 10 -crestart 3 -camou 1 -cimages 20 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)_Cifar10_cnn_Camouflaged_20images.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 200 -did 3 -go cnn -cstart 10 -crestart 3 -camou 0 -cimages 20 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-cnn-PoisonOnly-20images.txt 2>&1 &

nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 200 -did 3 -go cnn -cstart 10 -crestart 3 -camou 1 -cimages 40 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)_Cifar10_cnn_Camouflaged_40images.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 200 -did 3 -go cnn -cstart 10 -crestart 3 -camou 0 -cimages 40 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-cnn-PoisonOnly-40images.txt 2>&1 &


nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 200 -did 3 -go cnn -cstart 10 -crestart 3 -camou 1 -cimages 10 -def TrimmedMean -t 10 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-cnn-Camouflaged-10images-TrimmedMean-times10.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 200 -did 3 -go cnn -cstart 10 -crestart 3 -camou 0 -cimages 10 -def TrimmedMean -t 10 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-cnn-PoisonOnly-10images-TrimmedMean-times10.txt 2>&1 &

nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 200 -did 2 -go cnn -cstart 10 -crestart 3 -camou 1 -cimages 10 -t 10 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-cnn-Camouflaged-10images-NoDefense-times10.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 200 -did 2 -go cnn -cstart 10 -crestart 3 -camou 0 -cimages 10 -t 10 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-cnn-PoisonOnly-10images-NoDefense-times10.txt 2>&1 &

nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m resnet -algo Camouflaged_FedAvg -gr 200 -did 1 -go resnet -cstart 10 -crestart 3 -camou 1 -cimages 10 -def TrimmedMean -t 10 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-Resnet18-Camouflaged-10images-TrimmedMean-times10.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m resnet -algo Camouflaged_FedAvg -gr 200 -did 2 -go resnet -cstart 10 -crestart 3 -camou 0 -cimages 10 -def TrimmedMean -t 10 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-Resnet18-PoisonOnly-10images-TrimmedMean-times10.txt 2>&1 &

nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m googlenet -algo Camouflaged_FedAvg -gr 200 -did 0 -go googlenet -cstart 10 -crestart 3 -camou 1 -cimages 10 -def TrimmedMean -t 2 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-GoogleNet-Camouflaged-10images-TrimmedMean-times2.txt 2>&1 &
nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m googlenet -algo Camouflaged_FedAvg -gr 200 -did 0 -go googlenet -cstart 10 -crestart 3 -camou 0 -cimages 10 -def TrimmedMean -t 2 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-GoogleNet-PoisonOnly-10images-TrimmedMean-times2.txt 2>&1 &


nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m resnet -algo Camouflaged_FedAvg -gr 100 -did 0 -go resnet -cstart 10 -crestart 3 -camou 0 -cimages 10 -def TrimmedMean -cattackiter 100 -t 2 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-Resnet18-PoisonOnly-10images-TrimmedMean-times2.txt 2>&1 &


