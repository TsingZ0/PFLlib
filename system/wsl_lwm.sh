
#nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 5 -jr 1 -nb 10 -data Cifar10 -m resnet -algo Camouflaged_FedAvg -gr 50 -did 0 -go resnet -cstart 10 -crestart 3 -camou 0 -cimages 10 -cattackiter 100 -ceps 32 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-Resnet18-PoisonOnly-10images-NoDefences-att100-SGD-ceps32.txt 2>&1 &
#wait
#nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 5 -jr 1 -nb 10 -data Cifar10 -m resnet -algo Camouflaged_FedAvg -gr 50 -did 0 -go resnet -cstart 10 -crestart 3 -camou 0 -cimages 10 -cattackiter 100 -ceps 8 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-Resnet18-PoisonOnly-10images-NoDefences-att50-SGD-ceps8.txt 2>&1 &
#wait
#nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 5 -jr 1 -nb 10 -data Cifar10 -m resnet -algo Camouflaged_FedAvg -gr 50 -did 0 -go resnet -cstart 10 -crestart 3 -camou 0 -cimages 10 -cattackiter 100 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-Resnet18-PoisonOnly-10images-NoDefences-att100-SGD-ceps16.txt 2>&1 &
#wait
#nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 5 -jr 1 -nb 10 -data Cifar10 -m resnet -algo Camouflaged_FedAvg -gr 50 -did 0 -go resnet -cstart 10 -crestart 3 -camou 0 -cimages 10 -cattackiter 150 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-Resnet18-PoisonOnly-10images-NoDefences-att150.txt 2>&1 &
#wait
#nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 5 -jr 1 -nb 10 -data Cifar10 -m resnet -algo Camouflaged_FedAvg -gr 50 -did 0 -go resnet -cstart 10 -crestart 3 -camou 0 -cimages 10 -cattackiter 200 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-Resnet18-PoisonOnly-10images-NoDefences-att200.txt 2>&1 &
#wait
#nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 5 -jr 1 -nb 10 -data Cifar10 -m resnet -algo Camouflaged_FedAvg -gr 50 -did 0 -go resnet -cstart 0 -crestart 3 -camou 0 -cimages 10 -cattackiter 250 -ceps 32 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-Resnet18-PoisonOnly-10images-NoDefences-att250-SGD-ceps32.txt 2>&1 &

nohup python -u main.py -lbs 32 -nc 20 -lr 0.01 -ls 5 -jr 1 -nb 10 -data Cifar10 -m resnet -algo Camouflaged_FedAvg -gr 100 -did 0 -go resnet -cstart 0 -crestart 3 -camou 0 -cimages 10 -cattackiter 400 -ceps 16 > ../results/Camouflaged_FedAvg/logs/$(date +%Y%m%d-%H%M%S)-Cifar10-Resnet18-PoisonOnly-10images-NoDefences-att400-SGD-ceps16.txt 2>&1 &

