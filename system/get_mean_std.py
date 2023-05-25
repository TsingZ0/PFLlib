from statistics import mean
import numpy as np

file_name = input() + '.out'

acc = []

with open(file_name, 'r') as f:
    is_best = False
    for l in f.readlines():
        if is_best:
            acc.append(float(l))
            is_best = False
        elif 'Best accuracy' in l:
            is_best = True

print(acc)
print(mean(acc)*100, np.std(acc)*100)
