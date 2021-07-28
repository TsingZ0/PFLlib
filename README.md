# Personalized federated learning simulation platform with Non-IID dataset
The origin of the Non-IID phenomenon is the personalization of users, who generate the Non-IID data. With **Non-IID (Not Independent and Identically Distributed)** issue existing in the federated learning setting, a myriad of approaches have been proposed to crack this hard nut. In contrast, the personalized federated learning may take the advantage of the Non-IID data to learn the personalized model for each user. Thanks to [@Stonesjtu](https://github.com/Stonesjtu/pytorch_memlab/blob/d590c489236ee25d157ff60ecd18433e8f9acbe3/pytorch_memlab/mem_reporter.py#L185), this platform can also record the model size on GPU. 

## Environments
With the installed [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), we can run this platform in a conda virtual environment called *fl_torch*.
```
# current version
conda env create -f env.yml # for linux

# old version
cd ./system
conda env create -f env_linux.yml # for linux
# conda env create -f env_win.yml # for windows
```

## Datasets (updating)
Except for the **Synthetic** dataset, I currently using **six** famous datasets: **MNIST**, **Cifar10**, **Fashion-MNIST**, **Cifar10**, **AG_News** and **Sogou_News**, they can be easy split into **IID** and **Non-IID** version. Since some codes for generating dataset such as splitting are the same for all datasets, I move these codes into `./utils/dataset_utils.py`. Now it is easy to add other datasets into this FL platform. *If you need another data set, just write another code to download it and then using the utils.*

In **Non-IID** setting, there are three situations exist. The first one is the **extreme Non-IID** setting, the second one is **real-world Non-IID** setting and the third one is **feature skew Non-IID**. In the **extreme Non-IID** setting, for example, the data on each client only contains the specific number of labels (maybe only two labels), though the data on all clients contains 10 labels such as MNIST dataset. In the **real-world Non-IID** setting, the number of labels for each client is randomly chosen. In the **feature skew Non-IID**, specific Gaussian noise is added to each clients according to their IDs. 
- MNIST
    ```
    cd ./dataset
    python generate_mnist.py iid - - # for iid setting
    # python generate_mnist.py noniid - - # for extreme noniid setting
    # python generate_mnist.py noniid realworld - # for real-world noniid setting
    # python generate_mnist.py noniid realworld noise # for feature skew noniid setting
    ```
- Cifar10
    ```
    cd ./dataset
    python generate_cifar10.py iid - - # for iid setting
    # python generate_cifar10.py noniid - - # for extreme noniid setting
    # python generate_cifar10.py noniid realworld - # for real-world noniid setting
    # python generate_cifar10.py noniid realworld noise # for feature skew noniid setting
    ```
- Cifar100
    ```
    cd ./dataset
    python generate_cifar100py iid - - # for iid setting
    # python generate_cifar100.py noniid - - # for extreme noniid setting
    # python generate_cifar100.py noniid realworld - # for real-world noniid setting
    # python generate_cifar100.py noniid realworld noise # for feature skew noniid setting
    ```
- Fashion-MNIST
    ```
    cd ./dataset
    python generate_fmnist.py iid - - # for iid setting
    # python generate_fmnist.py noniid - - # for extreme noniid setting
    # python generate_fmnist.py noniid realworld - # for real-world noniid setting
    # python generate_fmnist.py noniid realworld noise # for feature skew noniid setting
    ```
- AG_News
    ```
    cd ./dataset
    python generate_agnews.py iid - - # for iid setting
    # python generate_agnews.py noniid - - # for extreme noniid setting
    # python generate_agnews.py noniid realworld - # for real-world noniid setting
    ```
- Sogou_News
    ```
    cd ./dataset
    python generate_sogounews.py iid - - # for iid setting
    # python generate_sogounews.py noniid - - # for extreme noniid setting
    # python generate_sogounews.py noniid realworld - # for real-world noniid setting
    ```
- Synthetic
    ```
    cd ./dataset
    python generate_synthetic.py iid # for iid setting
    # python generate_synthetic.py noniid # for extreme noniid setting
    ```

### Dataset generating examples
Output of `generate_mnist.py iid -`
```
Original number of samples of each label: [6903, 7877, 6990, 7141, 6824, 6313, 6876, 7293, 6825, 6958]

Client 0	 Size of data: 1064	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
Client 0	 Samples of labels:  [(0, 101), (1, 128), (2, 136), (3, 123), (4, 79), (5, 85), (6, 107), (7, 127), (8, 74), (9, 104)]
--------------------------------------------------
Client 1	 Size of data: 1023	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
Client 1	 Samples of labels:  [(0, 76), (1, 132), (2, 107), (3, 79), (4, 94), (5, 110), (6, 90), (7, 110), (8, 92), (9, 133)]
--------------------------------------------------
Client 2	 Size of data: 923	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
Client 2	 Samples of labels:  [(0, 136), (1, 89), (2, 84), (3, 88), (4, 78), (5, 124), (6, 120), (7, 66), (8, 69), (9, 69)]
--------------------------------------------------
```
<details>
    <summary>Show more</summary>

    Client 3	 Size of data: 906	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 3	 Samples of labels:  [(0, 73), (1, 151), (2, 94), (3, 73), (4, 83), (5, 67), (6, 133), (7, 92), (8, 69), (9, 71)]
    --------------------------------------------------
    Client 4	 Size of data: 1045	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 4	 Samples of labels:  [(0, 69), (1, 71), (2, 100), (3, 130), (4, 90), (5, 120), (6, 116), (7, 142), (8, 106), (9, 101)]
    --------------------------------------------------
    Client 5	 Size of data: 1026	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 5	 Samples of labels:  [(0, 128), (1, 90), (2, 71), (3, 135), (4, 71), (5, 88), (6, 91), (7, 139), (8, 116), (9, 97)]
    --------------------------------------------------
    Client 6	 Size of data: 1033	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 6	 Samples of labels:  [(0, 80), (1, 89), (2, 109), (3, 117), (4, 117), (5, 80), (6, 107), (7, 122), (8, 121), (9, 91)]
    --------------------------------------------------
    Client 7	 Size of data: 1043	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 7	 Samples of labels:  [(0, 65), (1, 86), (2, 132), (3, 133), (4, 111), (5, 110), (6, 65), (7, 106), (8, 120), (9, 115)]
    --------------------------------------------------
    Client 8	 Size of data: 1019	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 8	 Samples of labels:  [(0, 135), (1, 73), (2, 121), (3, 100), (4, 124), (5, 118), (6, 90), (7, 90), (8, 74), (9, 94)]
    --------------------------------------------------
    Client 9	 Size of data: 938	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 9	 Samples of labels:  [(0, 70), (1, 131), (2, 77), (3, 85), (4, 98), (5, 79), (6, 94), (7, 85), (8, 112), (9, 107)]
    --------------------------------------------------
    Client 10	 Size of data: 964	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 10	 Samples of labels:  [(0, 89), (1, 87), (2, 74), (3, 104), (4, 96), (5, 71), (6, 128), (7, 122), (8, 83), (9, 110)]
    --------------------------------------------------
    Client 11	 Size of data: 955	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 11	 Samples of labels:  [(0, 114), (1, 91), (2, 87), (3, 141), (4, 83), (5, 124), (6, 86), (7, 80), (8, 76), (9, 73)]
    --------------------------------------------------
    Client 12	 Size of data: 1015	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 12	 Samples of labels:  [(0, 84), (1, 101), (2, 71), (3, 113), (4, 131), (5, 78), (6, 116), (7, 101), (8, 89), (9, 131)]
    --------------------------------------------------
    Client 13	 Size of data: 856	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 13	 Samples of labels:  [(0, 82), (1, 121), (2, 88), (3, 111), (4, 88), (5, 77), (6, 67), (7, 75), (8, 80), (9, 67)]
    --------------------------------------------------
    Client 14	 Size of data: 1101	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 14	 Samples of labels:  [(0, 75), (1, 147), (2, 138), (3, 141), (4, 102), (5, 79), (6, 134), (7, 86), (8, 68), (9, 131)]
    --------------------------------------------------
    Client 15	 Size of data: 937	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 15	 Samples of labels:  [(0, 92), (1, 102), (2, 84), (3, 104), (4, 111), (5, 89), (6, 76), (7, 70), (8, 91), (9, 118)]
    --------------------------------------------------
    Client 16	 Size of data: 978	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 16	 Samples of labels:  [(0, 93), (1, 72), (2, 96), (3, 109), (4, 69), (5, 117), (6, 103), (7, 78), (8, 114), (9, 127)]
    --------------------------------------------------
    Client 17	 Size of data: 1016	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 17	 Samples of labels:  [(0, 78), (1, 96), (2, 76), (3, 80), (4, 127), (5, 84), (6, 112), (7, 139), (8, 132), (9, 92)]
    --------------------------------------------------
    Client 18	 Size of data: 1042	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 18	 Samples of labels:  [(0, 114), (1, 98), (2, 129), (3, 92), (4, 96), (5, 121), (6, 125), (7, 99), (8, 67), (9, 101)]
    --------------------------------------------------
    Client 19	 Size of data: 1178	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 19	 Samples of labels:  [(0, 132), (1, 74), (2, 124), (3, 109), (4, 106), (5, 122), (6, 134), (7, 127), (8, 122), (9, 128)]
    --------------------------------------------------
    Client 20	 Size of data: 948	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 20	 Samples of labels:  [(0, 77), (1, 87), (2, 88), (3, 131), (4, 130), (5, 85), (6, 77), (7, 96), (8, 76), (9, 101)]
    --------------------------------------------------
    Client 21	 Size of data: 917	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 21	 Samples of labels:  [(0, 73), (1, 79), (2, 66), (3, 130), (4, 94), (5, 114), (6, 100), (7, 113), (8, 66), (9, 82)]
    --------------------------------------------------
    Client 22	 Size of data: 1007	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 22	 Samples of labels:  [(0, 71), (1, 151), (2, 74), (3, 110), (4, 81), (5, 110), (6, 87), (7, 64), (8, 125), (9, 134)]
    --------------------------------------------------
    Client 23	 Size of data: 990	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 23	 Samples of labels:  [(0, 127), (1, 89), (2, 118), (3, 64), (4, 132), (5, 93), (6, 86), (7, 86), (8, 79), (9, 116)]
    --------------------------------------------------
    Client 24	 Size of data: 1137	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 24	 Samples of labels:  [(0, 125), (1, 135), (2, 134), (3, 93), (4, 128), (5, 108), (6, 130), (7, 134), (8, 76), (9, 74)]
    --------------------------------------------------
    Client 25	 Size of data: 1119	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 25	 Samples of labels:  [(0, 86), (1, 156), (2, 130), (3, 127), (4, 124), (5, 101), (6, 117), (7, 100), (8, 82), (9, 96)]
    --------------------------------------------------
    Client 26	 Size of data: 1059	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 26	 Samples of labels:  [(0, 121), (1, 138), (2, 135), (3, 139), (4, 81), (5, 86), (6, 73), (7, 82), (8, 94), (9, 110)]
    --------------------------------------------------
    Client 27	 Size of data: 1042	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 27	 Samples of labels:  [(0, 65), (1, 126), (2, 112), (3, 99), (4, 103), (5, 91), (6, 105), (7, 91), (8, 123), (9, 127)]
    --------------------------------------------------
    Client 28	 Size of data: 990	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 28	 Samples of labels:  [(0, 64), (1, 110), (2, 118), (3, 117), (4, 99), (5, 118), (6, 121), (7, 92), (8, 69), (9, 82)]
    --------------------------------------------------
    Client 29	 Size of data: 935	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 29	 Samples of labels:  [(0, 124), (1, 96), (2, 79), (3, 97), (4, 92), (5, 76), (6, 75), (7, 116), (8, 80), (9, 100)]
    --------------------------------------------------
    Client 30	 Size of data: 952	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 30	 Samples of labels:  [(0, 72), (1, 152), (2, 69), (3, 66), (4, 86), (5, 76), (6, 100), (7, 114), (8, 124), (9, 93)]
    --------------------------------------------------
    Client 31	 Size of data: 979	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 31	 Samples of labels:  [(0, 77), (1, 87), (2, 81), (3, 112), (4, 102), (5, 120), (6, 80), (7, 110), (8, 107), (9, 103)]
    --------------------------------------------------
    Client 32	 Size of data: 1034	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 32	 Samples of labels:  [(0, 111), (1, 119), (2, 106), (3, 118), (4, 105), (5, 123), (6, 94), (7, 71), (8, 95), (9, 92)]
    --------------------------------------------------
    Client 33	 Size of data: 1096	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 33	 Samples of labels:  [(0, 136), (1, 129), (2, 84), (3, 96), (4, 134), (5, 90), (6, 121), (7, 80), (8, 108), (9, 118)]
    --------------------------------------------------
    Client 34	 Size of data: 977	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 34	 Samples of labels:  [(0, 94), (1, 141), (2, 112), (3, 92), (4, 89), (5, 76), (6, 99), (7, 93), (8, 88), (9, 93)]
    --------------------------------------------------
    Client 35	 Size of data: 1015	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 35	 Samples of labels:  [(0, 135), (1, 67), (2, 86), (3, 119), (4, 112), (5, 71), (6, 105), (7, 75), (8, 126), (9, 119)]
    --------------------------------------------------
    Client 36	 Size of data: 871	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 36	 Samples of labels:  [(0, 67), (1, 64), (2, 77), (3, 95), (4, 114), (5, 87), (6, 66), (7, 125), (8, 85), (9, 91)]
    --------------------------------------------------
    Client 37	 Size of data: 1098	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 37	 Samples of labels:  [(0, 134), (1, 141), (2, 117), (3, 92), (4, 126), (5, 103), (6, 100), (7, 78), (8, 83), (9, 124)]
    --------------------------------------------------
    Client 38	 Size of data: 977	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 38	 Samples of labels:  [(0, 85), (1, 70), (2, 74), (3, 138), (4, 108), (5, 125), (6, 110), (7, 94), (8, 97), (9, 76)]
    --------------------------------------------------
    Client 39	 Size of data: 957	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 39	 Samples of labels:  [(0, 113), (1, 116), (2, 119), (3, 72), (4, 118), (5, 107), (6, 91), (7, 72), (8, 68), (9, 81)]
    --------------------------------------------------
    Client 40	 Size of data: 1109	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 40	 Samples of labels:  [(0, 121), (1, 149), (2, 125), (3, 96), (4, 64), (5, 76), (6, 136), (7, 104), (8, 103), (9, 135)]
    --------------------------------------------------
    Client 41	 Size of data: 993	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 41	 Samples of labels:  [(0, 67), (1, 134), (2, 120), (3, 72), (4, 80), (5, 114), (6, 92), (7, 112), (8, 131), (9, 71)]
    --------------------------------------------------
    Client 42	 Size of data: 987	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 42	 Samples of labels:  [(0, 132), (1, 66), (2, 85), (3, 141), (4, 83), (5, 102), (6, 66), (7, 94), (8, 98), (9, 120)]
    --------------------------------------------------
    Client 43	 Size of data: 972	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 43	 Samples of labels:  [(0, 88), (1, 140), (2, 89), (3, 114), (4, 73), (5, 91), (6, 77), (7, 87), (8, 98), (9, 115)]
    --------------------------------------------------
    Client 44	 Size of data: 1109	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 44	 Samples of labels:  [(0, 107), (1, 155), (2, 78), (3, 105), (4, 115), (5, 112), (6, 105), (7, 130), (8, 106), (9, 96)]
    --------------------------------------------------
    Client 45	 Size of data: 1035	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 45	 Samples of labels:  [(0, 90), (1, 85), (2, 77), (3, 128), (4, 74), (5, 125), (6, 100), (7, 128), (8, 102), (9, 126)]
    --------------------------------------------------
    Client 46	 Size of data: 1058	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 46	 Samples of labels:  [(0, 116), (1, 139), (2, 107), (3, 88), (4, 132), (5, 69), (6, 104), (7, 76), (8, 112), (9, 115)]
    --------------------------------------------------
    Client 47	 Size of data: 841	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 47	 Samples of labels:  [(0, 105), (1, 71), (2, 70), (3, 84), (4, 87), (5, 98), (6, 82), (7, 81), (8, 69), (9, 94)]
    --------------------------------------------------
    Client 48	 Size of data: 980	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 48	 Samples of labels:  [(0, 79), (1, 141), (2, 120), (3, 108), (4, 78), (5, 97), (6, 102), (7, 97), (8, 72), (9, 86)]
    --------------------------------------------------
    Client 49	 Size of data: 20754	 Labels:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
    Client 49	 Samples of labels:  [(0, 2155), (1, 2515), (2, 2142), (3, 1931), (4, 1926), (5, 1526), (6, 1981), (7, 2442), (8, 2208), (9, 1928)]
    --------------------------------------------------
    Total number of samples: 70000
    The number of train samples: [798, 767, 692, 679, 783, 769, 774, 782, 764, 703, 723, 716, 761, 642, 825, 702, 733, 762, 781, 883, 711, 687, 755, 742, 852, 839, 794, 781, 742, 701, 714, 734, 775, 822, 732, 761, 653, 823, 732, 717, 831, 744, 740, 729, 831, 776, 793, 630, 735, 15565]
    The number of test samples: [266, 256, 231, 227, 262, 257, 259, 261, 255, 235, 241, 239, 254, 214, 276, 235, 245, 254, 261, 295, 237, 230, 252, 248, 285, 280, 265, 261, 248, 234, 238, 245, 259, 274, 245, 254, 218, 275, 245, 240, 278, 249, 247, 243, 278, 259, 265, 211, 245, 5189]

    Finish generating dataset.
</details>
<br/>

Output of `generate_mnist.py noniid -`
```
Original number of samples of each label: [6903, 7877, 6990, 7141, 6824, 6313, 6876, 7293, 6825, 6958]

Client 0	 Size of data: 799	 Labels:  [0. 1.]
Client 0	 Samples of labels:  [(0, 141), (1, 658)]
--------------------------------------------------
Client 1	 Size of data: 687	 Labels:  [0. 1.]
Client 1	 Samples of labels:  [(0, 106), (1, 581)]
--------------------------------------------------
Client 2	 Size of data: 4649	 Labels:  [0. 1.]
Client 2	 Samples of labels:  [(0, 3903), (1, 746)]
--------------------------------------------------
```
<details>
    <summary>Show more</summary>

    Client 3	 Size of data: 853	 Labels:  [0. 1.]
    Client 3	 Samples of labels:  [(0, 213), (1, 640)]
    --------------------------------------------------
    Client 4	 Size of data: 826	 Labels:  [0. 1.]
    Client 4	 Samples of labels:  [(0, 350), (1, 476)]
    --------------------------------------------------
    Client 5	 Size of data: 1133	 Labels:  [0. 1.]
    Client 5	 Samples of labels:  [(0, 577), (1, 556)]
    --------------------------------------------------
    Client 6	 Size of data: 752	 Labels:  [0. 1.]
    Client 6	 Samples of labels:  [(0, 459), (1, 293)]
    --------------------------------------------------
    Client 7	 Size of data: 523	 Labels:  [0. 1.]
    Client 7	 Samples of labels:  [(0, 304), (1, 219)]
    --------------------------------------------------
    Client 8	 Size of data: 362	 Labels:  [0. 1.]
    Client 8	 Samples of labels:  [(0, 198), (1, 164)]
    --------------------------------------------------
    Client 9	 Size of data: 4196	 Labels:  [0. 1.]
    Client 9	 Samples of labels:  [(0, 652), (1, 3544)]
    --------------------------------------------------
    Client 10	 Size of data: 542	 Labels:  [2. 3.]
    Client 10	 Samples of labels:  [(2, 456), (3, 86)]
    --------------------------------------------------
    Client 11	 Size of data: 275	 Labels:  [2. 3.]
    Client 11	 Samples of labels:  [(2, 140), (3, 135)]
    --------------------------------------------------
    Client 12	 Size of data: 4615	 Labels:  [2. 3.]
    Client 12	 Samples of labels:  [(2, 500), (3, 4115)]
    --------------------------------------------------
    Client 13	 Size of data: 1322	 Labels:  [2. 3.]
    Client 13	 Samples of labels:  [(2, 630), (3, 692)]
    --------------------------------------------------
    Client 14	 Size of data: 930	 Labels:  [2. 3.]
    Client 14	 Samples of labels:  [(2, 523), (3, 407)]
    --------------------------------------------------
    Client 15	 Size of data: 701	 Labels:  [2. 3.]
    Client 15	 Samples of labels:  [(2, 333), (3, 368)]
    --------------------------------------------------
    Client 16	 Size of data: 1062	 Labels:  [2. 3.]
    Client 16	 Samples of labels:  [(2, 525), (3, 537)]
    --------------------------------------------------
    Client 17	 Size of data: 1134	 Labels:  [2. 3.]
    Client 17	 Samples of labels:  [(2, 696), (3, 438)]
    --------------------------------------------------
    Client 18	 Size of data: 707	 Labels:  [2. 3.]
    Client 18	 Samples of labels:  [(2, 611), (3, 96)]
    --------------------------------------------------
    Client 19	 Size of data: 2843	 Labels:  [2. 3.]
    Client 19	 Samples of labels:  [(2, 2576), (3, 267)]
    --------------------------------------------------
    Client 20	 Size of data: 880	 Labels:  [4. 5.]
    Client 20	 Samples of labels:  [(4, 347), (5, 533)]
    --------------------------------------------------
    Client 21	 Size of data: 878	 Labels:  [4. 5.]
    Client 21	 Samples of labels:  [(4, 663), (5, 215)]
    --------------------------------------------------
    Client 22	 Size of data: 3938	 Labels:  [4. 5.]
    Client 22	 Samples of labels:  [(4, 3553), (5, 385)]
    --------------------------------------------------
    Client 23	 Size of data: 1009	 Labels:  [4. 5.]
    Client 23	 Samples of labels:  [(4, 381), (5, 628)]
    --------------------------------------------------
    Client 24	 Size of data: 748	 Labels:  [4. 5.]
    Client 24	 Samples of labels:  [(4, 223), (5, 525)]
    --------------------------------------------------
    Client 25	 Size of data: 2630	 Labels:  [4. 5.]
    Client 25	 Samples of labels:  [(4, 449), (5, 2181)]
    --------------------------------------------------
    Client 26	 Size of data: 627	 Labels:  [4. 5.]
    Client 26	 Samples of labels:  [(4, 194), (5, 433)]
    --------------------------------------------------
    Client 27	 Size of data: 934	 Labels:  [4. 5.]
    Client 27	 Samples of labels:  [(4, 356), (5, 578)]
    --------------------------------------------------
    Client 28	 Size of data: 551	 Labels:  [4. 5.]
    Client 28	 Samples of labels:  [(4, 234), (5, 317)]
    --------------------------------------------------
    Client 29	 Size of data: 942	 Labels:  [4. 5.]
    Client 29	 Samples of labels:  [(4, 424), (5, 518)]
    --------------------------------------------------
    Client 30	 Size of data: 781	 Labels:  [6. 7.]
    Client 30	 Samples of labels:  [(6, 220), (7, 561)]
    --------------------------------------------------
    Client 31	 Size of data: 477	 Labels:  [6. 7.]
    Client 31	 Samples of labels:  [(6, 78), (7, 399)]
    --------------------------------------------------
    Client 32	 Size of data: 846	 Labels:  [6. 7.]
    Client 32	 Samples of labels:  [(6, 576), (7, 270)]
    --------------------------------------------------
    Client 33	 Size of data: 1180	 Labels:  [6. 7.]
    Client 33	 Samples of labels:  [(6, 616), (7, 564)]
    --------------------------------------------------
    Client 34	 Size of data: 4165	 Labels:  [6. 7.]
    Client 34	 Samples of labels:  [(6, 3623), (7, 542)]
    --------------------------------------------------
    Client 35	 Size of data: 885	 Labels:  [6. 7.]
    Client 35	 Samples of labels:  [(6, 637), (7, 248)]
    --------------------------------------------------
    Client 36	 Size of data: 3646	 Labels:  [6. 7.]
    Client 36	 Samples of labels:  [(6, 164), (7, 3482)]
    --------------------------------------------------
    Client 37	 Size of data: 1024	 Labels:  [6. 7.]
    Client 37	 Samples of labels:  [(6, 337), (7, 687)]
    --------------------------------------------------
    Client 38	 Size of data: 480	 Labels:  [6. 7.]
    Client 38	 Samples of labels:  [(6, 278), (7, 202)]
    --------------------------------------------------
    Client 39	 Size of data: 685	 Labels:  [6. 7.]
    Client 39	 Samples of labels:  [(6, 347), (7, 338)]
    --------------------------------------------------
    Client 40	 Size of data: 740	 Labels:  [8. 9.]
    Client 40	 Samples of labels:  [(8, 251), (9, 489)]
    --------------------------------------------------
    Client 41	 Size of data: 4175	 Labels:  [8. 9.]
    Client 41	 Samples of labels:  [(8, 299), (9, 3876)]
    --------------------------------------------------
    Client 42	 Size of data: 683	 Labels:  [8. 9.]
    Client 42	 Samples of labels:  [(8, 164), (9, 519)]
    --------------------------------------------------
    Client 43	 Size of data: 769	 Labels:  [8. 9.]
    Client 43	 Samples of labels:  [(8, 164), (9, 605)]
    --------------------------------------------------
    Client 44	 Size of data: 653	 Labels:  [8. 9.]
    Client 44	 Samples of labels:  [(8, 385), (9, 268)]
    --------------------------------------------------
    Client 45	 Size of data: 726	 Labels:  [8. 9.]
    Client 45	 Samples of labels:  [(8, 636), (9, 90)]
    --------------------------------------------------
    Client 46	 Size of data: 472	 Labels:  [8. 9.]
    Client 46	 Samples of labels:  [(8, 78), (9, 394)]
    --------------------------------------------------
    Client 47	 Size of data: 838	 Labels:  [8. 9.]
    Client 47	 Samples of labels:  [(8, 473), (9, 365)]
    --------------------------------------------------
    Client 48	 Size of data: 883	 Labels:  [8. 9.]
    Client 48	 Samples of labels:  [(8, 677), (9, 206)]
    --------------------------------------------------
    Client 49	 Size of data: 3844	 Labels:  [8. 9.]
    Client 49	 Samples of labels:  [(8, 3698), (9, 146)]
    --------------------------------------------------
    Total number of samples: 70000
    The number of train samples: [599, 515, 3486, 639, 619, 849, 564, 392, 271, 3147, 406, 206, 3461, 991, 697, 525, 796, 850, 530, 2132, 660, 658, 2953, 756, 561, 1972, 470, 700, 413, 706, 585, 357, 634, 885, 3123, 663, 2734, 768, 360, 513, 555, 3131, 512, 576, 489, 544, 354, 628, 662, 2883]
    The number of test samples: [200, 172, 1163, 214, 207, 284, 188, 131, 91, 1049, 136, 69, 1154, 331, 233, 176, 266, 284, 177, 711, 220, 220, 985, 253, 187, 658, 157, 234, 138, 236, 196, 120, 212, 295, 1042, 222, 912, 256, 120, 172, 185, 1044, 171, 193, 164, 182, 118, 210, 221, 961]

    Finish generating dataset.
</details>
<br/>

Output of `generate_mnist.py noniid realworld`
```
Original number of samples of each label: [6903, 7877, 6990, 7141, 6824, 6313, 6876, 7293, 6825, 6958]

Client 0         Size of data: 1059      Labels:  [1. 3. 4. 6. 8.]
Client 0         Samples of labels:  [(1, 71), (3, 98), (4, 228), (6, 577), (8, 85)]
--------------------------------------------------
Client 1         Size of data: 1138      Labels:  [2. 3. 4. 7. 8.]
Client 1         Samples of labels:  [(2, 198), (3, 138), (4, 201), (7, 515), (8, 86)]
--------------------------------------------------
Client 2         Size of data: 755       Labels:  [0. 1. 3. 7. 8.]
Client 2         Samples of labels:  [(0, 75), (1, 107), (3, 130), (7, 291), (8, 152)]
--------------------------------------------------
```
<details>
    <summary>Show more</summary>

    Client 3         Size of data: 875       Labels:  [1. 3. 5. 7.]
    Client 3         Samples of labels:  [(1, 254), (3, 74), (5, 160), (7, 387)]
    --------------------------------------------------
    Client 4         Size of data: 4228      Labels:  [0. 2. 4. 5. 7. 8.]
    Client 4         Samples of labels:  [(0, 77), (2, 276), (4, 173), (5, 483), (7, 3087), (8, 132)]
    --------------------------------------------------
    Client 5         Size of data: 800       Labels:  [0. 1. 2. 3. 4. 8.]
    Client 5         Samples of labels:  [(0, 140), (1, 269), (2, 120), (3, 94), (4, 77), (8, 100)]
    --------------------------------------------------
    Client 6         Size of data: 3286      Labels:  [0. 1. 2. 3. 4. 8.]
    Client 6         Samples of labels:  [(0, 2434), (1, 213), (2, 281), (3, 132), (4, 117), (8, 109)]
    --------------------------------------------------
    Client 7         Size of data: 413       Labels:  [2. 3. 4. 8.]
    Client 7         Samples of labels:  [(2, 160), (3, 80), (4, 87), (8, 86)]
    --------------------------------------------------
    Client 8         Size of data: 641       Labels:  [1. 3. 7. 8.]
    Client 8         Samples of labels:  [(1, 129), (3, 127), (7, 238), (8, 147)]
    --------------------------------------------------
    Client 9         Size of data: 3359      Labels:  [0. 2. 3. 6. 8.]
    Client 9         Samples of labels:  [(0, 132), (2, 263), (3, 69), (6, 2791), (8, 104)]
    --------------------------------------------------
    Client 10        Size of data: 461       Labels:  [0. 3. 4. 8.]
    Client 10        Samples of labels:  [(0, 171), (3, 96), (4, 103), (8, 91)]
    --------------------------------------------------
    Client 11        Size of data: 7555      Labels:  [0. 1. 3. 7. 9.]
    Client 11        Samples of labels:  [(0, 135), (1, 247), (3, 142), (7, 73), (9, 6958)]
    --------------------------------------------------
    Client 12        Size of data: 2435      Labels:  [0. 2. 3. 8.]
    Client 12        Samples of labels:  [(0, 160), (2, 88), (3, 138), (8, 2049)]
    --------------------------------------------------
    Client 13        Size of data: 883       Labels:  [3. 5. 7. 8.]
    Client 13        Samples of labels:  [(3, 64), (5, 267), (7, 417), (8, 135)]
    --------------------------------------------------
    Client 14        Size of data: 542       Labels:  [0. 1. 4. 8.]
    Client 14        Samples of labels:  [(0, 89), (1, 138), (4, 186), (8, 129)]
    --------------------------------------------------
    Client 15        Size of data: 1403      Labels:  [0. 1. 2. 3. 4. 5. 7. 8.]
    Client 15        Samples of labels:  [(0, 78), (1, 262), (2, 312), (3, 83), (4, 116), (5, 96), (7, 348), (8, 108)]
    --------------------------------------------------
    Client 16        Size of data: 990       Labels:  [0. 1. 3. 7. 8.]
    Client 16        Samples of labels:  [(0, 169), (1, 224), (3, 73), (7, 374), (8, 150)]
    --------------------------------------------------
    Client 17        Size of data: 296       Labels:  [2. 3. 8.]
    Client 17        Samples of labels:  [(2, 74), (3, 143), (8, 79)]
    --------------------------------------------------
    Client 18        Size of data: 242       Labels:  [0. 3.]
    Client 18        Samples of labels:  [(0, 114), (3, 128)]
    --------------------------------------------------
    Client 19        Size of data: 642       Labels:  [0. 1. 3. 4. 8.]
    Client 19        Samples of labels:  [(0, 151), (1, 94), (3, 88), (4, 159), (8, 150)]
    --------------------------------------------------
    Client 20        Size of data: 852       Labels:  [0. 3. 5. 8.]
    Client 20        Samples of labels:  [(0, 177), (3, 126), (5, 470), (8, 79)]
    --------------------------------------------------
    Client 21        Size of data: 2732      Labels:  [0. 1. 2. 3. 8.]
    Client 21        Samples of labels:  [(0, 73), (1, 140), (2, 248), (3, 2119), (8, 152)]
    --------------------------------------------------
    Client 22        Size of data: 1114      Labels:  [1. 3. 4. 6. 8.]
    Client 22        Samples of labels:  [(1, 66), (3, 89), (4, 134), (6, 719), (8, 106)]
    --------------------------------------------------
    Client 23        Size of data: 503       Labels:  [0. 4. 8.]
    Client 23        Samples of labels:  [(0, 143), (4, 214), (8, 146)]
    --------------------------------------------------
    Client 24        Size of data: 634       Labels:  [2. 3. 4. 5. 8.]
    Client 24        Samples of labels:  [(2, 180), (3, 115), (4, 162), (5, 70), (8, 107)]
    --------------------------------------------------
    Client 25        Size of data: 3779      Labels:  [0. 1. 2. 3. 4. 5. 7. 8.]
    Client 25        Samples of labels:  [(0, 76), (1, 192), (2, 205), (3, 108), (4, 2571), (5, 206), (7, 323), (8, 98)]
    --------------------------------------------------
    Client 26        Size of data: 1243      Labels:  [0. 1. 2. 3. 4. 6. 8.]
    Client 26        Samples of labels:  [(0, 158), (1, 116), (2, 141), (3, 92), (4, 152), (6, 472), (8, 112)]
    --------------------------------------------------
    Client 27        Size of data: 1092      Labels:  [0. 1. 3. 6. 8.]
    Client 27        Samples of labels:  [(0, 114), (1, 110), (3, 134), (6, 600), (8, 134)]
    --------------------------------------------------
    Client 28        Size of data: 494       Labels:  [0. 3. 6. 8.]
    Client 28        Samples of labels:  [(0, 69), (3, 81), (6, 229), (8, 115)]
    --------------------------------------------------
    Client 29        Size of data: 887       Labels:  [0. 1. 3. 6. 8.]
    Client 29        Samples of labels:  [(0, 80), (1, 267), (3, 112), (6, 336), (8, 92)]
    --------------------------------------------------
    Client 30        Size of data: 520       Labels:  [2. 3. 8.]
    Client 30        Samples of labels:  [(2, 269), (3, 105), (8, 146)]
    --------------------------------------------------
    Client 31        Size of data: 1619      Labels:  [0. 1. 2. 3. 4. 7. 8.]
    Client 31        Samples of labels:  [(0, 165), (1, 264), (2, 201), (3, 131), (4, 240), (7, 491), (8, 127)]
    --------------------------------------------------
    Client 32        Size of data: 846       Labels:  [0. 2. 3. 4. 8.]
    Client 32        Samples of labels:  [(0, 73), (2, 295), (3, 86), (4, 249), (8, 143)]
    --------------------------------------------------
    Client 33        Size of data: 1833      Labels:  [0. 1. 3. 4. 6. 7.]
    Client 33        Samples of labels:  [(0, 170), (1, 140), (3, 141), (4, 128), (6, 743), (7, 511)]
    --------------------------------------------------
    Client 34        Size of data: 1080      Labels:  [0. 1. 2. 3. 4. 6. 8.]
    Client 34        Samples of labels:  [(0, 92), (1, 84), (2, 160), (3, 145), (4, 94), (6, 409), (8, 96)]
    --------------------------------------------------
    Client 35        Size of data: 962       Labels:  [0. 1. 3. 5. 8.]
    Client 35        Samples of labels:  [(0, 84), (1, 215), (3, 106), (5, 407), (8, 150)]
    --------------------------------------------------
    Client 36        Size of data: 493       Labels:  [0. 2. 3. 8.]
    Client 36        Samples of labels:  [(0, 70), (2, 247), (3, 96), (8, 80)]
    --------------------------------------------------
    Client 37        Size of data: 468       Labels:  [0. 1. 3. 8.]
    Client 37        Samples of labels:  [(0, 128), (1, 141), (3, 124), (8, 75)]
    --------------------------------------------------
    Client 38        Size of data: 3961      Labels:  [0. 1. 3. 4. 8.]
    Client 38        Samples of labels:  [(0, 169), (1, 3440), (3, 83), (4, 204), (8, 65)]
    --------------------------------------------------
    Client 39        Size of data: 1104      Labels:  [0. 2. 3. 4. 5. 8.]
    Client 39        Samples of labels:  [(0, 148), (2, 89), (3, 124), (4, 148), (5, 443), (8, 152)]
    --------------------------------------------------
    Client 40        Size of data: 613       Labels:  [0. 1. 3. 4. 8.]
    Client 40        Samples of labels:  [(0, 139), (1, 70), (3, 102), (4, 167), (8, 135)]
    --------------------------------------------------
    Client 41        Size of data: 3678      Labels:  [0. 1. 3. 5. 8.]
    Client 41        Samples of labels:  [(0, 82), (1, 141), (3, 99), (5, 3292), (8, 64)]
    --------------------------------------------------
    Client 42        Size of data: 444       Labels:  [0. 2. 3. 8.]
    Client 42        Samples of labels:  [(0, 151), (2, 85), (3, 118), (8, 90)]
    --------------------------------------------------
    Client 43        Size of data: 955       Labels:  [0. 1. 3. 4. 5. 8.]
    Client 43        Samples of labels:  [(0, 150), (1, 177), (3, 81), (4, 214), (5, 255), (8, 78)]
    --------------------------------------------------
    Client 44        Size of data: 486       Labels:  [3. 4. 7. 8.]
    Client 44        Samples of labels:  [(3, 102), (4, 125), (7, 144), (8, 115)]
    --------------------------------------------------
    Client 45        Size of data: 523       Labels:  [0. 3. 4. 5.]
    Client 45        Samples of labels:  [(0, 65), (3, 147), (4, 147), (5, 164)]
    --------------------------------------------------
    Client 46        Size of data: 386       Labels:  [0. 1. 3. 8.]
    Client 46        Samples of labels:  [(0, 93), (1, 67), (3, 114), (8, 112)]
    --------------------------------------------------
    Client 47        Size of data: 794       Labels:  [0. 1. 3. 4. 7. 8.]
    Client 47        Samples of labels:  [(0, 136), (1, 100), (3, 150), (4, 233), (7, 94), (8, 81)]
    --------------------------------------------------
    Client 48        Size of data: 471       Labels:  [0. 3. 4.]
    Client 48        Samples of labels:  [(0, 173), (3, 103), (4, 195)]
    --------------------------------------------------
    Client 49        Size of data: 3431      Labels:  [1. 2. 3. 8.]
    Client 49        Samples of labels:  [(1, 139), (2, 3098), (3, 111), (8, 83)]
    --------------------------------------------------
    Total number of samples: 70000
    The number of train samples: [794, 853, 566, 656, 3171, 600, 2464, 309, 480, 2519, 345, 5666, 1826, 662, 406, 1052, 742, 222, 181, 481, 639, 2049, 835, 377, 475, 2834, 932, 819, 370, 665, 390, 1214, 634, 1374, 810, 721, 369, 351, 2970, 828, 459, 2758, 333, 716, 364, 392, 289, 595, 353, 2573]
    The number of test samples: [265, 285, 189, 219, 1057, 200, 822, 104, 161, 840, 116, 1889, 609, 221, 136, 351, 248, 74, 61, 161, 213, 683, 279, 126, 159, 945, 311, 273, 124, 222, 130, 405, 212, 459, 270, 241, 124, 117, 991, 276, 154, 920, 111, 239, 122, 131, 97, 199, 118, 858]

    Finish generating dataset.
</details>

## Models
- for MNIST and Fashion-MNIST

    1. Mclr_Logistic(1\*28\*28)
    2. LeNet()
    3. DNN(1\*28\*28, 100) # non-convex

- for Cifar10 and Cifar100

    1. Mclr_Logistic(3\*32\*32)
    2. CifarNet()
    3. DNN(3\*32\*32, 100) # non-convex
    4. ResNet18, Resnet50 and Resnet152

- for AG_News and Sogou_News

    1. LSTM()
    2. fastText() in [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
    3. TextCNN() in [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

- for Synthetic

    1. Mclr_Logistic(60)
    2. DNN(60, 20) # non-convex

## Algorithms (updating)
- **FedAvg** — [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) *AISTATS 2017*
- **Per-FadAvg** — [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf) *NeurIPS 2020*
- **pFedMe** — [Personalized Federated Learning with Moreau Envelopes](https://proceedings.neurips.cc/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf) *NeurIPS 2020*
- **FedProx** — [Federated Optimization for Heterogeneous Networks](https://openreview.net/pdf?id=SkgwE5Ss3N) *ICLR 2020*
- **FedFomo** — [Personalized Federated Learning with First Order Model Optimization](https://openreview.net/pdf?id=ehJqJQk9cw) *ICLR 2021*
- **MOCHA** — [Federated multi-task learning](https://arxiv.org/abs/1705.10467) *NIPS 2017*
- **FedPlayer** — [Federatedlearning with personalization layers](https://arxiv.org/abs/1912.00818)
- **FedAMP & HeurFedAMP** — [Personalized Cross-Silo Federated Learning on Non-IID Data](https://www.aaai.org/AAAI21Papers/AAAI-5802.HuangY.pdf) *AAAI 2021*


## How to start simulating 
- Build dataset: [Datasets](#datasets-updating)

- Train and evaluate the model:
    ```
    cd ./system
    python main.py -data mnist -m cnn -algo FedAvg -gr 2500 -did 0 -go cnn # for FedAvg and MNIST
    ```
    Or you can uncomment the lines you need in `./system/auto_train.sh` and run:
    ```
    cd ./system
    sh auto_train.sh
    ```

- Plot the result test accuracy and training loss curves and save to figures:
    ```
    python plot.py 
    ```
    Then check the figures in `./figures`.

Note: All the hyper-parameters have been tuned for all the algorithms, which are recorded in `./system/auto_train.sh`

## Practical setting
If you need to simulate FL in a practical setting, which include **client dropout**, **slow trainers**, **slow senders** and **network TTL**, you can set the following parameters to realize it.

- `-cdr`: The dropout rate for total clients. The selected clients will randomly drop at each training round.
- `-tsr` and `-ssr`: The rates for slow trainers and slow senders among all clients. Once a client was selected as "slow trainers", for example, it will always train slower than original one. So does "slow senders". 
- `-tth`: The threshold for network TTL (ms). 

## Easy to extend
This platform is easy to extend both dataset and algorithm. 

- To add a new dataset into this platform, all you need to do is writing the download code and using the utils the same as `./dataset/generate_mnist.py` (you can also consider it as the template). 

- To add a new algorithm, you can utilize the class **server** and class **client**, which are wrote in `./system/flcore/servers/serverbase.py` and `./system/flcore/clients/clientbase.py`, respectively. 

- To add a new model, just add it into `./system/flcore/trainmodel/models.py`.

- If you have your own optimizer while training, please add it into `./system/flcore/optimizers/fedoptimizer.py`
