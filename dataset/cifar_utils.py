from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import torch
import numpy as np


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        img_norm = test_transform(img) #这里不能直接返回 img，而是要返回归一化后的 img
        if self.target_transform is not None:
                target = self.target_transform(target)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            return pos_1, pos_2, img_norm, target
        else:
            return img_norm, target

class CIFAR100Pair(CIFAR100):
    """CIFAR100 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        img_norm = test_transform_100(img) #这里不能直接返回 img，而是要返回归一化后的 img
        if self.target_transform is not None:
                target = self.target_transform(target)
        if self.train:
            if self.transform is not None:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)
            data = [pos_1, pos_2, img_norm]
            return data, target
        else:
            return img_norm, target


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_transform_100 = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])

test_transform_100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])

if __name__ == "__main__":
    # data = CIFAR10(root="./data", train=True, download=True)
    image = Image.open("/home/leon/workspace/pFedSD/pcode/datasets/data/1.jpg")
    path = "tmp/"
    imgs = []
    
    for i in range(10):
        img = train_transform(image)
        img.save("tmp/"+str(i)+".png")
    # for i,(p1,p2,img,target) in enumerate(data):
    #     # 将PyTorch张量转换为NumPy数组
    #     d1 = p1.permute(1, 2, 0).byte().numpy()
    #     d2 = p2.permute(1, 2, 0).byte().numpy()
    #     d = img.permute(1, 2, 0).byte().numpy()
    #     # 创建PIL图像对象
    #     img1 = Image.fromarray(d1)
    #     img2 = Image.fromarray(d2)
    #     img = Image.fromarray(d)

    #     # 保存图像
    #     img1.save(path+str(i)+"img1.png")
    #     img2.save(path+str(i)+"img2.png")
    #     img.save(path+str(i)+"img.png")
        
    #     if i == 10: break
