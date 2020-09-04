# coding:utf-8
import sys
import numpy as np


def split(file,path):
    F = open(file, 'r')
    L = []

    for line in F:
        lst = line.strip()  # 移除字符串头尾指定的字符（默认为空格)
        L.append(lst)

    np.random.shuffle(L)  # 将序列的所有元素随机排序

    # 将L平均分成五份
    L1 = L[0:79]
    L2 = L[79:158]
    L3 = L[158:237]
    L4 = L[237:317]
    L5 = L[317:397]


    # 输出
    path1 = path + "/1"
    output = open(path1, 'w')
    for l in L1:
        output.write(l+"\n")
    output.close()

    path2 = path + "/2"
    output = open(path2, 'w')
    for l in L2:
        output.write(l + "\n")
    output.close()

    path3 = path + "/3"
    output = open(path3, 'w')
    for l in L3:
        output.write(l + "\n")
    output.close()

    path4 = path + "/4"
    output = open(path4, 'w')
    for l in L4:
        output.write(l + "\n")
    output.close()

    path5 = path + "/5"
    output = open(path5, 'w')
    for l in L5:
        output.write(l + "\n")
    output.close()


#两个参数，要切分的矩阵以及切分后存储的路径
if __name__ == '__main__':
    split(file=sys.argv[1], path=sys.argv[2])
