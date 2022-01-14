import pickle
import numpy
import torch
import matplotlib.pyplot as plt

#caculate metric based on pkl file
l1 = ["roc", "sst2", "mrpc", "qnli", "rte", "stsb"]
l2 = ["flesch", "uid_variance", "uid_super-linear", "neural"]
l1_1 = {"roc": 0.95, "sst2": 0.98, "qnli": 0.99, "rte": 0.95, "stsb": 0.75, "mrpc": 0.99}
for tmp1 in l1:
    for tmp2 in l2:
        tmp = '' + tmp1 + '_' + tmp2 + '.pkl'
        print(tmp)
        f = open(tmp, 'rb')
        curve = pickle.load(f)
        for idd, para1 in enumerate(curve):
            maximum_list = []
            percent_list = []
            for para2 in para1[1]:
                maximum = numpy.max(para2)
                arg_max = numpy.argmax(para2)
                tmp_idx = 0
                for idx, para3 in enumerate(para2):
                    if para3 * 1.0 >= maximum * l1_1[tmp1]:
                        tmp_idx = idx
                        break
                maximum_list.append(maximum.item())
                percent_list.append(tmp_idx*1.0/(arg_max.item()+1))
            a1 = numpy.average(numpy.array(maximum_list)).item() * 100
            a2 = numpy.std(numpy.array(maximum_list), ddof=1).item() * 100
            a3 = numpy.average(numpy.array(percent_list)).item() * 100
            a4 = numpy.std(numpy.array(percent_list), ddof=1).item() * 100
            print("%.2f(%.2f)/%.2f(%.2f)" % (a1, a2, a3, a4))
            #print(str(numpy.std(numpy.array(maximum_list), ddof=1)) + "---" + str(numpy.average(numpy.array(maximum_list))))
            #print(str(numpy.std(numpy.array(aaaaaaa_list), ddof=1)) + "***" + str(numpy.average(numpy.array(aaaaaaa_list))))
        f.close()
