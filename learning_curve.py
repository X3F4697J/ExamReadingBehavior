import pickle
from matplotlib import pyplot as plt
import torch
import matplotlib.cm as cm
import numpy as np

model = 'lstm'
criteria = 'uid'
type = 'variance'


f = open('', 'rb')
curve = pickle.load(f)
f.close()
f = open('',  'rb')
normal_curve = pickle.load(f)
f.close()



nor_lo_mean = torch.tensor(normal_curve[0][0][0])
nor_lo_std = torch.tensor(normal_curve[0][0][0])
lo_std = torch.tensor(normal_curve[0][0]).std(dim=0)
nor_lo_pstd = nor_lo_mean + lo_std
nor_lo_sstd = nor_lo_mean - lo_std

nor_ac_mean = torch.tensor(normal_curve[0][1][0])
nor_ac_std = torch.tensor(normal_curve[0][1][0])
acc_std = torch.tensor(normal_curve[0][1]).std(dim=0)
nor_ac_pstd = nor_ac_mean + acc_std
nor_ac_sstd = nor_ac_mean - acc_std

e2h_lo_mean = torch.tensor(curve[0][0]).mean(dim=0)
e2h_lo_std = torch.tensor(curve[0][0]).std(dim=0)
e2h_lo_pstd = e2h_lo_mean + e2h_lo_std
e2h_lo_sstd = e2h_lo_mean - e2h_lo_std

e2h_ac_mean = torch.tensor(curve[0][1]).mean(dim=0)
e2h_ac_std = torch.tensor(curve[0][1]).std(dim=0)
e2h_ac_pstd = e2h_ac_mean + e2h_ac_std
e2h_ac_sstd = e2h_ac_mean - e2h_ac_std

h2e_lo_mean = torch.tensor(curve[4][0]).mean(dim=0)
h2e_lo_std = torch.tensor(curve[4][0]).std(dim=0)
h2e_lo_pstd = h2e_lo_mean + h2e_lo_std
h2e_lo_sstd = h2e_lo_mean - h2e_lo_std

h2e_ac_mean = torch.tensor(curve[4][1]).mean(dim=0)
h2e_ac_std = torch.tensor(curve[4][1]).std(dim=0)
h2e_ac_pstd = h2e_ac_mean + h2e_ac_std
h2e_ac_sstd = h2e_ac_mean - h2e_ac_std

nor_flag = False
e2h_flag = False
h2e_flag = False
for step, (nor_acc, e2h_acc, h2e_acc) in enumerate(zip(nor_ac_mean, e2h_ac_mean, h2e_ac_mean)):
    if (nor_acc >= 0.81) & (nor_flag == False):# set your performance cut point, this is different from task to task
        normal_step = step
        nor_flag = True
    if (e2h_acc >= 0.81) & (e2h_flag == False):
        e2h_step = step
        e2h_flag = True
    if (h2e_acc >= 0.81) & (h2e_flag == False):
        h2e_step = step
        h2e_flag = True

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

lns1 = ax2.plot(nor_lo_mean[0:normal_step], 'c', label='Loss Random')
ax2.fill_between(range(len(nor_lo_mean[0:normal_step])), nor_lo_sstd[0:normal_step], nor_lo_pstd[0:normal_step], color='c', alpha=0.2)

lns2 = ax1.plot(nor_ac_mean[0:normal_step], color='violet', label='Acc Random')
ax1.fill_between(range(len(nor_ac_mean[0:normal_step])), nor_ac_sstd[0:normal_step], nor_ac_pstd[0:normal_step], color='violet', alpha=0.2)
lns3 = ax2.plot(e2h_lo_mean[0:e2h_step], 'b', label='Loss Easy --> Hard')
ax2.fill_between(range(len(e2h_lo_mean[0:e2h_step])), e2h_lo_sstd[0:e2h_step], e2h_lo_pstd[0:e2h_step], color='b', alpha=0.2)

lns4 = ax1.plot(e2h_ac_mean[0:e2h_step], 'r', label='Acc Easy --> Hard')
ax1.fill_between(range(len(e2h_ac_mean[0:e2h_step])), e2h_ac_sstd[0:e2h_step], e2h_ac_pstd[0:e2h_step], color='r', alpha=0.2)

lns5 = ax2.plot(h2e_lo_mean[0:h2e_step], 'g', label='Loss Hard --> Easy')
ax2.fill_between(range(len(h2e_lo_mean[0:h2e_step])), h2e_lo_sstd[0:h2e_step], h2e_lo_pstd[0:h2e_step], color='g', alpha=0.2)

lns6 = ax1.plot(h2e_ac_mean[0:h2e_step], 'y', label='Acc Hard --> Easy')
ax1.fill_between(range(len(h2e_ac_mean[0:h2e_step])), h2e_ac_sstd[0:h2e_step], h2e_ac_pstd[0:h2e_step], color='y', alpha=0.2)

lns7 = ax1.plot(range(len(h2e_ac_mean[0:h2e_step])),np.zeros(len(h2e_ac_mean[0:h2e_step]))+0.81, color='crimson', label='Convergence Accuracy')#the cut point, this is different from task to task
lns8 = ax2.plot(range(len(h2e_ac_mean[0:h2e_step])),np.zeros(len(h2e_ac_mean[0:h2e_step]))+e2h_lo_mean[e2h_step].tolist(), color='chocolate', label='Convergence Loss')


lns = lns1+lns2+lns3+lns4+lns5+lns6+lns7
labs = [l.get_label() for l in lns]
ax1.set_xlabel('Training Iterations')
ax1.set_ylabel('Accuracy', fontsize=13)
ax2.set_ylabel('Loss', fontsize=13)
ax1.legend(loc='center right')
ax1.legend(lns, labs, loc='center right')
ax1.grid()
ax1.set_title('Performance and Loss of different training order')
plt.savefig('schedule.jpg', dpi=300)
plt.show()
