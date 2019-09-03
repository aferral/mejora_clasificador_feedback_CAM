import numpy as np
import os
def tf(t):
    a=list(map(lambda x : x.strip().replace('[','').replace(']','').split(),t.split(']')))
    a=filter(lambda x : len(x) > 0, a)
    a=map(lambda x : [int(y) for y in x], a)
    m=np.array(a)
    rows,cols = m.shape

    nm = (m*1.0 / m.sum(axis=0)).round(3)
    diag = nm[np.eye(4).astype(np.bool)]
    clase_dificiles = diag.argsort()
    nodiag=nm.copy()
    nodiag[np.eye(4).astype(np.bool)] = 0 
    confusion=list(reversed([(i%cols,i/rows) for i in nodiag.flatten().argsort()]))

    diag = m[np.eye(4).astype(np.bool)]
    clase_dificiles_normal = diag.argsort()
    nodiag=m.copy()
    nodiag[np.eye(4).astype(np.bool)] = 0 
    confusion_normal=list(reversed([(i%cols,i/rows) for i in nodiag.flatten().argsort()]))

    for i in range(rows):
       prc = m[i,i] *1.0/ np.sum(m[:,i])
       recl= m[i,i] *1.0/ np.sum(m[i,:])
       f1  = 2.0* (prc*recl)/(prc+recl) 
       print('Class {0} prec: {1} recall: {2} f1: {3}'.format(i,prc,recl,f1))

    print(nm)
    print('')
    print('Clases dificiles : {0}'.format(clase_dificiles_normal))
    print('Confusion  : {0}'.format(confusion_normal[0:4]))
    print('')
    print('Clases dificiles NORM: {0}'.format(clase_dificiles))
    print('Confusion NORM : {0}'.format(confusion[0:4]))
    print('')



folder = 'exp_cam_loss_using_missclass_cont_training_big_sample30_missclass_27_Jan_2019__13_42'

eval_txt = os.path.join(folder,'eval.txt')
st_folder = os.path.join(folder,'start')
en_folder = os.path.join(folder,'end')

# open eval
with open(eval_txt,'r') as f:
    eval_str = f.read()

all_blocks = eval_str.split('\n\n')


# open start, end,
with open(os.path.join(st_folder,'conf_matrix.txt'),'r') as f:
    start_str = f.read()

with open(os.path.join(en_folder,'conf_matrix.txt'),'r') as f:
    end_str = f.read()


val_blocks= all_blocks[0::2]
f_val_b = "\n".join(val_blocks[0].split('\n')[1:])
final_val_b = "\n".join(val_blocks[-2].strip().split('\n')[3:])


test_blocks= all_blocks[1::2]
f_test_b = "\n".join(val_blocks[0].split('\n')[1:])
final_test_b = "\n".join(val_blocks[-2].strip().split('\n')[3:])



start_b = "\n".join(start_str.split('\n')[1:])



end_b = "\n".join(end_str.split('\n')[1:])


print("Results for {0}".format(folder))
print(" ")
print(" ")
print("Train st")
tf(start_b)
print("Train en")
tf(end_b)
print(" ")
print(" ")


print("Val st")
tf(f_val_b)
print("Val en")
tf(final_val_b)
print(" ")
print(" ")

print("Test st")
tf(f_test_b)
print("Test end")
tf(final_test_b)
print(" ")
print(" ")





