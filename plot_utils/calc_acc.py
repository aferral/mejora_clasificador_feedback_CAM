import os
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np

#path= '/home/aferral/PycharmProjects/generative_supervised_data_augmentation/out_backprops/loss_v1_lambda_iterative_2_CONV_loss_06_May_2019__18_33'
#path = './out_backprops/loss_v1_lambda_iterative_1_CONV_loss_06_May_2019__18_20/' # path v1
path = '/home/aferral/PycharmProjects/generative_supervised_data_augmentation/out_backprops/loss_v2_few_FIN_cam_V2_loss_07_May_2019__14_43' # path v2
path = '/home/aferral/PycharmProjects/generative_supervised_data_augmentation/out_backprops/xray_V2_f_training_all_masks_V2_08_May_2019__19_00'

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator(os.path.join(path,'logs'))
event_acc.Reload()
# Show all tags in the log file
# print(event_acc.Tags())


# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, vals = zip(*event_acc.Scalars('loss_ce'))
plt.plot(np.arange(len(w_times)),vals,label='loss_ce')
w_times, step_nums, vals = zip(*event_acc.Scalars('loss_activacion_mean'))
plt.plot(np.arange(len(w_times)),vals,label='loss_act')
w_times, step_nums, vals = zip(*event_acc.Scalars('loss_total'))
plt.plot(np.arange(len(w_times)),vals,label='loss_total')
plt.legend()
plt.xlabel('Iteraciones')
plt.ylabel('Perdida')
plt.savefig('seleccion_loss_v2.png',dpi=100)
plt.show()



# plt indexs preds by iteration
path_preds = os.path.join(path,'indexs_pred.csv')

#it,0,index,n02120079_4409.JPEG,target,3,0,0.686551570892334,1,0.0715784952044487,2,0.027324171736836433,3,0.21454568207263947
x=pd.read_csv(path_preds,header=None)
# consigue todos los indices
inds = x[3].unique().tolist()
print(inds)
# selecciona las de un mismo indice
plt.clf()
for ind_x in inds:
    df_x = x[x[3] == ind_x]
    # consigue el target
    target = df_x.iloc[0][5]
    print("ind: {0} target: {1}".format(ind_x,target))

    col_target = 6+target*2 + 1

    # filtra todas las predicciones clase target
    iterations = df_x[1].values
    preds = df_x[col_target].values
    # grafica
    plt.plot(iterations, preds, label=ind_x)
plt.ylim(0,1)
plt.xlabel('Iteraciones')
plt.ylabel('Prediccion clase objetivo')
plt.legend()
plt.savefig('seleccion_pred_v2.png',dpi=100)
plt.show()

