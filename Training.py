# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:13:48 2023

@author: ljeantet
"""

import numpy as np
import pandas as pd
import datetime
import keras
import tensorflow as tf

import matplotlib.pyplot as plt

from Vnet_architecture import *
from Training_helper import *
from Global_generator import*
from Prediction import *


print("          ")
print(tf.__version__)

memo=pd.Series()

### I-Parameters

#data
folder_out='D:/Chinstrap_data/Out_model/'
folder_in="C:/Users/ljeantet/Documents/Postdoc/Manchots_Accelero/model_output/preprocessed"

folder_out='Out_model/'
folder_in=memo['dir_in']='preprocessed'


now = datetime.datetime.now()
today=now.strftime("%Y_%m_%d")
folder_out=memo['dir_out']=folder_out+"model="+today+"_"+CASE

#only accelero
NAMES_TRAIN=memo["NAMES_TRAIN"]=[
                     'CC-07-48_18-02-2019', #1
                     'CC-07-48_06-10-2018', #2
                     'CC-07-48_04-10-2018', #3
                     'CC-07-115_13-05-2019', #4
                     'CC-07-115_14-05-2019', #4
                     'CC-07-48_08-04-2019_1', #5
                     'CC-07-48_08-04-2019_2', #5
                     'CC-07-48_08-04-2019_3', #5
                     'CC-07-48_22-05-2018', #6
                      'CC-07-108_11-02-2019', #7


                    ]

NAMES_VAL=memo["NAMES_VAL"]=[
    'CC-07-59_20-05-2018', #8
    'CC-07-47_14-02-2018', #9
    'CC-07-48_10-10-2018',  #10

    ]

NAMES_TEST=memo["NAMES_TEST"]=[
    'CC-07-48_26-05-2018', #11
    'CC-07-107_11-02-2019', #12
    'CC-07-48_08-10-2018', #13
    ]


NAMES_ALL=NAMES_TRAIN+NAMES_VAL+NAMES_TEST


Other=classCLASS(0,"Other",None,0,1,0.1)
Breathing=classCLASS(1,"Breathing",[1,2,40],1)
Feeding=classCLASS(2,"Feeding",[3,4,5,6,7,11,12,13,18,19,20,21],10,2,2)
Gliding=classCLASS(3,"Gliding",[16,17],1,3)
Resting=classCLASS(4,"Resting",[29,30,31,32],1,0.5)
Scratching=classCLASS(5,"Scratching",[34,35,36],2,2,2)
Swimming=classCLASS(6,"Swimming",[42,45,48,43,46,49,27,44,47,50,56,52,54,55],1,0.5)

Classes_beh=[Other, Breathing, Feeding, Gliding, Resting, Scratching, Swimming]


label_column=7
window_dura=40 #size of the window in seconds =>has to be mutilple of 8
freq=25 #in Hz
descriptor_select=memo['descriptor_select']=[0,1,2,3,4,5,9]
window_size=int(window_dura*freq)


for beh in Classes_beh:
    memo["Behavior_"+beh.name]=beh.__dict__

#model
nb_outputs=len(Classes_beh)
nb_inputs=len(descriptor_select)

model_depth=memo["model_depth"]=32
dropout=memo["dropout"]=0.5
kernel_size=memo['kernel_size']=5
activation_head='softmax'



batch_size=memo['batch_size']=200
lr_initial=memo['lr_initial']=0.01
proba=0.3

#training
nb_val_epoch=memo["nb_val_per_epoch"]=10000
nb_train_epoch=memo["nb_train_per_epoch"]=20000


#### II-Load Data


train=trainer(folder_in, folder_out,  Classes_beh, descriptor_select)
train.create_saving_folder()

dico=train.load_dico()
train.print_selected_variables(dico)

matrices, labels=train.load_data(NAMES_ALL, label_column, Classes_beh,  freq)
matrices=train.normalize(matrices)

##### III- Load Global generator

global_generator=Global_generator(NAMES_ALL, Classes_beh, window_size, matrices,labels, descriptor_select, proba)


train_gen=global_generator.oneEpoch_generator_window(NAMES_TRAIN, size = nb_train_epoch)
valid_gen=global_generator.oneEpoch_generator_window(NAMES_VAL, size = nb_val_epoch)


###### IV- Load Model
model=Vnet_4levels(window_size,nb_inputs,nb_outputs, 
             model_depth, dropout, kernel_size,activation_head)()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_initial),loss=train.dice_loss,metrics=["accuracy"])
model.summary()

#### V- Train

train.train(model, 20 ,batch_size, train_gen, valid_gen)

##### VI-Save results

train.plot_training()

print("Training took {:.2f} seconds".format(train.total_duration))
print("Which is {:.2f} minutes".format(train.total_duration/60))

print(train.__dict__)
memo["trainer"]=train.__dict__
memo["trainer.epoch_count"]=train.epoch_count
memo["trainer.best_epoch"]=train.best_epoch
memo["trainer.best_value"]=train.best_value
memo["trainer.total_duration"]=train.total_duration



#memo.to_pickle(folder_out+"/memo")
memo.to_csv(folder_out+"/memo.csv")


######VII- Predictions & PLot

predict=Prediction(NAMES_ALL, model, window_size, matrices, labels, folder_out, Classes_beh)
all_pred,all_true=predict.get_all()

predict.plot_compa(NAMES_TRAIN,all_true,all_pred,"Training dataset")
predict.plot_compa(NAMES_VAL,all_true,all_pred,"Validation dataset")
predict.plot_compa(NAMES_TEST,all_true,all_pred,"Testing dataset")


test_true=[]
test_pred=[]
for name in NAMES_TEST:
    test_true.append(all_true[name])
    test_pred.append(all_pred[name])

test_true=np.concatenate(test_true)
test_pred=np.concatenate(test_pred)
class_names=[beh.name for beh in Classes_beh]
    
predict.plot_confusion_matrix(test_true,test_pred, np.array(class_names),True,"Confusion matrix on testing dataset")




