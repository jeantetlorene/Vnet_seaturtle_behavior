# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:13:48 2023

@author: ljeantet
"""

import os
import numpy as np
import pandas as pd
import datetime
import keras
import tensorflow as tf
import matplotlib.pyplot as plt


from Vnet_architecture_GT import *
from Training_helper_GT import *
from Global_generator_GT import*
from Prediction_GT import *


print(tf.__version__)

memo=pd.Series()

### I-Parameters
#Folders to load the preprocessed dataset and to save the model and its predictions
folder_out="c:/Users/loren/Documents/Articles/github/Vnet_seaturtle_behavior"
folder_in=memo['dir_in']="c:/Users/loren/Documents/Articles/github/Data_GT_preprocessed"


#name of the new folder that's going to be created to save the results 
now = datetime.datetime.now()
today=now.strftime("%Y_%m_%d")
folder_out=memo['dir_out']=folder_out+"model="+today+"_GreenTurtles"


#Split the individuals into training/validation/testing dataset
NAMES_TRAIN=memo["NAMES_TRAIN"]=[
                     'CC-07-48_18-02-2019',#10
                     'CC-07-48_06-10-2018',#6
                     'CC-07-48_04-10-2018',#5
                     'CC-07-115_13-05-2019',#3
                     'CC-07-115_14-05-2019',#3
                     'CC-07-48_08-04-2019_1', #7
                     'CC-07-48_08-04-2019_2', #7
                     'CC-07-48_08-04-2019_3', #7
                     'CC-07-48_22-05-2018', #121
                      'CC-07-108_11-02-2019', #

                    ]

NAMES_VAL=memo["NAMES_TEST"]=[
    'CC-07-59_20-05-2018', #13
    'CC-07-47_14-02-2018', #4
    'CC-07-48_10-10-2018'#9
    ]


NAMES_TEST=memo["NAMES_VAL"]=[
    'CC-07-48_08-10-2018',#8
    'CC-07-48_26-05-2018',#12
    'CC-07-107_11-02-2019' #1
    ]

NAMES_ALL=NAMES_TRAIN+NAMES_VAL+NAMES_TEST


#the Behavioral categories defined by classCLASS : id ,name of behavior ,behaviors number in the data csv, score_weight to calculate accuracy,sampling_weight=1,loss_weight=1
Other=classCLASS(0,"Other",None,0,1,0.1)
Breathing=classCLASS(1,"Breathing",[1,2,40],1)
Feeding=classCLASS(2,"Feeding",[3,4,5,6,7,11,12,13,18,19,20,21],10,2,2)
Gliding=classCLASS(3,"Gliding",[16,17],1,3)
Resting=classCLASS(4,"Resting",[29,30,31,32],1,0.5)
Scratching=classCLASS(5,"Scratching",[34,35,36],2,2,2)
Swimming=classCLASS(6,"Swimming",[42,45,48,43,46,49,27,44,47,50,56,52,54,55],1,0.5)

Behaviors=[Other,Breathing,Feeding,Gliding,Resting,Scratching,Swimming]

#save in the memo
for CLASS in Behaviors:
    memo["CLASSES_"+CLASS.name]=CLASS.__dict__


descriptor_select=memo['descriptor_select']=[0,1,2,3,4,5,23] #the variables that are going to be used to train the Vnet
label_column=7 #number of the column containing the labels
WINDOW_DURA=40 #size of the window in seconds
RESAMPLING_STEP_S=0.0500 #freq of the csv files
window_size=int(WINDOW_DURA/RESAMPLING_STEP_S) #size of the window and therefore input
freq=20 #frequency


for CLASS in Behaviors:
    memo["CLASSES_"+CLASS.name]=CLASS.__dict__

#model parameters
nb_outputs=len(Behaviors)
nb_inputs=len(descriptor_select)

model_depth=memo["model_depth"]=32
dropout=memo["dropout"]=0.5
kernel_size=memo['kernel_size']=5
activation_head='softmax'


#parameters for the training
batch_size=memo['batch_size']=32
lr_initial=memo['lr_initial']=0.0001

memo["nb_val_per_epoch"]=8000
memo["nb_train_per_epoch"]=13000


#### II-Load Data


train=trainer(folder_in, folder_out,  Behaviors, descriptor_select)
train.create_saving_folder()

dico=train.load_dico()
train.print_selected_variables(dico)

matrices, labels=train.load_data(NAMES_ALL, Behaviors, label_column, freq)
matrices=train.normalize(matrices)

train.histoOneIndividual(matrices[NAMES_ALL[5]])


##### III- Load Global generator

global_generator=Global_generator(NAMES_ALL, Behaviors, window_size, matrices,labels, descriptor_select)


train_gen=global_generator.oneEpoch_generator_window(NAMES_TRAIN,size = memo["nb_train_per_epoch"])
valid_gen=global_generator.oneEpoch_generator_window(NAMES_VAL,size = memo["nb_val_per_epoch"])


###### IV- Load Model
model=Vnet_4levels(window_size,nb_inputs,nb_outputs, 
             model_depth, dropout, kernel_size,activation_head)()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_initial),loss=train.dice_loss,metrics=["accuracy"])
model.summary()

#### V- Train

train.train(model, 5 ,batch_size, train_gen, valid_gen)

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

predict=Prediction(NAMES_ALL, model, window_size, matrices, labels, folder_out, Behaviors)
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
class_names=[CLASS.name for CLASS in Behaviors]
    
predict.plot_confusion_matrix(test_true,test_pred, np.array(class_names),True,"Confusion matrix on testing dataset")




