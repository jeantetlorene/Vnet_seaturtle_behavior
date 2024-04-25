# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:23:46 2023

@author: ljeantet
"""
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import time
import json

class classCLASS:

    def __init__(self,id,name,behaviors,score_weight,sampling_weight=1,loss_weight=1):
        self.id=id
        self.name=name
        self.behaviors=behaviors
        self.score_weight=score_weight
        self.sampling_weight=sampling_weight
        self.loss_weight=loss_weight

class trainer:
    def __init__(self, folder_in, folder_out, Behaviors, descriptor_select):
        
        self.dir_in=folder_in
        self.dir_out=folder_out
        self.descriptor_select=descriptor_select

                
        self.model_name=self.dir_out+"/model.json"
        self.weights_name=self.dir_out+"/model.h5"
        self.monitor="val_loss"
         
        self.ponde=np.array([CLASS.loss_weight for CLASS in Behaviors],dtype=np.float32)

        self.ponde/=np.sum(self.ponde)
        

        # training info
        self.history = None
        self.Nb_epochs = 10
        self.best_epoch = 0
        self.durations = []
        self.total_duration = 0
        self.loss_history = []
        self.val_loss_history = []
        self.acc_history = []
        self.val_acc_history = []
        #self.nb_models = 8

        # evaluation info
        self.score = None
        self.y_true = None
        self.y_pred = None


        self.best_epoch = 0    
        self.best_value = 1e10
        self.best_epochs=[]
        self.best_values=[]

        self.epoch_count=-1
        
    def load_data(self, names, categories, beh_class, freq):

        matrices={}
        labels={}

        for name in names:
            full_name=self.dir_in+"/"+name+".npy"
            if not os.path.exists(full_name):
                assert 1==0,"the file:"+full_name+" cannot be found"
            else:
                full_mat=np.load(full_name, allow_pickle=True)
                pre_label=full_mat[:, beh_class]
            
                label=np.zeros_like(pre_label)
                for CLASS in categories:
                    if CLASS.name!="Other":
                        for beh in CLASS.behaviors:
                            label[pre_label==beh]=CLASS.id

                #np.save(self.dir_out+"/"+name[0:-3]+"=label",label)
                mat=full_mat[:,self.descriptor_select]
                print("loading the matrix:"+name)
                print("shape:",mat.shape)
                print("duration in seconds:",len(mat)/freq)
                print("duration in hour:",round(len(mat)/freq/60/60,2))
                print()
                matrices[name]=mat
                labels[name]=label
            
        return matrices,labels
        
        
    def load_dico(self, key_int=False, print_me=False):
        with open(self.dir_in+"/dico_info.json") as f:
                dico_str = json.loads(json.load(f))
    
   
        if key_int: 
            conv_key=lambda k:int(k)
        else:
            conv_key=lambda k:k
        
        dico={conv_key(k):v for k,v in dico_str.items()}
    
        if print_me:
            print(dico)
        
        return dico  
    
    def print_selected_variables(self, dico):
        
        va_names=dico['col_names']
        va_names_sel=[]
        for i,name in enumerate(va_names):
            if i in self.descriptor_select:
                prefix=""
                va_names_sel.append(name)
            else:
                prefix="-----suppressed----->"
            print(prefix,i,name)
        
        
    
    def create_saving_folder(self):
   
        if os.path.exists(self.dir_out):
            shutil.rmtree(self.dir_out)
            print("we clear the directory:",self.dir_out)
        else:
            print("we create the directory:",self.dir_out)
    
        """creation of the folder """
        os.makedirs(self.dir_out)
        
    def normalize(self, matrices):

        nb_col=len(self.descriptor_select)
        for matrix in matrices.values():
            for j in range(nb_col):
                matrix[:,j]-=matrix[:,j].mean()
                matrix[:,j]/=matrix[:,j].std()
        
        return matrices
    
    
    def histoOneIndividual(self, matrix):
        nb_col=matrix.shape[1]
        fig,axs=plt.subplots(1,nb_col,figsize=(nb_col*3,2))
        for j in range(nb_col):
            axs[j].hist(matrix[:,j],bins=40,edgecolor='k')
            axs[j].set_title(self.descriptor_select[j], color="coral")
            axs[j].tick_params(colors='coral')
        plt.show()
        
    def dice_coef_perCat(self, y_true, y_pred,smooth=1e-8):
        intersection = tf.reduce_sum(y_true * y_pred,axis=[0])
        dice=(2. * intersection + smooth) / (tf.reduce_sum(y_true,axis=[0]) + tf.reduce_sum(y_pred,axis=[0]) + smooth)
        "at the end, we average on the categories"
        return dice
    

    def dice_loss(self, y_true, y_pred):
        smooth=1e-8
    
    
        "we average the dices by categories"
        return 1-tf.reduce_mean(self.dice_coef_perCat(y_true, y_pred,smooth)*self.ponde)

    
    def train(self, model, nb_additional_epochs, batch_size, train_gene, val_gene, learning_rate=None ,patience=1e10):

        if learning_rate is not None:
          tf.set_value(model.optimizer.lr, learning_rate)
     
        # early stopping initialization
        counter = 0

        # training epoch
        try:
            for k in range(nb_additional_epochs):

                self.epoch_count+=1


                if counter >= patience:
                    print("The model is not improving any more. Stopping the training process..")                   
                    break


                starting_time = time.time()

                # data generation
                print("new data generation")
                X_ep, Y_ep = next(train_gene)
                X_val_ep, Y_val_ep = next(val_gene)

                self.history = model.fit(
                    X_ep,
                    Y_ep,
                    batch_size=batch_size,
                    initial_epoch=self.epoch_count,
                    epochs=self.epoch_count+1,    
                    validation_data=(X_val_ep,Y_val_ep)
                )


                # saving training epoch history
                duration=time.time()-starting_time
                self.durations.append(duration)
                self.total_duration+=duration

                self.loss_history.append(self.history.history["loss"][0])
                self.val_loss_history.append(self.history.history["val_loss"][0])
                self.acc_history.append(self.history.history["accuracy"][0])
                self.val_acc_history.append(self.history.history["val_accuracy"][0])

                current_value = self.history.history[self.monitor][0]


                if current_value < self.best_value:
                    print(self.monitor +  " improved from {:.5f} to {:.5f}".format(self.best_value, current_value))
                    self.best_value = current_value
                    self.best_epoch = self.epoch_count
                    self._save_model(model)
                    print("Model is saved on epoch {:d}.".format(self.best_epoch))
                    counter = 0
                    self.best_epochs.append(self.best_epoch)
                    self.best_values.append(self.best_value)

                else:
                    print(self.monitor + " did not improve.")
                    counter += 1


        except KeyboardInterrupt:
            print("\n Interuption volontaire")



    def _save_model(self, model):
      
        # serialize model to JSON
        model_json = model.to_json()
        with open(self.model_name, "w") as json_file:
              json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(self.weights_name)



    def plot_training(self):

        print(self.monitor,":",self.best_epoch,self.best_value)
        print(self.best_epochs,self.best_values)



        fig,ax = plt.subplots(2,1 , figsize=(16, 10), sharex=True)

        # loss
        a=np.arange(len(self.loss_history))
        ax[0].plot(a,self.loss_history,label="loss")
        ax[0].plot(a,self.val_loss_history,label="val_loss")
        ax[0].legend()
        ax[0].set_title("Loss per epoch", color='coral', fontsize=16)

        if self.monitor == "val_loss":
            ax[0].plot(self.best_epochs,self.best_values,'o',label="best_loss")


        # accuracy
        ax[1].plot(a,self.acc_history,label="accuracy")
        ax[1].plot(a,self.val_acc_history,label="val_accuracy")
        ax[1].legend()
        ax[1].set_title("Accuracy per epoch", color='coral', fontsize=16)
        if self.monitor == "val_accuracy":
            ax[1].plot(self.best_epochs,self.best_values,'o',label="best_acc")

        ax[1].set_xlabel("epoch")

        fig.savefig(self.dir_out+"/loss_acc_per_epoch.png")
        
    def load_model(self, folder):

        # load json and create model
        json_file = open(folder+"/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(folder+"/model.h5")
        return loaded_model
        