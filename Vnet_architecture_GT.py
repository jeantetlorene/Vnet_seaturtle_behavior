# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:59:16 2023

@author: ljeantet
"""
import keras
import tensorflow as tf
from keras.layers import Conv1D,UpSampling1D,Add,SpatialDropout1D, BatchNormalization,Concatenate
import keras.layers as layers


class Vnet_4levels:
    
    def __init__(self, window_width,nb_inputs,nb_outputs, 
                 model_depth, dropout, kernel_size,activation_head):        
        self.depth=model_depth
        self.dropout_rate=dropout
        self.batch_norm=True
        self.padding="same"
        self.kernel_size=kernel_size
        self.model_head=activation_head
        self.window_width=window_width
        self.nb_inputs=nb_inputs
        self.nb_outputs=nb_outputs

        
    def doubleConv(self, Y, depth,NAME=False):         
        Y = Conv1D(depth, self.kernel_size, activation="relu", padding='same')(Y)       
        if NAME==True:
            Y = Conv1D(depth, self.kernel_size, activation="relu", padding='same',use_bias=False,name='Last_conv')(Y)
        else:
            Y = Conv1D(depth, self.kernel_size, activation="relu", padding='same',use_bias=False)(Y)
        Y = BatchNormalization()(Y)
        Y = SpatialDropout1D(self.dropout_rate)(Y)
        return Y

    def makeUp(self,Y,depth):
        Y = UpSampling1D()(Y)
        return Conv1D(depth,2,activation="relu",padding = "same")(Y)
   
    
            
    def Vnet(self,inputs):        
        #PATCH_DIM
        conv_1 = self.doubleConv(inputs,self.depth)

        #PATCH_DIM/2
        down_1 = Conv1D(self.depth*2,2, strides=2, padding = "same", activation="relu")(conv_1)
        conv_2 = self.doubleConv(down_1,self.depth*2)
        
        #PATCH_DIM/4
        Y=Add()([down_1,conv_2])
        down_2 = Conv1D(self.depth*4, 2, strides=2, padding = "same", activation="relu")(Y)
        conv_3 = self.doubleConv(down_2,self.depth*4)
                 
        #PATCH_DIM/8
        Y=Add()([down_2,conv_3])
        down_3 = Conv1D(self.depth*8, 2, strides=2, padding = "same", activation="relu")(Y)        
        conv_4 = self.doubleConv(down_3,self.depth*8)

        #PATCH_DIM/4
        up_1=self.makeUp(conv_4,self.depth*4)              
        Y = Add()([conv_3, up_1])        
        pre_conv_5= self.doubleConv(Y,self.depth*4)
        conv_5=Add()([pre_conv_5,up_1])        
        
        #PATCH_DIM/2
        up_2=self.makeUp(conv_5,self.depth*2)                      
        Y = Add()([conv_2, up_2])        
        pre_conv_6= self.doubleConv(Y,self.depth*2)
        conv_6=Add()([pre_conv_6,up_2])
        
        #PATCH_DIM
        up_3=self.makeUp(conv_6,self.depth)                      
        Y = Add()([conv_1, up_3])          
        pre_conv_7= self.doubleConv(Y,self.depth,NAME=True)
        conv_7=Add()([pre_conv_7,up_3])  

        return conv_7 
    
    def head_classif_multiclass(self, Y):
        return Conv1D(self.nb_outputs, 1,activation=self.model_head)(Y)
    
    def __call__(self):
    
        inputs= keras.Input(shape=(self.window_width,self.nb_inputs))
        Y=self.Vnet(inputs)
        outputs=self.head_classif_multiclass(Y)
        model = keras.Model(inputs=inputs, outputs=outputs)
    
        return model
    
    
class ContinuousPadding1D(tf.keras.layers.Layer):

    def __init__(self, padding, **kwargs):
        super(ContinuousPadding1D, self).__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        return tf.concat([inputs[:,:1,:]*tf.ones((1, self.padding, 1)),
                          inputs,
                          inputs[:,-1:,:]*tf.ones((1, self.padding, 1))], axis=1)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]+2*self.padding, input_shape[2])

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ContinuousPadding1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class VNet:
    
    def __init__(self, levels, depth, kernel_size, up_kernel_size, activation,
                 dropout, batch_norm, up_constant_init): 
        self.levels = levels
        self.depth = depth
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.activation = activation
        self.dropout_rate = dropout
        self.batch_norm = batch_norm
        self.up_constant_init = up_constant_init
        
    def doubleConv(self, Y, depth):
        Y = ContinuousPadding1D(self.kernel_size//2)(Y)
        Y = tf.keras.layers.Conv1D(depth, self.kernel_size,
                                   activation=self.activation,
                                   padding='valid')(Y)
        Y = ContinuousPadding1D(self.kernel_size//2)(Y) 
        Y = tf.keras.layers.Conv1D(depth, self.kernel_size,
                                   activation=self.activation,
                                   padding='valid',use_bias=False)(Y)
        if self.batch_norm:
            Y = tf.keras.layers.BatchNormalization()(Y)
        Y = tf.keras.layers.SpatialDropout1D(self.dropout_rate)(Y)
        return Y

    def makeDown(self, Y, depth):
        # down = tf.keras.layers.MaxPooling1D()(Y)
        # down = ContinuousPadding1D(1)(down)
        # down = tf.keras.layers.Conv1D(self.depth, 3, activation="softplus")(down)
        down = tf.keras.layers.Conv1D(depth, 2, strides=2,
                                      padding="same", activation=self.activation)(Y)
        return down

    def makeUp(self, Y, depth):
        Y = tf.keras.layers.UpSampling1D()(Y)
        if self.up_constant_init:
            up = tf.keras.layers.Conv1D(depth, self.up_kernel_size,
                                    activation=self.activation, padding="same",
                                    kernel_initializer=tf.keras.initializers.Constant(value=1/self.up_kernel_size/depth))(Y)
        else:
            up = tf.keras.layers.Conv1D(depth, self.up_kernel_size,
                                    activation=self.activation, padding="same")(Y)
        return up
            
    def __call__(self,inputs):
        left = {}
        left[0] = self.doubleConv(inputs,self.depth)
        for i in range(1, self.levels):
            down = self.makeDown(left[i-1], self.depth*2**i)
            conv = self.doubleConv(down, self.depth*2**i)
            left[i] = tf.keras.layers.Add()([down, conv])
        right = left[self.levels-1]
        for i in range(self.levels-2, -1,-1):
            up = self.makeUp(right, self.depth*2**i)
            add = tf.keras.layers.Add()([left[i], up])
            conv = self.doubleConv(add, self.depth*2**i)
            right = tf.keras.layers.Add()([up, conv])
        return right
