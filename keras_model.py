import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, Conv2D, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

def sigma_dd(n,f=1575.42*10**6,x=0.05):
    # Estimates carrier measurement noise as x cycles (default 0.05 cycles for double difference)
    std=x*299792458/f
    Q=np.ones((n,n))+np.eye(n)
    return 2*std**2*Q

class noise_layer(tf.keras.layers.Layer):
    def __init__(self,n,input_size,**kwargs):
        super(noise_layer,self).__init__(**kwargs)
        self.n=n
        Q=np.zeros((2*n,2*n))
        Q[n:,n:]=sigma_dd(n)
        Q[:n,:n]=sigma_dd(n)
        self.cov=np.zeros((2*n,2*n))
        self.cov[:n,:n]=sigma_dd(n)
        self.cov[n:2*n,n:2*n]=sigma_dd(n)
        self.input_size=input_size
        #tfd=tfp.distributions
        #self.dist=tfd.MultivariateNormalFullCovariance(covariance_matrix=cov)
        

    def call(self,input,training=None):
        if training:
            #spl=self.dist.sample()
            m,k=input.get_shape().as_list()
            if m is not None:
              add=np.zeros((m,k))
              add[:,:2*self.n]=np.random.multivariate_normal(np.zeros(2*self.n),self.cov,m)
              return tf.identity(input)+tf.stop_gradient(tf.convert_to_tensor(add,dtype=tf.float32))
            else:
              return tf.identity(input)
        else:
            return tf.identity(input)
    def get_config(self):
        config = super(noise_layer, self).get_config()
        config['n']=self.n 
        config['cov']=self.cov
        config['input_size']=self.input_size
        return config


def FCModel_reg(hidden_sizes,non_norm_layer_size,output_size=10,use_BN=False,use_dropout=0.0):
    model = tf.keras.Sequential()
    for hidden_size in hidden_sizes:
        model.add(Dense(hidden_size,activation=mish))#,kernel_regularizer=tf.keras.regularizers.l1(0.0005)))
        if use_BN:
            model.add(BatchNormalization())
        if use_dropout>0:
            model.add(Dropout(use_dropout))
    if non_norm_layer_size>0:
        model.add(Dense(non_norm_layer_size,activation='relu'))
    model.add(Dense(output_size,activation='linear'))
    model.compile(optimizer='adam',loss='mse')
    return model


def init_branch(hidden_sizes,use_BN=False):
    model=tf.keras.Sequential()
    for hidden_size in hidden_sizes:
        model.add(Dense(hidden_size,activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
        if use_BN:
            model.add(BatchNormalization())
    return model

def end_branch(hidden_sizes,output_size,use_BN=False,act='softmax'):
    model=tf.keras.Sequential()
    for hidden_size in hidden_sizes:
        model.add(Dense(hidden_size,activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
        if use_BN:
            model.add(BatchNormalization())
    model.add(Dense(output_size,activation=act))
    return model


def conv_init_branch():
    model=tf.keras.Sequential()   
    model.add(Conv2D(64,(1,1),activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(BatchNormalization())
    model.add(Flatten())
    return model


def conv_reg(hidden_sizes, output_size):
    model=tf.keras.Sequential()
    model.add(Conv2D(32,(1,1),padding='same',activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),padding='same',activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(BatchNormalization())
    model.add(Flatten())
    for hidden_size in hidden_sizes:
        model.add(Dense(hidden_size,activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
        model.add(BatchNormalization())
    model.add(Dense(output_size,activation='linear'))
    model.compile(optimizer='adam',loss='mse')
    return model



def MO_model(init_sizes,end_sizes,output_size,n_branches,input_shape=(80,),use_BN=False,noise=False,noise_n=0,act='sotfmax'):
    inputs=tf.keras.Input(shape=input_shape)
    if noise:
      noisy_inputs=noise_layer(noise_n,input_shape[0])(inputs)
      init=init_branch(init_sizes,use_BN)
      init_out=init(noisy_inputs)
    else:
      init=init_branch(init_sizes,use_BN)
      init_out=init(inputs)
    outputs=[]
    for i in range(n_branches):
        out_branch=end_branch(end_sizes,output_size,use_BN,act)
        outputs.append(out_branch(init_out))
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer='adam',loss=['categorical_crossentropy' for i in range(n_branches)],metrics=['acc'])
    return model

def convMO_model(end_sizes,output_size,n_branches,input_shape=(6,4,2,),use_BN=False):
    inputs=tf.keras.Input(shape=input_shape)
    init=conv_init_branch()
    init_out=init(inputs)
    outputs=[]
    for i in range(n_branches):
        out_branch=end_branch(end_sizes,output_size,use_BN)
        outputs.append(out_branch(init_out))
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer='adam',loss=['categorical_crossentropy' for i in range(n_branches)],metrics=['acc'])    
    return model

