import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, Conv2D, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
#import tensorflow_probability
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
#from AdaBelief_tf import AdaBeliefOptimizer as adab
#from tensorflow_addons.activations import mish,lisht
import tensorflow_addons as tfa
import AdaBelief_tf

def log_act(x):
  return tf.math.log(1+(x*tf.dtypes.cast(x>=0,tf.float32)))-tf.math.log(1-(x*tf.dtypes.cast(x<0,tf.float32)))

def exp_act(x):
  return (tf.math.exp(x)-1)*tf.dtypes.cast(x>=0,tf.float32)-(tf.math.exp(-x)-1)*tf.dtypes.cast(x<0,tf.float32)

def lisht(x):
  return x*tf.math.tanh(x)

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


def get_centralized_gradients(optimizer, loss, params):
    """Compute a list of centralized gradients.
    
    Modified version of tf.keras.optimizers.Optimizer.get_gradients:
    https://github.com/keras-team/keras/blob/1931e2186843ad3ca2507a3b16cb09a7a3db5285/keras/optimizers.py#L88-L101
    Reference:
        https://arxiv.org/pdf/2004.01461.pdf
    """
    grads = []
    for grad in K.gradients(loss, params):
        rank = len(grad.shape)
        if rank > 1:
            grad -= tf.reduce_mean(grad, axis=list(range(rank-1)), keep_dims=True)
        grads.append(grad)
    if None in grads:
        raise ValueError('An operation has `None` for gradient. '
                         'Please make sure that all of your ops have a '
                         'gradient defined (i.e. are differentiable). '
                         'Common ops without gradient: '
                         'K.argmax, K.round, K.eval.')
    if hasattr(optimizer, 'clipnorm') and optimizer.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [tf.keras.optimizers.clip_norm(g, optimizer.clipnorm, norm) for g in grads]
    if hasattr(optimizer, 'clipvalue') and optimizer.clipvalue > 0:
        grads = [K.clip(g, -optimizer.clipvalue, optimizer.clipvalue) for g in grads]
    return grads

def get_centralized_gradients_function(optimizer):
    """Produce a get_centralized_gradients function for a particular optimizer instance."""

    def get_centralized_gradients_for_instance(loss, params):
        return get_centralized_gradients(optimizer, loss, params)

    return get_centralized_gradients_for_instance



def reg_acc(y_true,y_pred):
    rounded=tf.math.round(y_pred)
    return tf.math.count_nonzero(tf.abs(rounded-y_true)<0.1,dtype=tf.dtypes.float32)/y_true.shape[0]/y_true.shape[1]

def reg_pseudo_acc(y_true,y_pred):
    rounded=tf.math.round(y_pred)
    return tf.math.count_nonzero(tf.abs(rounded-y_true)<3.5,dtype=tf.dtypes.float32)/y_true.shape[0]/y_true.shape[1]

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
    radam = RectifiedAdam()
    ranger = Lookahead(radam, sync_period=6, slow_step_size=0.5)
    ranger.get_gradients=get_centralized_gradients_function(ranger)

    model.compile(optimizer=ranger,loss='mae')#,metrics=[reg_acc,reg_pseudo_acc])
    return model

def custom_loss(inputs):
  def loss(y_true,y_pred):
    n,m=y_pred.get_shape().as_list()
    if n is None:
      y1=inputs[0,:m]-y_pred
      y2=inputs[0,m:2*m]-y_pred
      y=tf.stack([y1,y2])
      y=tf.reshape(y,(-1,))
      Gi=tf.reshape(inputs[2*m:8*m],(3,-1))
      x=tf.linalg.matvec(Gi,y)
      return K.mean(K.sum(K.square(x-y_true),axis=-1))
    x=tf.zeros_like((n,3))
    for i in range(n):
      raise ValueError('tada')
      y=tf.zeros(2*m)
      y[:m]=inputs[i,:m]-y_pred[i]
      y[m:]=inputs[i,m:2*m]-y_pred[i]
      Gi=tf.reshape(inputs[i,2*m:8*m],(3,-1))
      x[i]=tf.linalg.matvec(Gi,y)
    return K.mean(K.sum(K.square(x-y_true),axis=1))
  return loss

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed
    
    

def FCModel_reg2(hidden_sizes,input_shape,output_size,use_BN=False):
    inputs=tf.keras.Input(shape=input_shape)
    xi=Lambda(lambda x:x)(inputs)
    x=Dense(hidden_sizes[0],activation='relu')(xi)
    for i in range(1,len(hidden_sizes)):
        x=Dense(hidden_sizes[i],activation='relu')(x)
        if use_BN:
            x=BatchNormalization()(x)
    output=Dense(output_size,activation='linear')(x)

    radam = RectifiedAdam()
    ranger = Lookahead(radam, sync_period=6, slow_step_size=0.5)
    ranger.get_gradients=get_centralized_gradients_function(ranger)
    model=tf.keras.Model(inputs,output)
    model.compile(optimizer=ranger,loss=custom_loss(xi))#,metrics=[reg_acc,reg_pseudo_acc])
    return model


def FCModel_lda(hidden_sizes,use_BN=False):
    model = tf.keras.Sequential()
    for hidden_size in hidden_sizes:
        model.add(Dense(hidden_size,activation=lisht))#,kernel_regularizer=tf.keras.regularizers.l1_l2(0.01)))
        if use_BN:
            model.add(BatchNormalization())
    model.add(Dense(1,activation='sigmoid'))
    radam = RectifiedAdam()
    ranger = Lookahead(radam, sync_period=6, slow_step_size=0.5)
    ranger.get_gradients=get_centralized_gradients_function(ranger)

    model.compile(optimizer=ranger,loss='binary_crossentropy',metrics=['acc'])
    return model



def init_branch(hidden_sizes,use_BN=False):
    model=tf.keras.Sequential()
    #model.add(Dense(300,activation=log_act))
    #model.add(BatchNormalization())
    #model.add(Dense(300,activation=exp_act))
    #model.add(BatchNormalization())
    for hidden_size in hidden_sizes:
        model.add(Dense(hidden_size,activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
        #model.add(tf.keras.layers.PReLU())
        if use_BN:
            model.add(BatchNormalization())
    return model

def end_branch(hidden_sizes,output_size,use_BN=False,act='softmax'):
    model=tf.keras.Sequential()
    for hidden_size in hidden_sizes:
        model.add(Dense(hidden_size,activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
        #model.add(tf.keras.layers.PReLU())
        if use_BN:
            model.add(BatchNormalization())

    if output_size==1:
      model.add(Dense(output_size,activation='sigmoid'))#,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    else:
      model.add(Dense(output_size,activation=act))
    return model


def conv_init_branch():
    model=tf.keras.Sequential()
    
    model.add(Conv2D(64,(1,1),activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(BatchNormalization())
    #model.add(Conv2D(128,(3,1),padding='same',activation='relu'))
    #model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(BatchNormalization())
    #model.add(tf.keras.layers.MaxPool2D)
    #model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    #model.add(BatchNormalization())
    model.add(Flatten())
    #model.add(Dropout(0.3))
    return model
def conv_reg(hidden_sizes, output_size):
    model=tf.keras.Sequential()
    model.add(Conv2D(32,(1,1),padding='same',activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(BatchNormalization())
    #model.add(Conv2D(128,(3,1),padding='same',activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Conv2D(128,(1,3),padding='same',activation='relu'))
    #model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),padding='same',activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(BatchNormalization())
    #model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    #model.add(BatchNormalization())
    #model.add(tf.keras.layers.MaxPool2D())
    model.add(Flatten())
    #model.add(Dropout(0.3))
    for hidden_size in hidden_sizes:
        model.add(Dense(hidden_size,activation=mish,kernel_regularizer=tf.keras.regularizers.l1(0.001)))#,kernel_regularizer=tf.keras.regularizers.l1_l2(0.01)))
        model.add(BatchNormalization())
    model.add(Dense(output_size,activation='linear'))
    radam = RectifiedAdam()
    ranger = Lookahead(radam, sync_period=6, slow_step_size=0.5)
    ranger.get_gradients=get_centralized_gradients_function(ranger)

    model.compile(optimizer='adam',loss='mse')#,metrics=[reg_acc,reg_pseudo_acc])
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
    radam = RectifiedAdam()
    ranger = Lookahead(radam, sync_period=6, slow_step_size=0.5)
    ranger.get_gradients=get_centralized_gradients_function(ranger)
    opt = AdaBelief_tf.AdaBeliefOptimizer(
        lr=1e-3,
        total_steps=10000,
        warmup_proportion=0.1,
        min_lr=1e-5,
        epsilon=1e-10
    )
    opt=AdaBelief_tf.RadaBelief()
    sgd_opt=tf.keras.optimizers.SGD(1e-3,momentum=0.9,nesterov=True)
    model.compile(optimizer='adam',loss=['categorical_crossentropy' for i in range(n_branches)],metrics=['acc'])
    l=focal_loss()
    #model.compile(optimizer=opt,loss=l,metrics=['acc'])
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
    radam = RectifiedAdam()
    ranger = Lookahead(radam, sync_period=6, slow_step_size=0.5)
    ranger.get_gradients=get_centralized_gradients_function(ranger)
    sgd_opt=tf.keras.optimizers.SGD(1e-3,momentum=0.9,nesterov=True)
    opt=AdaBelief_tf.RadaBelief()
    model.compile(optimizer=opt,loss=['categorical_crossentropy' for i in range(n_branches)],metrics=['acc'])
    
    return model

