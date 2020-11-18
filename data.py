import numpy as np 
import gzip
import pickle
import LAMBDA
import itertools

#base_std=np.array([0.00025,0.00037,0.0005]) #for one second interval, approx s(t)=t*base
base_std=np.array([0.005,0.005,0.001])
f=1575.42*10**6
c=299792458
lda=c/f

def float_sol(y,A,B,Qi):
  if A is not None:
    H=np.hstack([A,B])
  else:
    H=B
  Qhat=np.linalg.inv(np.dot(H.T,np.dot(Qi,H)))
  Qhat=(Qhat+Qhat.T)/2
  xhat=np.dot(Qhat,np.dot(H.T,np.dot(Qi,y)))
  #xhat=np.linalg.solve(np.dot(H.T,np.dot(Qi,H)),np.dot(H.T,np.dot(Qi,y)))
  return xhat,Qhat

def compute_all_floats(n,y,B,Qi,add_zero=False):
  xhats,Qhats=np.array([]),[]
  if add_zero:
    xhat,Qhat=float_sol(y,None,B,Qi)
    xhats=np.concatenate((xhats,xhat))
    Qhats.append(Qhat.copy())
  for i in range(n):
    A=np.zeros((2*n,1))
    A[i,0]=1
    A[i+n,0]=1
    xhat,Qhat=float_sol(y,A,B,Qi)
    xhats=np.concatenate((xhats,xhat))
    Qhats.append(Qhat.copy())

  return xhats,Qhats
  

def std_g(t,base_std=base_std):
    return t*base_std

def gen_dx(n,rmax=1000,zmax=50,sphere=False,fixed_dx=False):
    if not fixed_dx:
      if not sphere:
        r=np.random.uniform(0,rmax,(n,1))
        theta=np.random.uniform(0,2*np.pi,(n,1))
        z=np.random.uniform(-zmax,zmax,(n,1))
        x=r*np.cos(theta)
        y=r*np.sin(theta)
        if n==1:
            return np.array([x,y,z]).reshape(-1,)
        return np.hstack([x,y,z])
      else:
        r=np.random.uniform(0,rmax,(n,1))
        d=np.random.normal(0,1,(n,3))
        return r*d/np.linalg.norm(d,axis=1,keepdims=True)
    else:
      if not sphere:
        r=np.random.uniform(0,rmax,1)
        theta=np.random.uniform(0,2*np.pi,1)
        z=np.random.uniform(-100,100,1)
        x=r*np.cos(theta)
        y=r*np.sin(theta)
        dx=np.array([[x,y,z] for i in range(n)])
        return dx
      else:
        r=np.random.uniform(0,rmax,1)
        d=np.random.normal(0,1,3)
        x=r*d/np.linalg.norm(d)
        return np.array([x for i in range(n)])

def gen_dg(n,n_sat,t=1,base_std=base_std,n_epochs=1):
    std=std_g(t,base_std)
    dg=np.random.multivariate_normal(np.zeros(3),np.diag(std)**2,n*(n_sat-1))
    if n_epochs==1:
      return dg.reshape(n,n_sat-1,3)
    else:
      std=std/10
      dgs=[dg]
      for i in range(n_epochs-1):
        dg_n=dgs[-1].copy()
        dg_n=dg_n+np.random.multivariate_normal(np.zeros(3),np.diag(std)**2,n*(n_sat-1))
        dgs.append(dg_n)
      shapped_output=[]
      for i in range(len(dgs)):
        shapped_output.append(dgs[i].reshape(n,n_sat-1,3))
      return np.array(shapped_output)

def sigma_dd(n,f=1575.42*10**6,x=0.05):
    # Estimates carrier measurement noise as x cycles (default 0.05 cycles for double difference)
    std=x*299792458/f
    Q=np.ones((n,n))+np.eye(n)
    return 2*std**2*Q

def sigma_td(n,f=1575.42*10**6,x=0.05):
    # Estimates carrier measurement noise as x cycles (default 0.05 cycles for double difference)
    std=x*299792458/f
    Q=-np.ones((n,n))+3*np.eye(n)
    return 4*std**2*Q

def gen_G(n_sat,max_angle=75*np.pi/180,lda=lda):
    z=np.random.uniform(np.cos(max_angle),1,n_sat)
    phi=np.random.uniform(0,2*np.pi,n_sat)
    los=np.zeros((n_sat,3))
    los[:,0]=np.sqrt(1-z**2)*np.cos(phi)
    los[:,1]=np.sqrt(1-z**2)*np.sin(phi)
    los[:,2]=z
    return -(los[1:]-los[0])/lda

def gen_multiG(n,n_sat,max_angle=75*np.pi/180,lda=lda):
    z=np.random.uniform(np.cos(max_angle),1,(n,n_sat))
    phi=np.random.uniform(0,2*np.pi,(n,n_sat))
    los=np.zeros((n,n_sat,3))
    los[:,:,0]=np.sqrt(1-z**2)*np.cos(phi)
    los[:,:,1]=np.sqrt(1-z**2)*np.sin(phi)
    los[:,:,2]=z
    los=np.transpose(los,[1,0,2])
    mg=-(los[1:]-los[0])/lda
    mg=np.transpose(mg,[1,0,2])
    return mg

def gen1(n_sat):
    N=np.random.randint(-20,21,n_sat-1)
    dx=gen_dx(1)
    G=gen_G(n_sat)
    dg=gen_dg(1,n_sat)
    y1=np.dot(G,dx)+N 
    y2=y1+np.dot(dg,dx)
    y1=y1+np.random.multivariate_normal(np.zeros(n_sat-1),sigma_dd(n_sat-1),1)
    y2=y2+np.random.multivariate_normal(np.zeros(n_sat-1),sigma_td(n_sat-1),1)
    y=np.concatenate([y1,y2])
    Gf=np.vstack([G,G+dg])
    return N,dx,Gf,y.reshape((-1,))

def gen_N(int_range,number_per_int,n_sat,n=0,random_pos=False,min_N=0):
    if n==-1:
      Nv=np.arange(int_range[0],int_range[1]+1)
      Nv=np.repeat(Nv,number_per_int)
      r=((int_range[1]-int_range[0]+1)*number_per_int)%(n_sat-1)
      if r > 0:
          Nv=np.concatenate((Nv,np.random.randint(int_range[0],int_range[1]+1,n_sat-r-1)))
      np.random.shuffle(Nv)
      return Nv.reshape(-1,n_sat-1)
    elif n==0:
      return np.zeros((number_per_int,n_sat-1)).astype('int')
    else:
      Nv=np.array([i for i in range(int_range[0],int_range[1]+1) if abs(i)>=min_N])
      Nv=np.repeat(Nv,number_per_int)
      r=((int_range[1]-int_range[0]+1)*number_per_int)%n
      if r > 0:
          Nv=np.concatenate((Nv,np.random.randint(int_range[0],int_range[1]+1,n-r)))
      np.random.shuffle(Nv)
      Nv=Nv.reshape(-1,n)
      out=np.zeros((Nv.shape[0],n_sat-1),dtype='int32')
      if random_pos:
        pos=np.arange(n_sat-1)
        for i in range(len(out)):
          np.random.shuffle(pos)
          out[i,pos[:n]]=Nv[i]
      else:
        out[:,:n]=Nv

      return out

def compute_A(n,n_epochs,n_sat,single_design=False,N=None):
    if not single_design:
      Ai=np.zeros(((n_epochs+1)*(n_sat-1),n_sat-1))
      for i in range(n_epochs+1):
        Ai[i*(n_sat-1):(i+1)*(n_sat-1)]=np.eye(n_sat-1)
      A=np.zeros((n,Ai.shape[0],Ai.shape[1]))
      for i in range(n):
        A[i]=Ai
      return A
    else:
      A=np.zeros((n,(n_epochs+1)*(n_sat-1),1))
      for i in range(n):
        A[i,np.argmax(N[i])::(n_sat-1)]=np.ones((n_epochs+1,1))
      return A

def stack_G(G,dgs,n_epochs):
    if n_epochs==1:
      return np.concatenate((G,G+dgs),axis=1)
    else:
      return np.concatenate([G]+[G+dgs[i] for i in range(n_epochs)],axis=1)

      

def compute_y(n,n_sat,N,dx,G,dg,noisy=True,n_epochs=1,p_cycle=0,p_multipath=0,all_multipath=False):
    if n_epochs==1:
      y=np.zeros((n,2*(n_sat-1)))
      cycles=np.random.rand(n)<p_cycle
      nc=np.random.randint(-20,20,n)
      ni=np.random.randint(0,n_sat-2,n)
      imp=np.random.randint(0,n_sat-2,n)
      mp=np.random.rand(n)<p_multipath

      for i in range(n):
          y1=np.dot(G[i],dx[i])+N[i]
          y2=y1+np.dot(dg[i],dx[i])
          y2[ni[i]]+=cycles[i]*nc[i]
          if not all_multipath:
            mp_value=mp[i]*np.random.uniform(-0.5,0.5,1)
            y1[imp[i]]+=mp_value
            y2[imp[i]]+=mp_value
          if noisy:
            y1=y1+np.random.multivariate_normal(np.zeros(n_sat-1),sigma_dd(n_sat-1),1)
            y2=y2+np.random.multivariate_normal(np.zeros(n_sat-1),sigma_dd(n_sat-1),1)
          if all_multipath:
            multipath=np.random.uniform(-0.05,0.05,n_sat-1)/lda
            y1+=multipath
            y2+=multipath
          y[i]=np.concatenate([y1,y2]).reshape(-1,)
      return y,ni,nc,cycles.astype('float'),mp,imp
    else:
      y=np.zeros((n,(1+n_epochs)*(n_sat-1)))
      for i in range(n):
        y1=np.dot(G[i],dx[i])+N[i]
        yl=[y1]
        for j in range(n_epochs):
          yj=y1.copy()+np.dot(dg[j,i],dx[i])
          yl.append(yj)
        if noisy:
          for j in range(n_epochs+1):
            yl[j]=yl[j]+np.random.multivariate_normal(np.zeros(n_sat-1),sigma_dd(n_sat-1),1)
        y[i]=np.concatenate(yl).reshape(-1,)
      return y,None,None,None,None,None
    




class Dataset():
    def __init__(self,int_range,number_per_int,n_sat,noisy=True,to_fix=-1,n_epochs=1,random_pos=False,p_cycle=0,single_design=False,p_multipath=0,all_multipath=False,min_N=0):
        
        self.int_range=int_range
        self.n_sat=n_sat
        self.number_per_int=number_per_int
        self.N=gen_N(int_range,number_per_int,n_sat,to_fix,random_pos,min_N)
        self.n=self.N.shape[0]
        self.G=gen_multiG(self.n,n_sat)
        self.dx=gen_dx(self.n,300,300,True,False)
        self.dg=gen_dg(self.n,n_sat,n_epochs=n_epochs)
        self.stacked_G=stack_G(self.G,self.dg,n_epochs)
        self.y,self.ni,self.nc,self.cycles,self.mp,self.imp=compute_y(self.n,n_sat,self.N,self.dx,self.G,self.dg,noisy,n_epochs,p_cycle,p_multipath,all_multipath)
        self.A=compute_A(self.n,n_epochs,n_sat,single_design,self.N)
        self.H=None
        self.Q=None
        self.Qi=None
        self.Qh=None
        self.x_float=None
        self.lda=None
        self.n_epochs=n_epochs
        self.xhats=None
        self.Qhats=None
        self.single_design=single_design
    
    @staticmethod
    def isGZIP(filename):
        if filename.split('.')[-1] == 'gz':
            return True
        return False

    # Using HIGHEST_PROTOCOL is almost 2X faster and creates a file that
    # is ~10% smaller.  Load times go down by a factor of about 3X.
    def save(self, filename='Dataset.pkl'):
        if self.isGZIP(filename):
            f = gzip.open(filename, 'wb')
        else:
            f = open(filename, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    # Note that loading to a string with pickle.loads is about 10% faster
    # but probaly comsumes a lot more memory so we'll skip that for now.
    @classmethod
    def load(cls, filename='Dataset.pkl'):
        if cls.isGZIP(filename):
            f = gzip.open(filename, 'rb')
        else:
            f = open(filename, 'rb')
        n = pickle.load(f)
        f.close()
        return n

    def prepare_data(self,normalize=True,classification=True,normalize_output_reg=False,add_Q=False,add_design=False,float_solution=False,relaxed=False,cheat=False):
        if add_Q:
          x_train=np.zeros((self.n,8*(self.n_sat-1)+4*(self.n_sat-1)**2))
          x_train[:,:2*(self.n_sat-1)]=self.y
          if self.Qi is None:
            self.compute_Q()
          for i in range(self.n):
            x_train[i,-4*(self.n_sat-1)**2:]=self.Qi.reshape(-1,)
        elif add_design:
          x_train=np.zeros((self.n,2*(self.n_sat-1)+(self.n_sat-1)*self.n_sat//2))
          x_train[:,:2*(self.n_sat-1)]=self.y
          if self.Qh is None:
            self.compute_design()
          for i in range(self.n):
            Q=self.Qh[i,:-3,:-3]
            x_train[i,-(self.n_sat-1)*self.n_sat//2:]=Q[np.triu_indices(self.n_sat-1)]
        elif cheat:
          x_train=np.zeros((self.n,8*(self.n_sat-1)+3))
          x_train[:,:2*(self.n_sat-1)]=self.y
          x_train[:,2*(self.n_sat-1):-3]=self.stacked_G.reshape((self.n,-1))
          x_train[:,-3:]=self.dx
        else:
          x_train=np.zeros((self.n,8*(self.n_sat-1)))
          x_train[:,:2*(self.n_sat-1)]=self.y
          x_train[:,2*(self.n_sat-1):]=self.stacked_G.reshape((self.n,-1))
        
        if float_solution:
          if self.x_float is None:
            self.compute_float()
          x_train=np.zeros((self.n,self.n_sat-1+(self.n_sat-1)*self.n_sat//2))
          x_train[:,:-(self.n_sat-1)*self.n_sat//2]=self.x_float[:,:-3]
          for i in range(self.n):
            Q=self.Qh[i,:-3,:-3]
            x_train[i,-(self.n_sat-1)*self.n_sat//2:]=Q[np.triu_indices(self.n_sat-1)]
        if normalize:
            if not float_solution:
              x_train[:,:2*(self.n_sat-1)]/=100000
              x_train[:,2*(self.n_sat-1):]/=10
            else:
              x_train[:,:-(self.n_sat+2)**2]/=20
              x_train[:,-(self.n_sat+2)**2:]/=500
        if classification:
            if relaxed:
              y_train=self.relaxed_y()
              return x_train,y_train
            y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
            for i in range(self.n):
                y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
        else:
            y_train=self.N
            if normalize_output_reg:
                y_train=y_train/self.int_range[1]
        return x_train,y_train
    
    def compute_Q(self):
        self.Q=np.zeros(((self.n_epochs+1)*(self.n_sat-1),(self.n_epochs+1)*(self.n_sat-1)))
        for i in range(self.n_epochs+1):
          self.Q[i*(self.n_sat-1):(i+1)*(self.n_sat-1),i*(self.n_sat-1):(i+1)*(self.n_sat-1)]=sigma_dd(self.n_sat-1)
        self.Qi=np.linalg.inv(self.Q)

    def compute_design(self):
        if self.H is None:
          m,p=self.A[0].shape
          self.H=np.zeros((self.n,(1+self.n_epochs)*(self.n_sat-1),3+p))
          self.Qh=np.zeros((self.n,3+p,3+p))
          if self.Q is None:
              self.compute_Q()
          for i in range(self.n):
              self.H[i]=np.hstack((self.A[i],self.stacked_G[i]))
              self.Qh[i]=np.linalg.inv(np.dot(self.H[i].T,np.dot(self.Qi,self.H[i])))
              self.Qh[i]=(self.Qh[i]+self.Qh[i].T)/2

    def compute_float(self):
        if self.x_float is None:
          m,p=self.A[0].shape
          self.x_float=np.zeros((self.n,p+3))
          if self.Qh is None:
            self.compute_design()
          for i in range(self.n):
            self.x_float[i]=np.dot(self.Qh[i],np.dot(self.H[i].T,np.dot(self.Qi,self.y[i])))
    def compute_lambda(self):
      if not self.single_design:
        if self.lda is None:
          if self.x_float is None:
            self.compute_float()
          if self.Qh is None:
            self.compute_design()
          self.lda=np.zeros_like(self.N)
          for i in range(self.n):
            afixed,sqnorm,Ps,Qzhat,Z,nfixed,mu = LAMBDA.main(self.x_float[i,:-3],self.Qh[i,:-3,:-3],1)
            self.lda[i]=afixed[:,0]
      else:
        if self.lda is None:
          self.lda=np.zeros_like(self.N)
          if self.Qi is None:
            self.compute_Q()
          x_float
          A=compute_A(self.n,self.n_epochs,self.n_sat,single_design=False,N=None)
          for i in range(self.n):
            x,Q=float_sol(self.y[i],A[i],self.stacked_G[i],self.Qi)
            afixed,sqnorm,Ps,Qzhat,Z,nfixed,mu = LAMBDA.main(x[:-3],Q[:-3,:-3],1)
            self.lda[i]=afixed[:,0]

    def compute_lambda_accuracy(self):
      if self.lda is None:
        self.compute_lambda()
      return 1-np.count_nonzero(self.N-self.lda)/(self.n_sat-1)/self.n

    def compute_lambda_mse(self):
      if self.lda is None:
        self.compute_lambda()
      return np.sum((self.lda-self.N)**2)/self.n/(self.n_sat-1)

    def prepare_lda_val(self):
      if self.lda is None:
        self.compute_lambda()
      x_train=np.zeros((self.n,9*(self.n_sat-1)))
      x_train[:,3*(self.n_sat-1):]=self.stacked_G.reshape((self.n,-1))
      x_train[:,:2*(self.n_sat-1)]=self.y
      x_train[:,2*(self.n_sat-1):3*(self.n_sat-1)]=self.lda
      y_train=(np.sum(self.N!=self.lda,axis=1)>0).astype('float32')
      return x_train,y_train

    def prepare_lda_val2(self):
      if self.lda is None:
        self.compute_lambda()
      x_train=np.zeros((self.n,self.n_sat+2))
      x_train[:,:self.n_sat-1]=self.y[:,:self.n_sat-1]
      y_c=self.y[:,:self.n_sat-1]-self.lda
      for i in range(self.n):
        x_train[i,-3:]=np.dot(np.linalg.pinv(self.G[i]),y_c[i])
      y_train=(np.sum(self.N!=self.lda,axis=1)>0).astype('float32')
      return x_train,y_train

    def prepare_lda_val3(self):
      if self.lda is None:
        self.compute_lambda()
      x_train=np.zeros((self.n,4*(self.n_sat-1)))
      y_c=self.y[:,:self.n_sat-1]-self.lda
      x_train[:,:self.n_sat-1]=y_c
      x_train[:,self.n_sat-1:]=self.G.reshape((self.n,-1))
      y_train=(np.sum(self.N!=self.lda,axis=1)>0).astype('float32')
      return x_train,y_train

    def prepare_conv(self,reg=False,relaxed=False):
      x_train=np.zeros(((self.n,self.n_sat-1,4,2)))
      x_train[:,:,0,:]=self.y.reshape(self.n,-1,2)/10000
      x_train[:,:,1:,0]=self.G
      x_train[:,:,1:,1]=self.G+self.dg
      if reg:
        y_train=self.N
        return x_train,y_train
      if relaxed:
        y_train=self.relaxed_y()
        return x_train,y_train
      y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
      for i in range(self.n):
        y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      return x_train,y_train

    def relaxed_y(self,q=5):
      if (self.int_range[1]-self.int_range[0]+1)%q!=0:
        raise ValueError('Wrong size for relaxation')
      
      N_q=(self.N-self.int_range[0])//q
      n_class=(self.int_range[1]-self.int_range[0]+1)//q
      y_train=np.zeros((self.n,self.n_sat-1,n_class))
      for i in range(self.n):
        y_train[i,np.arange(self.n_sat-1),N_q[i]]=1
      return y_train

    def great_cheat(self,classification=True,relaxed=False):
      x_train=np.zeros((self.n,2*(self.n_sat-1)))
      for i in range(self.n):
        x_train[i]=self.y[i]-np.dot(self.stacked_G[i],self.dx[i])
      if classification:
            if relaxed:
              y_train=self.relaxed_y()
              return x_train,y_train
            y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
            for i in range(self.n):
                y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      else:
          y_train=self.N
      return x_train,y_train
    def semi_cheat(self,classification=True,relaxed=False, with_G=False):
      if not with_G:
        x_train=np.zeros((self.n,2*(self.n_sat-1)))
        noisy_dx=self.dx + np.random.normal(0,10,self.dx.shape)

        for i in range(self.n):
          x_train[i]=self.y[i]-np.dot(self.stacked_G[i],noisy_dx[i])
      else:
        x_train=np.zeros((self.n,8*(self.n_sat-1)))
        noisy_dx=self.dx + np.random.normal(0,10,self.dx.shape)

        for i in range(self.n):
          x_train[i,:2*self.n_sat-2]=self.y[i]-np.dot(self.stacked_G[i],noisy_dx[i])
        x_train[:,2*(self.n_sat-1):]=self.stacked_G.reshape((self.n,-1))  
      if classification:
            if relaxed:
              y_train=self.relaxed_y()
              return x_train,y_train
            y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
            for i in range(self.n):
                y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      else:
          y_train=self.N
      return x_train,y_train

    def floatx(self,classification=True,relaxed=False):
      x_train=np.zeros((self.n,2*self.n_sat-2))
      if self.x_float is None:
        self.compute_float()
      for i in range(self.n):
        x_train[i]=self.y[i]-np.dot(self.stacked_G[i],self.x_float[i,-3:])
      if classification:
              if relaxed:
                y_train=self.relaxed_y()
                return x_train,y_train
              y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
              for i in range(self.n):
                  y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      else:
          y_train=self.N
      return x_train,y_train


    def floatx_withy(self,classification=True,relaxed=False): 
      x_train=np.zeros((self.n,3*self.n_sat+(self.n_sat+2)*(self.n_sat+3)//2))
      if self.x_float is None:
        self.compute_float()
      if self.Qh is None:
        self.compute_design()
      x_train[:,:2*(self.n_sat-1)]=self.y
      x_train[:,2*(self.n_sat-1):-(self.n_sat+2)*(self.n_sat+3)//2]=self.x_float
      for i in range(self.n):
        Q=self.Qh[i]
        x_train[i,-(self.n_sat+2)*(self.n_sat+3)//2:]=Q[np.triu_indices(self.n_sat+2)]

      if classification:
              if relaxed:
                y_train=self.relaxed_y()
                return x_train,y_train
              y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
              for i in range(self.n):
                  y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      else:
          y_train=self.N
      return x_train,y_train

    def with_invg(self,classification=True,relaxed=False):
      x_train=np.zeros((self.n,14*self.n_sat-14))
      x_train[:,:2*self.n_sat-2]=self.y
      x_train[:,2*self.n_sat-2:8*self.n_sat-8]=self.stacked_G.reshape((self.n,-1))
      for i in range(self.n):
        x_train[i,8*self.n_sat-8:11*self.n_sat-11]=np.linalg.pinv(self.G[i]).reshape((-1,))
        x_train[i,11*self.n_sat-11:14*self.n_sat-14]=np.linalg.pinv(self.G[i]+self.dg[i]).reshape((-1,))
      if classification:
              if relaxed:
                y_train=self.relaxed_y()
                return x_train,y_train
              y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
              for i in range(self.n):
                  y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      else:
          y_train=self.N
      return x_train,y_train

    def new_reg(self):
      x_train=np.zeros((self.n,8*(self.n_sat-1)))
      x_train[:,:2*self.n_sat-2]=self.y
      for i in range(self.n):
        x_train[i,2*self.n_sat-2:8*self.n_sat-8]=np.linalg.pinv(self.stacked_G[i]).reshape((-1,))
      return x_train,self.dx

    def allinone(self,classification=True,relaxed=False):
      if self.x_float is None:
        self.compute_float()
      if self.Qh is None:
        self.compute_design()
      x_train=np.zeros((self.n,15*self.n_sat-12+(self.n_sat+2)*(self.n_sat+3)//2))
      x_train[:,:2*self.n_sat-2]=self.y
      x_train[:,2*self.n_sat-2:8*self.n_sat-8]=self.stacked_G.reshape((self.n,-1))
      for i in range(self.n):
        x_train[i,8*self.n_sat-8:11*self.n_sat-11]=np.linalg.pinv(self.G[i]).reshape((-1,))
        x_train[i,11*self.n_sat-11:14*self.n_sat-14]=np.linalg.pinv(self.G[i]+self.dg[i]).reshape((-1,))
      x_train[:,14*self.n_sat-14:15*self.n_sat-12]=self.x_float
      for i in range(self.n):
        Q=self.Qh[i]
        x_train[i,-(self.n_sat+2)*(self.n_sat+3)//2:]=Q[np.triu_indices(self.n_sat+2)]
      if classification:
              if relaxed:
                y_train=self.relaxed_y()
                return x_train,y_train
              y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
              for i in range(self.n):
                  y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      else:
          y_train=self.N
      return x_train,y_train

    def new_float(self,classification=True,relaxed=False):
      if self.x_float is None:
        self.compute_float()
      if self.Qh is None:
        self.compute_design()
      m=self.x_float.shape[1]
      k=self.Qh.shape[1]
      k2=k*(k+1)//2
      l=self.n_sat-1
      x_train=np.zeros((self.n,8*l+m+k2))
      x_train[:,:2*l]=self.y
      x_train[:,2*l:8*l]=self.stacked_G.reshape((self.n,-1))
      x_train[:,8*l:8*l+m]=self.x_float
      for i in range(self.n):
        Q=self.Qh[i]
        x_train[i,-k2:]=Q[np.triu_indices(k)]
      if classification:
              if relaxed:
                y_train=self.relaxed_y()
                return x_train,y_train
              y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
              for i in range(self.n):
                  y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      else:
          y_train=self.N
      return x_train,y_train
      




    def reg_test(self):
        x_train=np.zeros((self.n,8*(self.n_sat-1)))
        x_train[:,:2*(self.n_sat-1)]=self.y
        x_train[:,2*(self.n_sat-1):]=self.stacked_G.reshape((self.n,-1))
        y_train=np.zeros((self.n,self.n_sat+2))
        y_train[:,:self.n_sat-1]=self.N
        y_train[:,-3:]=self.dx
        return x_train,y_train
    
    def from_x(self,x):
      x_train=np.zeros((self.n,2*self.n_sat-2))
      for i in range(self.n):
        x_train[i]=self.y[i]-np.dot(self.stacked_G[i],x[i])
      return x_train

    def conv_multiepoch(self,classification=True,relaxed=False):
      x_train=np.zeros(((self.n,self.n_sat-1,6+self.n_sat,1+self.n_epochs)))
      x_train[:,:,0,:]=self.y.reshape(self.n,-1,1+self.n_epochs)/200
      x_train[:,:,1:4,0]=self.G
      if self.Qh is None:
        self.compute_design()
      for i in range(self.n_epochs):
        x_train[:,:,1:4,1+i]=self.G+self.dg[i]
      for i in range(self.n):
        for j in range(self.n_epochs+1):
          x_train[i,:,4:7,j]=np.linalg.pinv(x_train[i,:,1:4,j]).reshape((self.n_sat-1,3))
      for i in range(self.n):
        for j in range(self.n_epochs+1):
          x_train[i,:,7:,j]=self.Qh[i,:-3,:-3]
      if relaxed:
        y_train=self.relaxed_y()
        return x_train,y_train
      y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
      for i in range(self.n):
        y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      return x_train,y_train

    def detect_failure(self):
      return (self.N!=0).astype('float') 
    
    def all_float(self,add_zero=False,classification=True,relaxed=False):
      x_train=[]
      self.compute_Q()
      k=self.n_sat-1
      for i in range(self.n):
        xhats,Qhats=compute_all_floats(k,self.y[i],self.stacked_G[i],self.Qi,add_zero)
        Qhats2=np.array([])
        for j in range(len(Qhats)):
          Qhats2=np.concatenate((Qhats2,Qhats[j][np.triu_indices(len(Qhats[j]))].reshape(-1,)))
        #Qhats2=np.array(Qhats2)
        #print(Qhats2)
        x_train_i=np.concatenate((xhats,Qhats2))
        #x_train_i=xhats.reshape(-1,)
        x_train.append(x_train_i)
      x_train=np.array(x_train)
      if relaxed:
        y_train=self.relaxed_y()
        return x_train,y_train
      y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
      for i in range(self.n):
        y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      return x_train,y_train

    def raim(self,classification=True,relaxed=False):
      x_train=[]
      Q=np.zeros(((self.n_epochs+1)*(self.n_sat-2),(self.n_epochs+1)*(self.n_sat-2)))
      for i in range(self.n_epochs+1):
        Q[i*(self.n_sat-2):(i+1)*(self.n_sat-2),i*(self.n_sat-2):(i+1)*(self.n_sat-2)]=sigma_dd(self.n_sat-2)
      Qi=np.linalg.inv(Q)
      for i in range(self.n):
        x_train_i=[]
        for j in range(self.n_sat-1):
          y=self.y[i].copy()
          y=np.delete(y,[j,j+self.n_sat-1])
          G=self.stacked_G[i].copy()
          #print(G.shape)
          G=np.delete(G,[j,j+self.n_sat-1],axis=0)
          #print(G.shape)
          #x_train_i.append(np.dot(np.linalg.pinv(G),y))
          x,Q=float_sol(y,None,G,Qi)
          x_train_i.append(np.concatenate([x,Q.reshape(-1,)]))
        x_train_i=np.array(x_train_i)
        x_train.append(x_train_i.reshape(-1,))
      x_train=np.array(x_train)
      if relaxed:
        y_train=self.relaxed_y()
        return x_train,y_train
      y_train=np.zeros((self.n,self.n_sat-1,self.int_range[1]-self.int_range[0]+1))
      for i in range(self.n):
        y_train[i,np.arange(self.n_sat-1),self.N[i]-self.int_range[0]]=1
      return x_train,y_train
      

        



def normalize(x_train,group=None):
  if group is None:
    mean=np.mean(x_train,axis=0)
    std=np.std(x_train,axis=0,keepdims=True)
    x_trainn=(x_train-mean)/std
    return x_trainn,mean,std
  else:
    means,stds=[],[]
    x_trainn=np.zeros_like(x_train)
    for i in range(len(group)-1):
      mean=np.mean(x_train[:,group[i]:group[i+1]])
      std=np.std(x_train[:,group[i]:group[i+1]])
      x_trainn[:,group[i]:group[i+1]]=(x_train[:,group[i]:group[i+1]]-mean)/std
      means.append(mean)
      stds.append(std)
    return x_trainn,means,stds

def add_mult(x_train):
  n,m=x_train.shape
  x_trainm=np.zeros((n,m+m*m))
  x_trainm[:,:m]=x_train
  for i in range(n):
    x_trainm[i,m:]=np.outer(x_train[i],x_train[i]).reshape((-1,))
  return x_trainm

comb_table={}
combs=list(itertools.combinations(np.arange(7),2))#+list(itertools.combinations(np.arange(9),1))
for i in range(len(combs)):
  comb_table[combs[i]]=i

def comb_y(N):
  y_train=np.zeros((len(N),len(combs)))
  for i in range(len(N)):
    idx=np.reshape(np.where(N[i]!=0),(-1,))
    #print(idx)
    y_train[i,comb_table[tuple(idx)]]=1
  return y_train



    
    




