import georinex as gr 
import xarray as xr
import numpy as np
import georinex as gr 
import xarray as xr
import numpy as np
import constants
from lda_print import print_lambda
import coordinates
import cvxpy as cp
import scipy
import data

x0=np.array([ -742080.4125, -5462031.7412,  3198339.6909])
ref_lla=np.array([30.290839133333336, -97.73692683055556, 168.157])
ref_ecef=np.array([ -742080.47182436, -5462030.93596142,  3198338.99257102])
rov_lla=np.array([30.286660500236,-97.727418434941,180.943436])
rov_ecef=np.array([ -741206.92230444, -5462396.47140271,  3197945.43212226])
rov_lla2=np.array([30.286660500236,-97.727418434941,158.40663 ])
rov_ecef2=np.array([ -741204.30559422, -5462377.18730234,  3197934.06621161])

f=1575.42*10**6
c=299792458
lda=c/f
#sigma=0.025*lda
we=7.2921159*1e-5

obs_ref=xr.load_dataset('obs_ref1.nc')
obs_rov=xr.load_dataset('obs_rov1.nc')
eph_ref=xr.load_dataset('eph_ref.nc')
eph_rov=xr.load_dataset('eph_rov.nc')


def compute_sd(obs_ref,obs_rov,t_deb=0,t_fin=None):
    times=obs_rov.time.values
    if t_fin is None:
        t_fin = len(times)
    times=times[t_deb:t_fin]
    obs_rov=obs_rov.sel(time=times)
    obs_ref=obs_ref.sel(time=times)
    svs=np.intersect1d(obs_ref.sv.values,obs_rov.sv.values)
    obs_rov=obs_rov.sel(sv=svs)
    obs_ref=obs_ref.sel(sv=svs)
    l1_rov=obs_rov['L1'].values
    l1_ref=obs_ref['L1'].values
    sd=l1_rov-l1_ref
    final_svs=[]
    for i in range(len(svs)):
        if not np.isnan(sd[:,i]).any():
            final_svs.append(i)
    svs=svs[final_svs]
    sd=sd[:,final_svs]
    obs_rov=obs_rov.sel(sv=svs)
    obs_ref=obs_ref.sel(sv=svs)
    return sd,times,svs,obs_ref,obs_rov

#sd,times,svs,obs_ref,obs_rov=compute_sd(obs_ref,obs_rov,t_deb=0,t_fin=100)
def findOffsets(eph,svs):
    offsets=np.zeros(len(svs))
    eph=eph.sel(sv=svs)
    for i in range(len(svs)):
        t=eph['SVclockBias'].values[:,i]
        t=np.average(t[~np.isnan(t)])
        offsets[i]=t
    return offsets

def computeEmissionTimes(times,svs,obs,offsets):
    R=obs['C1'].values
    emTimes=np.zeros((len(times),len(svs)))
    for i in range(len(svs)):
        emTimes[:,i]=times-R[:,i]/c-offsets[i]
    return emTimes
    
def computeFlightTimes(times,svs,obs,eph):
    offsets=findOffsets(eph,svs)
    times=convertTimes(times)
    times=times.astype('float64')
    times=times*1e-9
    emTimes=computeEmissionTimes(times,svs,obs,offsets)
    flightTimes=-(emTimes-times.reshape(-1,1))
    return flightTimes,times

def compute_dd(sd,to_vec=False):
    dd=sd[:,1:]-sd[:,0].reshape(-1,1)
    if to_vec:
        dd=dd.reshape(-1,)
    return dd

def convertTimes(times,date=np.datetime64('2019-05-05')):
    return times-date

def sigma(n,f=1575.42*10**6,x=0.05):
    # Estimates carrier measurement noise as x cycles (default 0.05 cycles for double difference)
    std=x*299792458/f
    Q=np.ones((n,n))+np.eye(n)
    return 2*std**2*Q

def float_solution(y,A,B):
    H=np.hstack((A,B))
    Q=sigma(len(y))
    Qi=np.linalg.inv(Q)
    Qhat=np.linalg.inv(np.dot(H.T,np.dot(Qi,H)))
    xhat=np.dot(Qhat,np.dot(H.T,np.dot(Qi,y)))
    #xhat=np.linalg.solve(np.dot(H.T,np.dot(Qi,H)),np.dot(H.T,np.dot(Qi,y)))
    return xhat,Qhat


def kepler_solve(M,e,eps=1e-15,maxiter=20):
    E=M
    En=E-(E-e*np.sin(E)-M)/(1-e*np.cos(E))
    i=1
    while np.max(np.abs(En-E))>eps and i<=maxiter:
        E=En
        En=E-(E-e*np.sin(E)-M)/(1-e*np.cos(E))
    return En

def createSatIndex(eph,svs):
    T=eph['time'].values
    satIndex=np.zeros(len(svs),dtype='int8')
    ex=eph['Eccentricity'].values
    for t in range(len(T)):
        for s in range(len(svs)):
            if not np.isnan(ex[t,s]):
                satIndex[s]=t
    return satIndex

def createSatIndex2(eph,svs,t0):
    T=eph['time'].values
    dist=np.abs(T-t0)
    ex=eph['Eccentricity'].values
    t_ind=[]
    b=False
    for s in range(len(svs)):
        for t in range(len(T)):
            if not np.isnan(ex[t,s]):
                if not b:
                    t_ind.append(t)
                    b=True
                elif dist[t]<dist[t_ind[s]]:
                    t_ind[s]=t
        b=False
    return np.array(t_ind,dtype='int8')

def get_pos(eph,t,satIndex,flightTimes,svs):
    #Computes satellite position in ECEF from ephemeris data for one satellite at a given date
    #See https://gssc.esa.int/navipedia/index.php/GPS_and_Galileo_Satellite_Coordinates_Computation
    we=7.2921151467*1e-5
    mu=3.986004418*1e14
    eph=eph.sel(sv=svs)
    Toe=eph['Toe'].values[satIndex,np.arange(len(svs))]
    M0=eph['M0'].values[satIndex,np.arange(len(svs))]
    sqrtA=eph['sqrtA'].values[satIndex,np.arange(len(svs))]
    w=eph['omega'].values[satIndex,np.arange(len(svs))]
    e=eph['Eccentricity'].values[satIndex,np.arange(len(svs))]
    i0=eph['Io'].values[satIndex,np.arange(len(svs))]
    omega0=eph['Omega0'].values[satIndex,np.arange(len(svs))]
    dN=eph['DeltaN'].values[satIndex,np.arange(len(svs))]
    idot=eph['IDOT'].values[satIndex,np.arange(len(svs))]
    omegadot=eph['OmegaDot'].values[satIndex,np.arange(len(svs))]
    cuc=eph['Cuc'].values[satIndex,np.arange(len(svs))]
    cus=eph['Cus'].values[satIndex,np.arange(len(svs))]
    crc=eph['Crc'].values[satIndex,np.arange(len(svs))]
    crs=eph['Crs'].values[satIndex,np.arange(len(svs))]
    cic=eph['Cic'].values[satIndex,np.arange(len(svs))]
    cis=eph['Cis'].values[satIndex,np.arange(len(svs))]
    #a0=eph['SVclockBias'].values[satIndex,np.arange(len(svs))]
    #a1=eph['SVclockDrift'].values[satIndex,np.arange(len(svs))]
    #a2=eph['SVclockDriftRate'].values[satIndex,np.arange(len(svs))]
    tk=t-Toe-flightTimes
    tk+=604800*(-1*(tk>302400)+(tk<-302400))
    Mk=M0+(np.sqrt(mu)/sqrtA**3+dN)*tk
    Ek=kepler_solve(Mk,e)
    #Ek=Mk+e*np.sin(Mk)
    nuk=np.arctan2(np.sqrt(1-e**2)*np.sin(Ek),(np.cos(Ek)-e))
    uk=w+nuk+cuc*np.cos(2*(w+nuk))+cus*np.sin(2*(w+nuk))
    rk=sqrtA**2*(1-e*np.cos(Ek))+crc*np.cos(2*(w+nuk))+crs*np.sin(2*(w+nuk))
    ik=i0+idot*tk+cic*np.cos(2*(w+nuk))+cis*np.sin(2*(w+nuk))
    ldak=omega0+(omegadot-we)*tk-we*Toe
    '''
    R3_lda=np.array([[np.cos(ldak),-np.sin(ldak),0],[np.sin(ldak),np.cos(ldak),0],[0,0,1]])
    R1_i=np.array([[1,0,0],[0,np.cos(ik),-np.sin(ik)],[0,np.sin(ik),np.cos(ik)]])
    R3_u=np.array([[np.cos(uk),-np.sin(uk),0],[np.sin(uk),np.cos(uk),0],[0,0,1]])
    pos=np.array([rk,0,0])
    pos=np.dot(R3_u,pos)
    pos=np.dot(R1_i,pos)
    pos=np.dot(R3_lda,pos)
    '''
    #print(rk)
    #print(uk)
    xk=rk*np.cos(uk)
    yk=rk*np.sin(uk)
    x=xk*np.cos(ldak)-yk*np.sin(ldak)*np.cos(ik)
    y=xk*np.sin(ldak)+yk*np.cos(ldak)*np.cos(ik)
    z=yk*np.sin(ik)
    return np.array((x,y,z)).T

def compute_LOS(sat_pos,x0=x0):
    los=sat_pos-x0
    los=los/np.linalg.norm(los,axis=1,keepdims=True)
    return los

def compute_rotation(theta):
    return np.array([[np.cos(theta), np.sin(theta), 0.], [-np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])

def correct_pos(sat_pos,flightTimes):
    theta=we*flightTimes
    n=len(flightTimes)
    rots=np.array([[np.cos(theta), np.sin(theta), np.zeros(n)], [-np.sin(theta), np.cos(theta), np.zeros(n)], [np.zeros(n), np.zeros(n), np.ones((n,))]])
    cpos=np.zeros(sat_pos.shape)
    for i in range(n):
        cpos[i]=np.dot(rots[:,:,i],sat_pos[i])
    return cpos


def compute_geomatrix(los,lda=lda):
    return -(los[1:]-los[0])/lda

def compute_G(eph_rov,t,satIndex,ft,svs):
    sat_pos=get_pos(eph_rov,t,satIndex,ft,svs)
    sat_pos=correct_pos(sat_pos,ft)
    los=compute_LOS(sat_pos,x0)
    G=compute_geomatrix(los,lda)
    return G



def full_solve(obs_ref,obs_rov,eph_rov,eph_ref,t_ind,x0=x0,lda=lda,f=1575.42*10**6,x=0.05,N_solve=0,N_trick=0):
    sd,times,svs,obs_ref,obs_rov=compute_sd(obs_ref,obs_rov,0,max(t_ind)+1)
    t0=times[t_ind[0]]
    flightTimes,times=computeFlightTimes(times,svs,obs_ref,eph_ref)
    eph_rov=eph_rov.sel(sv=svs)
    eph_ref=eph_ref.sel(sv=svs)
    satIndex=createSatIndex2(eph_ref,svs,t0)
    G=[]
    for t in t_ind:
        G.append(compute_G(eph_ref,times[t],satIndex,flightTimes[t],svs))
    n=len(svs)-1
    m=len(t_ind)
    A=np.zeros((m*n,n))
    for i in range(m):
        A[i*n:(i+1)*n,:]=np.eye(n)
    dd=compute_dd(sd)
    dd=dd-N_solve+N_trick
    G=np.concatenate(G,axis=0)
    #y=np.concatenate((dd[5],dd[100]))
    y=np.reshape(dd[t_ind,:],(-1,))
    xhat,Qhat=float_solution(y,A,G)
    return xhat,Qhat,G,A,y,svs,dd


def pos_from_integers(dd,integers,G,t_ind,x0=0):
    dd_cor=dd[t_ind]-integers
    Q=sigma(len(dd_cor))
    Qi=np.linalg.inv(Q)
    Qhat=np.linalg.inv(np.dot(G.T,np.dot(Qi,G)))
    xhat=np.dot(Qhat,np.dot(G.T,np.dot(Qi,dd_cor)))
    return xhat+x0
    
def apply_lambda(xhat,Qhat):
    ahat=xhat[:-3]
    Qahat=Qhat[:-3,:-3]
    Qahat=(Qahat.T+Qahat)/2
    print_lambda(ahat,Qahat)
    return ahat,Qahat

def test_cvx_sol(y,A,B):
    H=np.hstack((A,B))
    Q=sigma(len(y))
    Qi=np.linalg.inv(Q)
    sQi=scipy.linalg.sqrtm(Qi)
    xhat=cp.Variable(3,)
    Nhat=cp.Variable(A.shape[1],integer=True)
    x=cp.hstack([Nhat,xhat])
    #obj=cp.Minimize((y-B @ xhat - A @ Nhat).T @ Qi @ (y-B @ xhat - A @ Nhat))
    obj=cp.Minimize(cp.sum_squares(y-H@x))
    prob=cp.Problem(obj)
    prob.solve()
    return xhat.value,Nhat.value

def test_cvx(obs_ref,obs_rov,eph_rov,eph_ref,t_ind,x0=x0,lda=lda,f=1575.42*10**6,x=0.05,N_solve=0,N_trick=0):
    sd,times,svs,obs_ref,obs_rov=compute_sd(obs_ref,obs_rov,0,max(t_ind)+1)
    t0=times[t_ind[0]]
    flightTimes,times=computeFlightTimes(times,svs,obs_ref,eph_ref)
    eph_rov=eph_rov.sel(sv=svs)
    eph_ref=eph_ref.sel(sv=svs)
    satIndex=createSatIndex2(eph_ref,svs,t0)
    G=[]
    for t in t_ind:
        G.append(compute_G(eph_ref,times[t],satIndex,flightTimes[t],svs))
    n=len(svs)-1
    m=len(t_ind)
    A=np.zeros((m*n,n))
    for i in range(m):
        A[i*n:(i+1)*n,:]=np.eye(n)
    dd=compute_dd(sd)
    dd=dd-N_solve+N_trick
    G=np.concatenate(G,axis=0)
    #y=np.concatenate((dd[5],dd[100]))
    y=np.reshape(dd[t_ind,:],(-1,))
    xhat,Nhat=test_cvx_sol(y,A,G)
    return xhat,Nhat

def test_cvx2(ahat,Qahat):
    N=cp.Variable(len(ahat),integer=True)
    sq=scipy.linalg.sqrtm(Qahat)
    c=sq @ (N-ahat)
    o=cp.sum_squares(c)
    obj=cp.Minimize(cp.sum_squares(c))
    prob=cp.Problem(obj,[o<=10])
    prob.solve()
    return N.value
#xhat,Qhat,G,A,y,svs,dd=full_solve(obs_ref,obs_rov,eph_rov,eph_ref,[5,10,15])
