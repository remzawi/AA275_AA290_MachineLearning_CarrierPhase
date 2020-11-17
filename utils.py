import numpy as np
import xarray as xr     
import georinex as gr
import constants

def kepler_solve(M,e,eps=1e-8,maxiter=20):
    E=M
    En=E-(E-e*np.sin(E)-M)/(1-e*np.cos(E))
    i=1
    while np.abs(En-E)>eps and i<=maxiter:
        E=En
        En=E-(E-e*np.sin(E)-M)/(1-e*np.cos(E))
    return En

def get_sv(eph,intID=False):
    sv=eph['sv'].values
    if intID:
        sv_int=[int(s[i,1:]) for s in sv]
        return sv_int
    return sv

def get_visible(eph): #list(time,visible)
    T=eph['time'].values
    sv=eph['sv'].values
    SVs=[]
    ex=eph['Eccentricity'].values
    for t in range(len(T)):
        sv_t=[]
        for s in range(len(sv)):
            if not np.isnan(ex[t,s]):
                sv_t.append(sv[s])
        SVs.append(sv_t)
    return SVs

def createSatIndex(eph):
    satIndex={}
    T=eph['time'].values
    sv=eph['sv'].values
    ex=eph['Eccentricity'].values
    for t in range(len(T)):
        for s in range(len(sv)):
            if not np.isnan(ex[t,s]):
                satIndex[sv[s]]=(s,t)
    return satIndex



def get_pos(eph,sat,t,satIndex):
    #Computes satellite position in ECEF from ephemeris data for one satellite at a given date
    #See https://gssc.esa.int/navipedia/index.php/GPS_and_Galileo_Satellite_Coordinates_Computation
    we=7.2921151467*1e-5
    mu=3.986004418*1e14
    sind,tind=satIndex[sat]
    Toe=eph['Toe'].values[tind,sind]
    M0=eph['M0'].values[tind,sind]
    sqrtA=eph['sqrtA'].values[tind,sind]
    w=eph['omega'].values[tind,sind]
    e=eph['Eccentricity'].values[tind,sind]
    i0=eph['Io'].values[tind,sind]
    omega0=eph['Omega0'].values[tind,sind]
    dN=eph['DeltaN'].values[tind,sind]
    idot=eph['IDOT'].values[tind,sind]
    omegadot=eph['OmegaDot'].values[tind,sind]
    cuc=eph['Cuc'].values[tind,sind]
    cus=eph['Cus'].values[tind,sind]
    crc=eph['Crc'].values[tind,sind]
    crs=eph['Crs'].values[tind,sind]
    cic=eph['Cic'].values[tind,sind]
    cis=eph['Cis'].values[tind,sind]
    #a0=eph['SVclockBias'].values[tind,sind]
    #a1=eph['SVclockDrift'].values[tind,sind]
    #a2=eph['SVclockDriftRate'].values[tind,sind]
    tk=t-Toe
    if tk > 302400:
        tk-=604800
    elif tk < -302400:
        tk+=604800
    Mk=M0+(np.sqrt(mu)/sqrtA**3+dN)*tk
    Ek=kepler_solve(Mk,e)
    nuk=np.arctan(np.sqrt(1-e**2)*np.sin(Ek)/(np.cos(Ek)-e))
    uk=w+nuk+cuc*np.cos(2*(w+nuk))+cus*np.sin(2*(w+nuk))
    rk=sqrtA**2*(1-e*np.cos(Ek))+crc*np.cos(2*(w+nuk))+crs*np.sin(2*(w+nuk))
    ik=i0+idot*tk+cic*np.cos(2*(w+nuk))+cis*np.sin(2*(w+nuk))
    ldak=omega0+(omegadot-we)*tk-we*Toe
    R3_lda=np.array([[np.cos(ldak),-np.sin(ldak),0],[np.sin(ldak),np.cos(ldak),0],[0,0,1]])
    R1_i=np.array([[1,0,0],[0,np.cos(ik),-np.sin(ik)],[0,np.sin(ik),np.cos(ik)]])
    R3_u=np.array([[np.cos(uk),-np.sin(uk),0],[np.sin(uk),np.cos(uk),0],[0,0,1]])
    pos=np.array([rk,0,0])
    pos=np.dot(R3_u,pos)
    pos=np.dot(R1_i,pos)
    pos=np.dot(R3_lda,pos)
    return pos

def get_pos2(eph,sat,t,satIndex):
    #Computes satellite position in ECEF from ephemeris data for one satellite at a given date
    #See https://gssc.esa.int/navipedia/index.php/GPS_and_Galileo_Satellite_Coordinates_Computation
    we=7.2921151467*1e-5
    mu=3.986004418*1e14
    sind,tind=satIndex[sat]
    Toe=eph['Toe'].values[tind,sind]
    M0=eph['M0'].values[tind,sind]
    sqrtA=eph['sqrtA'].values[tind,sind]
    w=eph['omega'].values[tind,sind]
    e=eph['Eccentricity'].values[tind,sind]
    i0=eph['Io'].values[tind,sind]
    omega0=eph['Omega0'].values[tind,sind]
    dN=eph['DeltaN'].values[tind,sind]
    idot=eph['IDOT'].values[tind,sind]
    omegadot=eph['OmegaDot'].values[tind,sind]
    cuc=eph['Cuc'].values[tind,sind]
    cus=eph['Cus'].values[tind,sind]
    crc=eph['Crc'].values[tind,sind]
    crs=eph['Crs'].values[tind,sind]
    cic=eph['Cic'].values[tind,sind]
    cis=eph['Cis'].values[tind,sind]
    #a0=eph['SVclockBias'].values[tind,sind]
    #a1=eph['SVclockDrift'].values[tind,sind]
    #a2=eph['SVclockDriftRate'].values[tind,sind]
    tk=t-Toe
    if tk > 302400:
        tk-=604800
    elif tk < -302400:
        tk+=604800
    Mk=M0+(np.sqrt(mu)/sqrtA**3+dN)*tk
    Ek=kepler_solve(Mk,e)
    nuk=np.arctan(np.sqrt(1-e**2)*np.sin(Ek)/(np.cos(Ek)-e))
    uk=w+nuk+cuc*np.cos(2*(w+nuk))+cus*np.sin(2*(w+nuk))
    rk=sqrtA**2*(1-e*np.cos(Ek))+crc*np.cos(2*(w+nuk))+crs*np.sin(2*(w+nuk))
    ik=i0+idot*tk+cic*np.cos(2*(w+nuk))+cis*np.sin(2*(w+nuk))
    ldak=omega0+(omegadot-we)*tk-we*Toe
    R3_lda=np.array([[np.cos(ldak),-np.sin(ldak),0],[np.sin(ldak),np.cos(ldak),0],[0,0,1]])
    R1_i=np.array([[1,0,0],[0,np.cos(ik),-np.sin(ik)],[0,np.sin(ik),np.cos(ik)]])
    R3_u=np.array([[np.cos(uk),-np.sin(uk),0],[np.sin(uk),np.cos(uk),0],[0,0,1]])
    pos=np.array([rk,0,0])
    pos=np.dot(R3_u,pos)
    pos=np.dot(R1_i,pos)
    pos=np.dot(R3_lda,pos)
    return pos

def flight_time_correct(X, Y, Z, flight_time):
    theta = constants.WE * flight_time/1e6
    R = np.array([[np.cos(theta), np.sin(theta), 0.], [-np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])

    XYZ = np.array([X, Y, Z])
    rot_XYZ = R @  np.expand_dims(XYZ, axis=-1)
    return rot_XYZ[0], rot_XYZ[1], rot_XYZ[2]