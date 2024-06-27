import numpy as np
from numpy import pi, sqrt, sin, cos, exp, log10, array, real, conj

def int_gauss(n,k_lim):
    beta = np.zeros(n,dtype=float)

    for k in range(1,n+1):
        beta[k-1] = 0.5/np.sqrt(1-(2*(k))**(-2))

    m = n+1
    T_low = np.zeros((m,m))
    T_up = np.zeros((m,m))
    T = np.zeros((m,m))
    # T = np.zeros((m**2,m/3)) #invento Fran
    Q = 1

    # defino T_low
    for i in range(0,m):
        for j in range(0,m):
            if i==j+1:
                T_low[i,j]=beta[i-1]
    
    # defino T_up
    for i in range(0,m):
        for j in range(0,m):
            if j==i+1:
                T_up[i,j]=beta[i]

    T = T_low + T_up        
    d_,V = np.linalg.eig(T)
    D = np.zeros((m,m))
    
    for i in range(0,m):
        for j in range(0,m):
            if i==j:
                D[i,j]= k_lim*d_[i]
                
    W = (2*V[0,:]**2)
    Wt = np.kron(W,W)

    r = k_lim*d_
    X_IG,Y_IG = np.meshgrid(r,r)
    
    return {'Wt': Wt, 'X_IG': X_IG, 'Y_IG': Y_IG, 'm_gauss': m}

## Parámetros Globales ##

#onda incidente
def kiz(k0,thi): 
    return k0*cos(thi)

def kix(k0,thi,phi):
    return k0*sin(thi)*cos(phi)

def kiy(k0,thi,phi):
    return k0*sin(thi)*sin(phi)

def kir(k0,thi):
    return k0*sin(thi)

#onda dispersada
def kx(k0,th,ph):
    return k0*sin(th)*cos(ph)

def ky(k0,th,ph):
    return k0*sin(th)*sin(ph)

def k0z(k0,th):
    return k0*cos(th)
    
#ondas transmitidas
def k1z(k0,th,ep1):
    return k0*sqrt(ep1-sin(th)**2)

def k2z(k0,th,ep2):
    return k0*sqrt(ep2-sin(th)**2)

##kz interface plana

def k1zi(k0,thi,ep1):
    return k0*sqrt(ep1-sin(thi)**2)

def k2zi(k0,thi,ep2):
    return k0*sqrt(ep2-sin(thi)**2)

#componente tangencial
def kr(k0,th):
    return k0*sin(th)

#-------------------------------------------#

## CANAL HH ##
def a0HH(k0,thi,ep1,ep2,d):
    k1zi_ = k1zi(k0,thi,ep1)
    k2zi_ = k2zi(k0,thi,ep2)
    kiz_ = kiz(k0,thi)
    
    sin_ = np.sin(d*k1zi_)
    cos_ = np.cos(d*k1zi_)
    k2i = k2zi_*kiz_

    numerador =  (-1j*k1zi_*(k2zi_ - kiz_)*cos_ + (-k1zi_**2 + k2i)*sin_)
    denominador = (1j*k1zi_*(k2zi_ + kiz_)*cos_ + (k1zi_**2 + k2i)*sin_)
    
    return numerador/denominador

def bb0HH(k0,thi,ep1,ep2,d):
    k1zi_ = k1zi(k0,thi,ep1)
    k2zi_ = k2zi(k0,thi,ep2)
    kiz_ = kiz(k0,thi)
    
    numerador = (2*kiz_)
    denominador = (-k1zi_ + kiz_ + ((k1zi_ + k2zi_)*(k1zi_ + kiz_))/(exp(2j*d*k1zi_)*(k1zi_ - k2zi_)))
    return numerador/denominador

def b0HH(k0, thi, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    kiz_ = kiz(k0, thi)
    
    exp_2idk1zi = np.exp(2 * 1j * d * k1zi_)
    k1zi_plus_k2zi = k1zi_ + k2zi_
    k1zi_minus_k2zi = k1zi_ - k2zi_
    k1zi_minus_kiz = k1zi_ - kiz_
    k1zi_plus_kiz = k1zi_ + kiz_
    
    numerator = -2 * k1zi_plus_k2zi * kiz_
    denominator = (exp_2idk1zi * k1zi_minus_k2zi * k1zi_minus_kiz - k1zi_plus_k2zi * k1zi_plus_kiz)
    
    return numerator / denominator

def c0HH(k0, thi, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    kiz_ = kiz(k0, thi)
    
    exp_idk1zi_minus_k2zi = np.exp(1j * d * (k1zi_ - k2zi_))
    exp_2idk1zi = np.exp(2 * 1j * d * k1zi_)
    k1zi_minus_k2zi = k1zi_ - k2zi_
    k1zi_minus_kiz = k1zi_ - kiz_
    k1zi_plus_k2zi = k1zi_ + k2zi_
    k1zi_plus_kiz = k1zi_ + kiz_
    
    numerator = -4 * exp_idk1zi_minus_k2zi * k1zi_ * kiz_
    denominator = (exp_2idk1zi * k1zi_minus_k2zi * k1zi_minus_kiz - k1zi_plus_k2zi * k1zi_plus_kiz)
    
    return numerator / denominator


## CANAL VH ##

a0VH = 0
bb0VH = 0
b0VH = 0
c0VH = 0

## CANAL VV ##
def a0VV(k0, thi, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    kiz_ = kiz(k0, thi)
    
    sin_ = np.sin(d * k1zi_)
    cos_ = np.cos(d * k1zi_)
    k2i = k2zi_ * kiz_
    k1zi2 = k1zi_ ** 2
    ep1_sq = ep1 ** 2
    
    numerator = (-1j * k1zi_ * ep1 * (k2zi_ - kiz_ * ep2) * cos_ + (k2i * ep1_sq - k1zi2 * ep2) * sin_)
    denominator = (1j * k1zi_ * ep1 * (k2zi_ + kiz_ * ep2) * cos_ + (k2i * ep1_sq + k1zi2 * ep2) * sin_)
    
    return numerator / denominator

def bb0VV(k0, thi, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    kiz_ = kiz(k0, thi)
    
    exp_2idk1zi = np.exp(2 * 1j * d * k1zi_)
    sqrt_ep1 = np.sqrt(ep1)
    k1zi_minus_kiz_ep1 = k1zi_ - kiz_ * ep1
    k2zi_ep1_minus_k1zi_ep2 = k2zi_ * ep1 - k1zi_ * ep2
    k1zi_plus_kiz_ep1 = k1zi_ + kiz_ * ep1
    k2zi_ep1_plus_k1zi_ep2 = k2zi_ * ep1 + k1zi_ * ep2
    
    numerator = 2 * exp_2idk1zi * kiz_ * sqrt_ep1 * k2zi_ep1_minus_k1zi_ep2
    denominator = (exp_2idk1zi * k1zi_minus_kiz_ep1 * (-k2zi_ep1_minus_k1zi_ep2) - k1zi_plus_kiz_ep1 * k2zi_ep1_plus_k1zi_ep2)
    
    return numerator / denominator

def b0VV(k0, thi, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    kiz_ = kiz(k0, thi)
    
    exp_2idk1zi = np.exp(2 * 1j * d * k1zi_)
    sqrt_ep1 = np.sqrt(ep1)
    k1zi_minus_kiz_ep1 = k1zi_ - kiz_ * ep1
    k2zi_ep1_minus_k1zi_ep2 = -k2zi_ * ep1 + k1zi_ * ep2
    k1zi_plus_kiz_ep1 = k1zi_ + kiz_ * ep1
    k2zi_ep1_plus_k1zi_ep2 = k2zi_ * ep1 + k1zi_ * ep2
    
    numerator = -2 * kiz_ * sqrt_ep1 * k2zi_ep1_plus_k1zi_ep2
    denominator = (exp_2idk1zi * k1zi_minus_kiz_ep1 * k2zi_ep1_minus_k1zi_ep2 - k1zi_plus_kiz_ep1 * k2zi_ep1_plus_k1zi_ep2)
    
    return numerator / denominator

def c0VV(k0, thi, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    kiz_ = kiz(k0, thi)
    
    exp_idk1zi_minus_k2zi = np.exp(1j * d * (k1zi_ - k2zi_))
    exp_2idk1zi = np.exp(2 * 1j * d * k1zi_)
    sqrt_ep2 = np.sqrt(ep2)
    k1zi_minus_kiz_ep1 = k1zi_ - kiz_ * ep1
    k2zi_ep1_minus_k1zi_ep2 = -k2zi_ * ep1 + k1zi_ * ep2
    k1zi_plus_kiz_ep1 = k1zi_ + kiz_ * ep1
    k2zi_ep1_plus_k1zi_ep2 = k2zi_ * ep1 + k1zi_ * ep2
    
    numerator = -4 * exp_idk1zi_minus_k2zi * k1zi_ * kiz_ * ep1 * sqrt_ep2
    denominator = (exp_2idk1zi * k1zi_minus_kiz_ep1 * k2zi_ep1_minus_k1zi_ep2 - k1zi_plus_kiz_ep1 * k2zi_ep1_plus_k1zi_ep2)
    
    return numerator / denominator

## CANAL HV ##

a0HV = 0
bb0HV = 0
b0HV = 0
c0HV = 0

###### Coeficientes de orden uno ######


# **Coeficientes aF1: asociados a las ondas dispersadas en +z en la region 0 y proporcionales
# a el espectro de rugosisdad de la primera capa**

# **Coeficientes aF2: asociados a las ondas dispersadas en +z en la region 0 y proporcionales
# a el espectro de rugosisdad de la segunda capa**

# **Lo mismo vale para los coeficientes 'bb', 'b' y 'c'**

## CANAL HH ##
# AMPLITUD ONDA DISPERSADA #

def a1HHF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0,thi,ep1)
    k1z_ = k1z(k0,th,ep1)
    k2z_ = k2z(k0,th,ep2)
    kiz_ = kiz(k0,thi)
    kir_ = kir(k0,thi)
    k0z_ = k0z(k0,th)
    kr_ = kr(k0,th)  

    kix_ = kix(k0,thi,phi)
    kiy_ = kiy(k0,thi,phi)
    kx_ = kx(k0,th,ph)
    ky_ = ky(k0,th,ph)
    sumXY = kix_*kx_ + kiy_*ky_

    exp1 = np.exp(1j*d*(2*k1z_ + k1zi_))
    sq1 = np.sqrt(ep1)
    sq2 = np.sqrt(ep2)
    sq12 = sq1*sq2

    a0HH_ = a0HH(k0,thi,ep1,ep2,d)
    bb0HH_ = bb0HH(k0,thi,ep1,ep2,d)
    b0HH_ = b0HH(k0,thi,ep1,ep2,d)

    numerador = 1j*(kr_*((1 + a0HH_)*exp1*(kix_*(kiz_**2 + kir_**2)*kx_ + kiy_*kiz_**2*ky_ - kir_**2*(kx_**2 - kiy_*ky_ + ky_**2))*sq1*(k1z_*sq12 - k2z_*sq12)-\
        exp1*((-1 + a0HH_)*k1z_*kiz_*(sumXY) + bb0HH_*(k1z_**2*(sumXY) - k1z_*k1zi_*(sumXY) + kir_**2*(kix_*kx_ - kx_**2 + (kiy_ - ky_)*ky_)) + \
        b0HH_*(k1z_*k1zi_*(sumXY) + k1zi_**2*(sumXY) + kir_**2*(kix_*kx_ - kx_**2 + (kiy_ - ky_)*ky_)))*sq1*(k1z_*sq12 - k2z_*sq12) + \
        exp(1j*d*k1zi_)*((1 + a0HH_)*(kix_*(kiz_**2 + kir_**2)*kx_ + kiy_*kiz_**2*ky_ - kir_**2*(kx_**2 - kiy_*ky_ + ky_**2))*sq1 - \
        (-((-1 + a0HH_)*k1z_*kiz_*(sumXY)) + bb0HH_*(k1z_**2*(sumXY) + k1z_*k1zi_*(sumXY) + kir_**2*(kix_*kx_ - kx_**2 + (kiy_ - ky_)*ky_)) + \
        b0HH_*(-(k1z_*k1zi_*(sumXY)) + k1zi_**2*(sumXY) + kir_**2*(kix_*kx_ - kx_**2 + (kiy_ - ky_)*ky_)))*sq1)*(k1z_*sq12 + k2z_*sq12)))
    
    denominador = (exp(1j*d*k1zi_)*kir_*(kx_**2 + ky_**2)*(k0z_*sq1*((1 + exp(2j*d*k1z_))*k1z_*sq12 - (-1 + exp(2j*d*k1z_))*k2z_*sq12) + \
        k1z_*sq1*(-((-1 + exp(2j*d*k1z_))*k1z_*sq12) + (1 + exp(2j*d*k1z_))*k2z_*sq12)))
    
    return numerador/denominador

def a1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    sumXY = kix_ * kx_ + kiy_ * ky_
    
    exp1 = np.exp(1j * d * (k1z_ + 2 * k1zi_))
    exp2 = np.exp(1j * d * k1z_)
    exp3 = np.exp(1j * d * (k1z_ + k1zi_ + k2zi_))
    exp4 = np.exp(1j * d * k1zi_)
    exp5 = np.exp(2 * 1j * d * k1z_)
    
    sq1 = np.sqrt(ep1)
    sq2 = np.sqrt(ep2)
    sq12 = sq1 * sq2
    
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    c0HH_ = c0HH(k0, thi, ep1, ep2, d)
    
    numerador = -1j * (kr_ * (
        2 * b0HH_ * exp1 * k1z_ * sq1 * (
            k1zi_**2 * sumXY * sq12 - k1zi_ * k2z_ * sumXY * sq12 +
            kir_**2 * (kix_ * kx_ - kx_**2 + (kiy_ - ky_) * ky_) * sq12
        ) +
        2 * bb0HH_ * exp2 * k1z_ * sq1 * (
            k1zi_**2 * sumXY * sq12 + k1zi_ * k2z_ * sumXY * sq12 +
            kir_**2 * (kix_ * kx_ - kx_**2 + (kiy_ - ky_) * ky_) * sq12
        ) -
        2 * c0HH_ * exp3 * k1z_ * ep1 * (
            -k2z_ * k2zi_ * sumXY + k2zi_**2 * sumXY +
            kir_**2 * (kix_ * kx_ - kx_**2 + (kiy_ - ky_) * ky_)
        ) * np.sqrt(ep2)
    ))
    
    denominador = (
        exp4 * kir_ * (kx_**2 + ky_**2) * (
            k0z_ * sq1 * (
                (1 + exp5) * k1z_ * sq12 - (-1 + exp5) * k2z_ * sq12
            ) +
            k1z_ * sq1 * (
                -((-1 + exp5) * k1z_ * sq12) + (1 + exp5) * k2z_ * sq12
            )
        )
    )
    
    return numerador / denominador

def bb1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)  
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    sumXY = kix_ * kx_ + kiy_ * ky_

    exp1 = np.exp(1j * d * (k1z_ + k1zi_))
    sq1 = np.sqrt(ep1)
    sq2 = np.sqrt(ep2)
    sq12 = sq1 * sq2

    a0HH_ = a0HH(k0, thi, ep1, ep2, d)
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)

    numerador = 1j * (
        np.exp(1j * d * (k1z_ - k1zi_)) * kr(k0, th) * (
            -np.exp(1j * d * (k1z_ + k1zi_)) * (
                (-1 + a0HH_) * k0z_ * kiz_ * sumXY -
                (1 + a0HH_) * (
                    kix_ * (kiz_**2 + kir_**2) * kx_ + 
                    kiy_ * kiz_**2 * ky_ - 
                    kir_**2 * (kx_**2 - kiy_ * ky_ + ky_**2)
                )
            ) * sq1 * (k1z_ * sq12 - k2z_ * sq12) -
            bb0HH_ * exp1 * (
                k0z_ * k1zi_ * sumXY * sq1 - 
                (k1z_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq1
            ) * (-k1z_ * sq12 + k2z_ * sq12) +
            b0HH_ * exp1 * (
                k0z_ * k1zi_ * sumXY * sq1 + 
                (k1zi_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq1
            ) * (-k1z_ * sq12 + k2z_ * sq12)
        )
    )

    denominador = (
        kir_ * (kx_**2 + ky_**2) *
        (np.exp(2 * 1j * d * k1z_) * (k0z_ * sq1 - k1z_ * sq1) * (-k1z_ * sq12 + k2z_ * sq12) -
         (k0z_ * sq1 + k1z_ * sq1) * (k1z_ * sq12 + k2z_ * sq12))
    )

    return numerador / denominador

def bb1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)  
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    sumXY = kix_ * kx_ + kiy_ * ky_

    exp1 = np.exp(1j * d * (k1z_ - k1zi_))
    sq1 = np.sqrt(ep1)
    sq2 = np.sqrt(ep2)
    sq12 = sq1 * sq2

    b0HH_ = b0HH(k0, thi, ep1, ep2, d)
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    c0HH_ = c0HH(k0, thi, ep1, ep2, d)

    numerador = 1j * (
        exp1 * kr(k0, th) * (
            -b0HH_ * np.exp(2 * 1j * d * k1zi_) * (k0z_ * sq1 + k1z_ * sq1) * (
                k1zi_ * k2z_ * sumXY * sq12 -
                (k1zi_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq12
            ) +
            bb0HH_ * (k0z_ * sq1 + k1z_ * sq1) * (
                k1zi_ * k2z_ * sumXY * sq12 +
                (k1zi_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq12
            ) +
            c0HH_ * np.exp(1j * d * (k1zi_ + k2zi_)) * (
                k2z_ * k2zi_ * sumXY -
                k2zi_**2 * sumXY +
                kir_**2 * (-(kix_ * kx_) + kx_**2 - kiy_ * ky_ + ky_**2)
            ) * (k0z_ * sq1 + k1z_ * sq1) * sq12
        )
    )

    denominador = (
        kir_ * (kx_**2 + ky_**2) *
        (np.exp(2 * 1j * d * k1z_) * (k0z_ * sq1 - k1z_ * sq1) * (-k1z_ * sq12 + k2z_ * sq12) -
         (k0z_ * sq1 + k1z_ * sq1) * (k1z_ * sq12 + k2z_ * sq12))
    )

    return numerador / denominador

def b1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    sumXY = kix_ * kx_ + kiy_ * ky_

    exp1 = np.exp(1j * d * k1zi_)
    exp2 = np.exp(2 * 1j * d * k1z_)
    sq1 = np.sqrt(ep1)
    sq2 = np.sqrt(ep2)
    sq12 = sq1 * sq2

    a0HH_ = a0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)

    term1 = bb0HH_ * exp1 * (k0z_ * k1zi_ * sumXY * sq1 - (k1z_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq1) * (k1z_ * sq12 + k2z_ * sq12)
    term2 = b0HH_ * exp1 * (k0z_ * k1zi_ * sumXY * sq1 + (k1zi_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq1) * (k1z_ * sq12 + k2z_ * sq12)
    term3 = exp1 * ((-1 + a0HH_) * k0z_ * kiz_ * sumXY - (1 + a0HH_) * (kix_ * (kiz_**2 + kir_**2) * kx_ + kiy_ * kiz_**2 * ky_ - kir_**2 * (kx_**2 - kiy_ * ky_ + ky_**2))) * sq1 * (k1z_ * sq12 + k2z_ * sq12)

    numerador = 1j * kr(k0, th) * (term1 - term2 - term3)

    denominador = (
        exp1 * kir_ * (kx_**2 + ky_**2) *
        (exp2 * (k0z_ * sq1 - k1z_ * sq1) * (-k1z_ * sq12 + k2z_ * sq12) -
         (k0z_ * sq1 + k1z_ * sq1) * (k1z_ * sq12 + k2z_ * sq12))
    )

    return numerador / denominador

def b1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    sumXY = kix_ * kx_ + kiy_ * ky_

    exp1 = np.exp(1j * d * k1zi_)
    exp2 = np.exp(2 * 1j * d * k1z_)
    sq1 = np.sqrt(ep1)
    sq2 = np.sqrt(ep2)
    sq12 = sq1 * sq2

    b0HH_ = b0HH(k0, thi, ep1, ep2, d)
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    c0HH_ = c0HH(k0, thi, ep1, ep2, d)

    term1 = b0HH_ * np.exp(1j * d * (k1z_ + 2 * k1zi_)) * (k0z_ * sq1 - k1z_ * sq1) * (k1zi_ * k2z_ * sumXY * sq12 - (k1zi_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq12)
    term2 = bb0HH_ * np.exp(1j * d * k1z_) * (k0z_ * sq1 - k1z_ * sq1) * (k1zi_ * k2z_ * sumXY * sq12 + (k1zi_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq12)
    term3 = c0HH_ * np.exp(1j * d * (k1z_ + k1zi_ + k2zi(k0, thi, ep2))) * (-(k2z_ * k2zi(k0, thi, ep2) * sumXY) + k2zi(k0, thi, ep2)**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * (k0z_ * sq1 - k1z_ * sq1) * sq12

    denominator = exp1 * kir_ * (kx_**2 + ky_**2) * (exp2 * (k0z_ * sq1 - k1z_ * sq1) * (-k1z_ * sq12 + k2z_ * sq12) - (k0z_ * sq1 + k1z_ * sq1) * (k1z_ * sq12 + k2z_ * sq12))

    return 1j * kr(k0, th) * (term1 - term2 + term3) / denominator

def c1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    sumXY = kix_ * kx_ + kiy_ * ky_

    exp1 = np.exp(1j * d * (k1z_ + k1zi_))
    exp2 = np.exp(1j * d * (k1z_ + 2 * k1zi_))
    exp3 = np.exp(1j * d * (k1zi_ + k2z_))
    exp4 = np.exp(2 * 1j * d * k1z_)
    sq1 = np.sqrt(ep1)
    sq2 = np.sqrt(ep2)
    sq12 = sq1 * sq2

    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    a0HH_ = a0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)

    term1 = -2 * bb0HH_ * exp1 * k1z_ * (-(k0z_ * k1zi_ * sumXY * sq1) + (k1z_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq1) * sq12
    term2 = 2 * exp1 * k1z_ * (-( (-1 + a0HH_) * k0z_ * kiz_ * sumXY) + (1 + a0HH_) * (kix_ * (kiz_**2 + kir_**2) * kx_ + kiy_ * kiz_**2 * ky_ - kir_**2 * (kx_**2 - kiy_ * ky_ + ky_**2))) * ep1 * sq2
    term3 = -2 * b0HH_ * exp2 * k1z_ * (k0z_ * k1zi_ * sumXY * sq1 + (k1zi_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq1) * sq12 * np.cos(d * k1zi_)
    term4 = 2j * b0HH_ * exp2 * k1z_ * (k0z_ * k1zi_ * sumXY * sq1 + (k1zi_**2 * sumXY + kir_**2 * ((kix_ - kx_) * kx_ + (kiy_ - ky_) * ky_)) * sq1) * sq12 * np.sin(d * k1zi_)

    numerator = term1 + term2 + term3 + term4
    denominator = exp3 * kir_ * (kx_**2 + ky_**2) * (exp4 * (k0z_ * sq1 - k1z_ * sq1) * (-k1z_ * sq12 + k2z_ * sq12) - (k0z_ * sq1 + k1z_ * sq1) * (k1z_ * sq12 + k2z_ * sq12))

    return 1j * kr(k0, th) * numerator / denominator

def c1HHF2_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    sumXY = kix_ * kx_ + kiy_ * ky_

    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    c0HH_ = c0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )

    sq12 = sq1 * sq2

    kxsum = kix_-kx_
    kysum = kiy_-ky_

    return 1j*(kr(k0,th)*(bb0HH_*exp(2j*d*k1z_)*(k1z_*k1zi_*(sumXY) + k1zi_**2*(sumXY) + kir_**2*((kxsum)*kx_ + (kysum)*ky_))*(-(k0z_*sq1) + k1z_*sq1)*sq12 - \
            bb0HH_*(k1z_*k1zi_*(sumXY) - k1zi_**2*(sumXY) + kir_**2*(-(kix_*kx_) + kx_**2 - kiy_*ky_ + ky_**2))*(k0z_*sq1 + k1z_*sq1)*sq12 + \
            2*b0HH_*exp(1j*d*(k1z_ + 2*k1zi_))*k1z_*(k0z_*k1zi_*(sumXY)*sq1 + (k1zi_**2*(sumXY) + kir_**2*((kxsum)*kx_ + (kysum)*ky_))*sq1)*sq12*cos(d*k1z_) - \
            2j*b0HH_*exp(1j*d*(k1z_ + 2*k1zi_))*(k1z_**2*k1zi_*(sumXY)*sq1 + k0z_*(k1zi_**2*(sumXY) + kir_**2*((kxsum)*kx_ + (kysum)*ky_))*sq1)*sq12*\
            sin(d*k1z_) + 2j*c0HH_*exp(1j*d*(k1z_ + k1zi_ + k2zi_))*(1j*k1z_*ep1*(k0z_*k2zi_*(sumXY)*sq2 + (k2zi_**2*(sumXY) + kir_**2*((kxsum)*kx_ + (kysum)*ky_))*sq2)*\
            cos(d*k1z_) + (k1z_**2*k2zi_*(sumXY)*ep1*sq2 + k0z_*(k2zi_**2*(sumXY) + kir_**2*((kxsum)*kx_ + (kysum)*ky_))*ep1*sq2)*sin(d*k1z_))))/\
            (exp(1j*d*(k1zi_ + k2z_))*kir_*(kx_**2 + ky_**2)*(exp(2j*d*k1z_)*(k0z_*sq1 - k1z_*sq1)*(-(k1z_*sq12) + k2z_*sq12) - 
            (k0z_*sq1 + k1z_*sq1)*(k1z_*sq12 + k2z_*sq12))) 

## CANAL VH ##
# AMPLITUD ONDA DISPERSADA #
def a1VHF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kiz_ = kiz(k0, thi)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    sumXY = kix_ * kx_ + kiy_ * ky_

    a0HH_ = a0HH(k0, thi, ep1, ep2, d)
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )

    sq12 = sq1 * sq2

    kxsum = kix_-kx_
    kysum = kiy_-ky_

    
    return -1j*(((kiy(k0,thi,phi)*kx_ - kix(k0,thi,phi)*ky_)*kr(k0,th)*(exp(1j*d*(2*k1z_ + k1zi_))*(-((1 + a0HH_)*k1z_*(kiz_**2 + kir_**2)*sq1) + k1z_*(bb0HH_*k1z_**2 + b0HH_*k1zi_**2 + (bb0HH_ + b0HH_)*kir_**2)*sq1 + \
            (-bb0HH_ + b0HH_)*k1zi_ + (-1 + a0HH_)*kiz_)*ep1**1.5)*sq2*(-(k2z_*ep1) + k1z_*ep2) + \
            exp(1j*d*k1zi_)*((1 + a0HH_)*k1z_*(kiz_**2 + kir_**2)*sq1 - k1z_*(bb0HH_*k1z_**2 + b0HH_*k1zi_**2 + (bb0HH_ + b0HH_)*kir_**2)*sq1 + ((-bb0HH_ + b0HH_)*k1zi_ + (-1 + a0HH_)*kiz_)*ep1**1.5)*sq2*\
            (k2z_*ep1 + k1z_*ep2)))/(exp(1j*d*k1zi_)*kir_*(kx_**2 + ky_**2)*sq1*sq2*(exp(2j*d*k1z_)*(k1z_ - k0z_*ep1)*(-(k2z_*ep1) +\
            k1z_*ep2) - (k1z_ + k0z_*ep1)*(k2z_*ep1 +k1z_*ep2)))

def a1VHF2_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)

    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)
    sq12 = sq1 * sq2

    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)
    c0HH_ = c0HH(k0, thi, ep1, ep2, d)

    prod1 = k1zi_ * sq1 * ep2
    sum1 = k1zi_**2 + kir_**2
    
    return -1j*((kiy_*kx_ - kix_*ky_)*kr(k0,th)*(-2*c0HH_*exp(1j*d*(k1z_ + k1zi_ + k2zi_))*k1z_*ep1**1.5*sq2*(k2z_*(k2zi_**2 + kir_**2) - k2zi_*ep2) + \
            2*b0HH_*exp(1j*d*(k1z_ + 2*k1zi_))*k1z_*ep1*sq2*(k2z_*(sum1)*sq1 - prod1) + \
            2*bb0HH_*exp(1j*d*k1z_)*k1z_*ep1*sq2*(k2z_*(sum1)*sq1 + prod1)))/\
            (exp(1j*d*k1zi_)*kir_*(kx_**2 + ky_**2)*sq1*sq2*(exp(2j*d*k1z_)*(k1z_ - k0z_*ep1)*(-(k2z_*ep1) + k1z_*ep2) - (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2)))

def bb1VHF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )

    sq12 = sq1 * sq2
    a0HH_ = a0HH(k0, thi, ep1, ep2, d)
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)

    sum1 = k2z_* ep1 - k1z_* ep2
    
    return 1j*(exp(1j*d*(k1z_ - k1zi_))*(kiy_*kx_ - kix_*ky_)*kr(k0,th)*(bb0HH_*exp(1j*d*(k1z_ + k1zi_))*(k1zi_*sq1 - k0z_*(k1z_**2 + kir_**2)*sq1)*sq2*(sum1) + \
            b0HH_*exp(1j*d*(k1z_ + k1zi_))*(k1zi_*sq1 + k0z_*(k1zi_**2 + kir_**2)*sq1)*sq2*(-(k2z_*ep1) + k1z_*ep2) + \
            exp(1j*d*(k1z_ + k1zi_))*((-1 + a0HH_)*kiz(k0,thi) - (1 + a0HH_)*k0z_*(kiz(k0,thi)**2 + kir_**2))*sq1*sq2*(-(k2z_*ep1) + k1z_*ep2)))/\
            (kir_*(kx_**2 + ky_**2)*sq2*(exp(2*1j*d*k1z_)*(k1z_ - k0z_*ep1)*(sum1) + (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2))) 

def bb1VHF2_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)

    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)

    sq12 = sq1 * sq2
    
    
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)
    c0HH_ = c0HH(k0, thi, ep1, ep2, d)

    sum1 = k1z_ + k0z_*ep1
    
    return 1j*(exp(1j*d*(k1z_ - k1zi_))*(kiy_*kx_ - kix_*ky_)*kr(k0,th)*(-(c0HH_*exp(1j*d*(k1zi_ + k2zi_))*sq1*(sum1)*sq2*(k2z_*(k2zi_**2 + kir_**2) - k2zi_*ep2)) + \
            b0HH_*exp(2*1j*d*k1zi_)*(sum1)*sq2*(k2z_*(k1zi_**2 + kir_**2)*sq1 - k1zi_*sq1*ep2) + bb0HH_*(sum1)*sq2*(k2z_*(k1zi_**2 + kir_**2)*sq1 + k1zi_*sq1*ep2)))/\
            (kir_*(kx_**2 + ky_**2)*sq2*(exp(2*1j*d*k1z_)*(k1z_ - k0z_*ep1)*(k2z_*ep1 - k1z_*ep2) + (sum1)*(k2z_*ep1 + k1z_*ep2)))

def b1VHF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
   
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)

    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)

    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)

    sq12 = sq1 * sq2

    a0HH_ = a0HH(k0, thi, ep1, ep2, d)
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)

    exp1 = exp(1j *d * k1zi_)
    
    return  -1j*(((kiy_*kx_ - kix_*ky_)*kr(k0,th)*(bb0HH_*exp1*(k1zi_*sq1 - k0z_*(k1z_**2 + kir_**2)*sq1)*sq2*(k2z_*ep1 + k1z_*ep2) - \
            b0HH_*exp1*(k1zi_*sq1 + k0z_*(k1zi_**2 + kir_**2)*sq1)*sq2*(k2z_*ep1 + k1z_*ep2) - \
            exp1*((-1 + a0HH_)*kiz_ - (1 + a0HH_)*k0z_*(kiz_**2 + kir_**2))*sq1*sq2*(k2z_*ep1 + k1z_*ep2)))/\
            (exp1*kir_*(kx_**2 + ky_**2)*sq2*(exp(2*1j*d*k1z_)*(k1z_ - k0z_*ep1)*(k2z_*ep1 - k1z_*ep2) + (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2)))) 

def b1VHF2_j(k0,thi,phi,th,ph,ep1,ep2,d):
  
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)

    sq12 = sq1 * sq2
    
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)
    c0HH_ = c0HH(k0, thi, ep1, ep2, d)

    suma1 = k1z_ - k0z_ *ep1
    return -1j*((kiy_*kx_ - kix_*ky_)*kr(k0,th)*(c0HH_*exp(1j*d*(k1z_ + k1zi_ + k2zi_))*sq1*(suma1)*sq2*(k2z_*(k2zi_**2 + kir_**2) - k2zi_*ep2) + \
            b0HH_*exp(1j*d*(k1z_ + 2*k1zi_))*(suma1)*sq2*(-(k2z_*(k1zi_**2 + kir_**2)*sq1) + k1zi_*sq1*ep2) - \
            bb0HH_*exp(1j*d*k1z_)*(suma1)*sq2*(k2z_*(k1zi_**2 + kir_**2)*sq1 + k1zi_*sq1*ep2)))/\
            (exp(1j*d*k1zi_)*kir_*(kx_**2 + ky_**2)*sq2*(exp(2*1j*d*k1z_)*(suma1)*(k2z_*ep1 - k1z_*ep2) + (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2)))

def c1VHF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )

    sq12 = sq1 * sq2
    
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)

    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)
    a0HH_ = a0HH(k0, thi, ep1, ep2, d) 
    
    return -1j*(((kiy_*kx_ - kix_*ky_)*kr(k0,th)*(-2*bb0HH_*exp(1j*d*(k1z_ + k1zi_))*k1z_*(-(k1zi_*sq1) + k0z_*(k1z_**2 + kir_**2)*sq1)*ep1*sq2 - \
            2*b0HH_*exp(1j*d*(k1z_ + k1zi_))*k1z_*(k1zi_*sq1 + k0z_*(k1zi_**2 + kir_**2)*sq1)*ep1*sq2 - \
            2*exp(1j*d*(k1z_ + k1zi_))*k1z_*((-1 + a0HH_)*kiz(k0,thi) - (1 + a0HH_)*k0z_*(kiz(k0,thi)**2 + kir_**2))*ep1**1.5*sq2))/\
            (exp(1j*d*(k1zi_ + k2z_))*kir_*(kx_**2 + ky_**2)*sq1*(exp(2*1j*d*k1z_)*(k1z_ - k0z_*ep1)*(k2z_*ep1 - k1z_*ep2) + (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2)))) 

def c1VHF2_j(k0,thi,phi,th,ph,ep1,ep2,d):

    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )

    sq12 = sq1 * sq2
    
    bb0HH_ = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_ = b0HH(k0, thi, ep1, ep2, d)
    c0HH_ = c0HH(k0, thi, ep1, ep2, d)
    
    suma1 = k1z_ - k0z_ * ep1
    suma2 = k2zi_**2 + kir_**2
    suma3 = k1zi_**2 + kir_**2
    suma4 = k1z_ + k0z_*ep1

    exp1 = exp(2j*d*k1z_)
    
    return -1j*((kiy_*kx_ - kix_*ky_)*kr(k0,th)*(c0HH_*sq1*(exp(1j*d*(2*k1z_ + k1zi_ + k2zi_))*(suma1)*(k1z_*(suma2)*sq2 - k2zi_*ep1*sq2) - \
            exp(1j*d*(k1zi_ + k2zi_))*(suma4)*(k1z_*(suma2)*sq2 + k2zi_*ep1*sq2)) - \
            b0HH_*exp(2j*d*(k1z_ + k1zi_))*sq1*(suma1)*(k1z_*(suma3) - k1zi_*ep1)*sq2 + bb0HH_*sq1*(suma4)*(k1z_*(suma3) - k1zi_*ep1)*sq2 + \
            bb0HH_*exp1*sq1*(-suma4)*(k1z_*(suma3) + k1zi_*ep1)*sq2 + b0HH_*exp(2j*d*k1zi_)*sq1*(suma4)*(k1z_*(suma3) + k1zi_*ep1)*sq2))/\
            (exp(1j*d*(k1zi_ + k2z_))*kir_*(kx_**2 + ky_**2)*sq1*(exp1*(suma1)*(k2z_*ep1 - k1z_*ep2) + (suma4)*(k2z_*ep1 + k1z_*ep2)))

## CANAL VV ##
# AMPLITUD ONDA DISPERSADA #

def a1VVF1_j(k0,thi,phi,th,ph,ep1,ep2,d):

    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)

    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)

    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d) 
    
    exp1 = exp(1j*d*k1zi_)

    suma1 = kx_**2 - kiy_*ky_ + ky_**2
    suma2 = k2z_*ep1 + k1z_*ep2
    suma3 = kix_*kx_ + kiy_*ky_
    
    termino1 = (bb0VV_ + b0VV_)*(k1zi_**2*(suma3) + kir_**2*((kix_ - kx_)*kx_ + (kiy_ - ky_)*ky_))*sq1
    termino2 = (1 + a0VV_)*(kix_*(kiz_**2 + kir_**2)*kx_ + kiy_*kiz_**2*ky_ - kir_**2*(suma1))*ep1
    termino3 = bb0VV_*k1z_**2*(suma3)*sq1
    
    return 1j*(kr(k0,th)*(exp(1j*d*(2*k1z_ + k1zi_))*(-(k1z_*(suma3)*((-1 + a0VV_)*kiz_ + b0VV_*k1zi_*sq1)) + termino3 - termino1 + termino2)\
            *(-(k2z_*ep1) + k1z_*ep2) - exp1*(-(k1z_*(suma3)*((-1 + a0VV_)*kiz_ + b0VV_*k1zi_*sq1)) + termino3 + termino1 - termino2)\
            *(suma2)))/(exp1*kir_*(kx_**2 + ky_**2)*(exp(2j*d*k1z_)*(k1z_ - k0z_*ep1)*(-(k2z_*ep1) + k1z_*ep2) - (k1z_ + k0z_*ep1)*(suma2))) 

def a1VVF2_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)

    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)

    a0VV_ = a0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    suma1 = (kiy_ - ky_)*ky_
    suma2 = (kix_ - kx_)*kx_
    suma3 = kix_*kx_ + kiy_ *ky_
    suma4 = kir_**2*(suma2 + suma1)

    prod1 = k1zi_**2*(suma3)*ep2
    prod2 = k2z_*(suma3)*ep1
    
    return 1j*(kr(k0,th)*(-2*c0VV_*exp(1j*d*(k1z_ + k1zi_ + k2zi_))*k1z_*(-(k2z_*k2zi_*(suma3)) + k2zi_**2*(suma3) + suma4)*ep1*sq2 + \
            2*b0VV_*exp(1j*d*(k1z_ + 2*k1zi_))*k1z_*sq1*(-(k1zi_*prod2) + prod1 + suma4*ep2) + \
            2*bb0VV_*exp(1j*d*k1z_)*k1z_*sq1*(k1zi_*prod2 + prod1 + suma4*ep2)))/\
            (exp(1j*d*k1zi_)*kir_*(kx_**2 + ky_**2)*(exp(2j*d*k1z_)*(k1z_ - k0z_*ep1)*(-(k2z_*ep1) + k1z_*ep2) - (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2)))

def bb1VVF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)

    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)

    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d) 

    suma1 = (kiy_ - ky_) *ky_
    suma2 = (kix_ - kx_) *kx_
    suma4 = kir_**2 * (suma2 + suma1)
    suma5 = kix_*kx_ + kiy_*ky_
    suma6 = k1z_ + k1zi_
    suma7 = k2z_*ep1 - k1z_*ep2

    exp1 = exp(1j*d*(suma6))
        
    return -1j*((exp(1j*d*(k1z_ - k1zi_))*kr(k0,th)*(exp1*((-1 + a0VV_)*k0z_*kiz_*(suma5) - (1 + a0VV_)*(kix_*(kiz_**2 + kir_**2)*kx_ + kiy_*kiz_**2*ky_ - kir_**2*(kx_**2 - kiy_*ky_ + ky_**2)))*sq1*\
            (suma7) + bb0VV_*exp1*(k1zi_**2*(suma5) + suma4 - k0z_*k1z_*(suma5)*ep1)*(suma7) + \
            b0VV_*exp1*(k1zi_**2*(suma5) + suma4 + k0z_*k1zi_*(suma5)*ep1)*(suma7)))/\
            (kir_*(kx_**2 + ky_**2)*(exp(2j*d*k1z_)*(k1z_ - k0z_*ep1)*(suma7) + (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2)))) 

def bb1VVF2_j(k0,thi,phi,th,ph,ep1,ep2,d):

    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d) 

    suma1 = (kix_ * kx_ + kiy_ * ky_)
    suma2 = k1z_ + k0z_ * ep1

    termino1 = (k1zi_**2*suma1 + kir_**2*((kix_ - kx_)*kx_ + (kiy_ - ky_)*ky_))*ep2
    termino2 = k1zi_*k2z_*suma1*ep1
    
    return -1j*(exp(1j*d*(k1z_ - k1zi_))*kr(k0,th)*(c0VV_*exp(1j*d*(k1zi_ + k2zi_))*(k2z_*k2zi_*suma1 - k2zi_**2*suma1 + kir_**2*(-(kix_*kx_) +
            kx_**2 - kiy_*ky_ + ky_**2))*sqrt(ep1)*(suma2)*sqrt(ep2) - 
            b0VV_*exp(2j*d*k1zi_)*(suma2)*(termino2 - termino1) + 
                           bb0VV_*(suma2)*(termino2 + termino1)))/\
            (kir_*(kx_**2 + ky_**2)*(exp(2j*d*k1z_)*(k1z_ - k0z_*ep1)*(k2z_*ep1 - k1z_*ep2) + (suma2)*(k2z_*ep1 + k1z_*ep2)))

def b1VVF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)

    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)

    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d) 

    sq1 = np.sqrt(ep1)

    exp1 = exp(1j*d* k1zi_)
    suma1 = kix_ * kx_ + kiy_ * ky_
    producto1 = k0z_* suma1 *ep1
    
    termino1 = k1zi_**2*(suma1) + kir_**2*((kix_ - kx_)*kx_ + (kiy_ - ky_)*ky_)
    termino2 = exp1 * (k2z_*ep1 + k1z_*ep2)
    
    return  -1j*((kr(k0,th)*(-(exp1*((-1 + a0VV_)*k0z_*kiz_*(suma1) - (1 + a0VV_)*(kix_*(kiz_**2 + kir_**2)*kx_ + kiy_*kiz_**2*ky_ - kir_**2*(kx_**2 - kiy_*ky_ + ky_**2)))*sq1*(k2z_*ep1 + k1z_*ep2))-\
            bb0VV_*(termino1 - k1z_*producto1)*termino2 -\
             b0VV_*(termino1 + k1zi_*producto1)*termino2))/\
            (exp1*kir_*(kx_**2 + ky_**2)*(exp(2j*d*k1z_)*(k1z_ - k0z_*ep1)*(k2z_*ep1 - k1z_*ep2) + (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2)))) 

def b1VVF2_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)

    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    suma1 = kix_ * kx_ + kiy_ * ky_
    suma2 = k1z_ - k0z_ * ep1
    producto1 = k1zi_*k2z_ * suma1*ep1
    termino1 = k1zi_**2 *suma1*ep2 + kir_**2*((kix_ - kx_)*kx_ + (kiy_ - ky_)*ky_)*ep2

    return -1j*(kr(k0,th)*(-(c0VV_*exp(1j*d*(k1z_ + k1zi_ + k2zi_))*(-(k2z_*k2zi_*suma1) + k2zi_**2*suma1 + kir_**2*((kix_ - kx_)*kx_ + (kiy_ - ky_)*ky_))*sq1*suma2*sq2)+\
            b0VV_*exp(1j*d*(k1z_ + 2*k1zi_))*suma2*(-producto1 + termino1) +
            bb0VV_*exp(1j*d*k1z_)*suma2*(producto1 + termino1)))/\
            (exp(1j*d*k1zi_)*kir_*(kx_**2 + ky_**2)*(exp(2j*d*k1z_)*suma2*(k2z_*ep1 - k1z_*ep2) + (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2)))

def c1VVF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)

    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d) 

    exp1 = b0VV_*exp(1j*d*(k1z_+2*k1zi_))*k1z_*sq1
    suma1 = kix_*kx_ + kiy_*ky_

    termino1 = exp1*sq2*(k1zi_**2*suma1 + kir_**2*((kix_ - kx_)*kx_ + (kiy_ - ky_)*ky_) + k0z_*k1zi_*suma1*ep1)
    
    return -1j*(kr(k0,th)*(2*exp(1j*d*(k1z_ + k1zi_))*k1z_*((-1 + a0VV_)*k0z_*kiz_*suma1 - (1 + a0VV_)*(kix_*(kiz_**2 + kir_**2)*kx_ + kiy_*kiz_**2*ky_ - kir_**2*(kx_**2 - kiy_*ky_ + ky_**2)))*ep1*sq2 - \
             2*bb0VV_*exp(1j*d*(k1z_ + k1zi_))*k1z_*sq1*(-(k1zi_**2*suma1) + kir_**2*(-(kix_*kx_) + kx_**2 - kiy_*ky_ + ky_**2) + k0z_*k1z_*suma1*ep1)*sq2 + \
             2*termino1*cos(d*k1zi_) - 2j*termino1*sin(d*k1zi_)))/\
             (exp(1j*d*(k1zi_ + k2z_))*kir_*(kx_**2 + ky_**2)*(exp(2j*d*k1z_)*(k1z_ - k0z_*ep1)*(k2z_*ep1 - k1z_*ep2) + (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2)))

def c1VVF2_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    k2zi_ = k2zi(k0, thi, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)

    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    suma1 = kix_ * kx_ + kiy_ * ky_
    suma2 = kir_**2*((kix_ - kx_)*kx_ + (kiy_ - ky_)*ky_)
    suma3 = k2zi_**2*suma1 + suma2
    exp1 = exp(2j* d*k1z_)
    exp2 = exp(1j*d*(k1z_+2*k1zi_))
    prod1 = k1zi_**2 * suma1
    prod2 = b0VV_* exp2 *sq1

    return  1j*(kr(k0,th)*(bb0VV_*sq1*(-(exp1*(k1z_*k1zi_*suma1 + prod1 + suma2)*(k1z_ - k0z_*ep1)) + \
             (k1z_*k1zi_*suma1 - prod1 + kir_**2*(-(kix_*kx_) + kx_**2 - kiy_*ky_ + ky_**2))*(k1z_ + k0z_*ep1))*sq2 - \
             2*k1z_*prod2*(prod1 + suma2 + k0z_*k1zi_*suma1*ep1)*sq2*cos(d*k1z_) + \
             2j*prod2*(k1z_**2*k1zi_*suma1 + k0z_*(prod1 + suma2)*ep1)*sq2*sin(d*k1z_) + \
             2*c0VV_*exp(1j*d*(k1z_ + k1zi_ + k2zi_))*(k1z_*ep1*(suma3 + k0z_*k2zi_*suma1*ep2)*cos(d*k1z_) - \
             1j*(k0z_*suma3*ep1**2 + k1z_**2*k2zi_*suma1*ep2)*sin(d*k1z_))))/\
            (exp(1j*d*(k1zi_ + k2z_))*kir_*(kx_**2 + ky_**2)*(exp1*(k1z_ - k0z_*ep1)*(k2z_*ep1 - k1z_*ep2) + (k1z_ + k0z_*ep1)*(k2z_*ep1 + k1z_*ep2)))

## CANAL HV ##
# AMPLITUD ONDA DISPERSADA #
def a1HVF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)

    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)
    
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d)

    suma1 = k1zi_**2+kir_**2
    suma2 = bb0VV_ + b0VV_
    exp1 = exp(1j*d*(k1z_+k1zi_))
    
    return -1j*(((kiy_*kx_ - kix_*ky_)*kr(k0,th)*(2*exp1*k1z_*ep1*(-(suma2*k2z_*suma1) + (kiz_*(-1 + a0VV_ + (1 + a0VV_)*k2z_*kiz_)+(1 + a0VV_)*k2z_*kir_**2)*sq1 + \
            (-(bb0VV_*k1z_) + b0VV_*k1zi_)*ep1)*sq2*cos(d*k1z_) + 2j*exp1*ep1*\
            (suma2*k1z_**2*suma1 - ((-1 + a0VV_)*k2z_*kiz_ + (1 + a0VV_)*k1z_**2*(kiz_**2 + kir_**2))*sq1 + (bb0VV_*k1z_ - b0VV_*k1zi_)*k2z_*ep1)*sq2*sin(d*k1z_)))/\
            (exp(1j*d*k1zi_)*(exp(2j*d*k1z_)*(k0z(k0,th) - k1z_)*(k1z_ - k2z_) + (k0z(k0,th) + k1z_)*(k1z_ + k2z_))*kir_*(kx_**2 + ky_**2)*ep1**1.5*sq2))

def bb1HVF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)

    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d)

    suma1 =  k1z_ - k2z_
    
    return -1j*((exp(1j*d*(k1z_ - k1zi_) + 1j*d*(k1z_ + k1zi_))*suma1*(kiy_*kx_ - kix_*ky_)*kr(k0,th)*\
             ((bb0VV_ + b0VV_)*k0z_*(k1zi_**2 + kir_**2) + (-1 + a0VV_)*kiz_*sq1 - (1 + a0VV_)*k0z_*(kiz_**2 + kir_**2)*sq1 + (-(bb0VV_*k1z_) + b0VV_*k1zi_)*ep1))/\
             ((exp(2j*d*k1z_)*(k0z_ - k1z_)*suma1 + (k0z_ + k1z_)*(k1z_ + k2z_))*kir_*(kx_**2 + ky_**2)*sq1)) 

def b1HVF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)

    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d)
    
    suma1 = k1z_+k2z_
    suma2 = k0z_*(k1zi_**2 + kir_**2)
    exp1 = exp(1j*d*k1zi_)

    prod1 = exp1*suma1*sq1
    
    return -1j*(((kiy_*kx_ - kix_*ky_)*kr(k0,th)*(-(exp1*suma1*(kiz_*(1 - a0VV_ + (1 + a0VV_)*k0z_*kiz_) + (1 + a0VV_)*k0z_*kir_**2)*ep1*ep2) + \
            bb0VV_*prod1*(suma2 - k1z_*ep1)*ep2 + b0VV_*prod1*(suma2 + k1zi_*ep1)*ep2))/\
            (exp1*(exp(2j*d*k1z_)*(k0z_ - k1z_)*(k1z_ - k2z_) + (k0z_ + k1z_)*suma1)*kir_*(kx_**2 + ky_**2)*ep1*ep2)) 

def c1HVF1_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)

    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)
    
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d)
    
    exp1 = exp(1j*d*(k1z_ + k1zi_))
    suma1 = k1zi_**2 + kir_**2

    
    return 1j*((kiy_*kx_ - kix_*ky_)*kr(k0,th)*(2*bb0VV_*exp1*k1z_*sq1*(-(k0z_*(suma1)) + k1z_*ep1)*sq2 - \
            2*exp1*k1z_*sq1*(b0VV_*k0z_*(suma1) + (-1 + a0VV_)*kiz_*sq1 - (1 + a0VV_)*k0z_*(kiz_**2 + kir_**2)*sq1 + b0VV_*k1zi_*ep1)*sq2))/\
            (exp(1j*d*(k1zi_ + k2z_))*(exp(2j*d*k1z_)*(k0z_ - k1z_)*(k1z_ - k2z_) + (k0z_ + k1z_)*(k1z_ + k2z_))*kir_*(kx_**2 + ky_**2)*ep1*sq2) 

def a1HVF2_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)

    sq2 = sqrt(ep2)
    
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    suma1 = k2z_*(k1zi_**2 +kir_**2)
    suma2 = k1z_ + k1zi_
    prod1 = k1zi_*ep1
    
    sumBB = bb0VV_ + b0VV_
    resBB = bb0VV_ - b0VV_
    dk = d*k1zi_

    termino1 = sq2*exp(1j*d*suma2)*k1z_*ep1
    
    return -1j*((kiy_*kx_ - kix_*ky_)*kr(k0,th)*(-2*c0VV_*exp(1j*d*(suma2 + k2zi_))*k1z_*ep1**1.5*(k2z_*(k2zi_**2 + kir_**2) - k2zi_*ep2) + \
            2*termino1*(sumBB*suma1 + resBB*prod1)*cos(dk) - 2j*termino1*(resBB*suma1 + sumBB*prod1)*sin(dk)))/\
            (exp(1j*dk)*(exp(2j*d*k1z_)*(k0z_ - k1z_)*(k1z_ - k2z_) + (k0z_ + k1z_)*(k1z_ + k2z_))*kir_*(kx_**2 + ky_**2)*ep1**1.5*sq2)

def bb1HVF2_j(k0,thi,phi,th,ph,ep1,ep2,d):
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)
    
    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)
    
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    suma1 = k1zi_**2 +kir_**2
    suma2 = k0z_ + k1z_
    
    return -1j*(exp(1j*d*(k1z_ - k1zi_))*(kiy_*kx_ - kix_*ky_)*kr(k0,th)*(suma2*sq1*(bb0VV_*k2z_*suma1 + bb0VV_*k1zi_*ep1 + b0VV_*exp(2j*d*k1zi_)*(k2z_*suma1 - k1zi_*ep1))*ep2 + \
            c0VV_*exp(1j*d*(k1zi_ + k2zi_))*suma2*ep1*sq2*(-(k2z_*(k2zi_**2 + kir_**2)) + k2zi_*ep2)))/\
            ((exp(2j*d*k1z_)*(k0z_ - k1z_)*(k1z_ - k2z_) + suma2*(k1z_ + k2z_))*kir_*(kx_**2 + ky_**2)*ep1*ep2)

def b1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)

    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)
    
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    suma1 = k2z_ * (k1zi_**2 + kir_**2)
    suma2 = -k0z_ + k1z_
    
    exp1 = exp(1j * d * k1zi_)
    exp2 = exp(1j * d * (k1z_ + k1zi_))
    exp3 = exp(1j * d * k1z_)
    exp4 = exp(1j * d * (k1z_ + k1zi_ + k2zi_))
    exp5 = exp(2j * d * k1z_)
    
    numerator = -1j * (kiy_ * kx_ - kix_ * ky_) * kr(k0, th) * (
        b0VV_ * exp2 * suma2 * sq1 * (suma1 - k1zi_ * ep1) * ep2 +
        bb0VV_ * exp3 * suma2 * sq1 * (suma1 + k1zi_ * ep1) * ep2 +
        c0VV_ * exp4 * suma2 * ep1 * sq2 * (-k2z_ * (k2zi_**2 + kir_**2) + k2zi_ * ep2)
    )
    
    denominator = exp1 * (
        exp5 * (k0z_ - k1z_) * (k1z_ - k2z_) +
        (k0z_ + k1z_) * (k1z_ + k2z_)
    ) * kir_ * (kx_**2 + ky_**2) * ep1 * ep2
    
    return numerator / denominator

def c1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d):
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kir_ = kir(k0, thi)
    k0z_ = k0z(k0, th)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    kx_ = kx(k0, th, ph)
    ky_ = ky(k0, th, ph)

    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)
    
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    suma1 = k1zi_**2 + kir_**2
    suma2 = -k0z_ + k1z_
    suma3 = k0z_ + k1z_
    suma4 = k1z_*(k2zi_**2 + kir_**2)
    exp1 = exp(1j * d * (k1zi_ + k2z_))
    exp2 = exp(2j * d * k1z_)
    exp3 = exp(2j * d * (k1z_ + k1zi_))
    exp4 = exp(2j * d * k1zi_)
    exp5 = exp(1j * d * (k1zi_ + k2zi_))

    numerator = 1j * (kiy_ * kx_ - kix_ * ky_) * kr(k0, th) * (
        -b0VV_ * exp3 * suma2 * sq1 * (k1z_ * suma1 - k1zi_ * ep1) * sq2 +
        b0VV_ * exp4 * suma3 * sq1 * (k1z_ * suma1 + k1zi_ * ep1) * sq2 +
        bb0VV_ * sq1 * (
            suma3 * (k1z_ * suma1 - k1zi_ * ep1) +
            exp2 * suma2 * (k1z_ * suma1 + k1zi_ * ep1)
        ) * sq2 +
        c0VV_ * exp5 * ep1 * (
            exp2 * suma2 * (suma4 - k2zi_ * ep2) -
            suma3 * (suma4 + k2zi_ * ep2)
        )
    )

    denominator = exp1 * (
        exp2 * suma2 * (k1z_ - k2z_) +
        suma3 * (k1z_ + k2z_)
    ) * kir_ * (kx_**2 + ky_**2) * ep1 * sq2

    return numerator / denominator

##### Coeficientes de orden dos ######

# **Están definidos en función de [sx,sy] que son variables auxiliares que deben ser integradas en (-Inf ; Inf)**

def sr(sx,sy):
    return sqrt(sx**2+sy**2)

def s0z(k0,sx,sy):
    return sqrt(k0**2-sx**2-sy**2+0j)

def s1z(k0,sx,sy,ep1):
    return sqrt(ep1*k0**2-sx**2-sy**2+0j)

# # para recuperar una sola capa
# def s1z(k0,sx,sy,ep1):
#     return 0*sqrt(ep1*k0**2-sx**2-sy**2+0j)

def s2z(k0,sx,sy,ep2):
    return sqrt(ep2*k0**2-sx**2-sy**2+0j)


def L01TE_j(k0, thi, phi, ep1, ep2, d):
    b0HH_val = b0HH(k0, thi, ep1, ep2, d)
    bb0HH_val = bb0HH(k0, thi, ep1, ep2, d)
    a0HH_val = a0HH(k0, thi, ep1, ep2, d)
    
    k1zi_val = k1zi(k0, thi, ep1)
    kix_val = kix(k0, thi, phi)
    kiz_val = kiz(k0, thi)
    kir_val = kir(k0, thi)
    
    numerator = -((b0HH_val + bb0HH_val) * k1zi_val**2 * kix_val) + (1 + a0HH_val) * kix_val * kiz_val**2
    denominator = 2 * kir_val
    
    return numerator / denominator

def L02TE_j(k0, thi, phi, ep1, ep2, d):
    b0HH_val = b0HH(k0, thi, ep1, ep2, d)
    bb0HH_val = bb0HH(k0, thi, ep1, ep2, d)
    a0HH_val = a0HH(k0, thi, ep1, ep2, d)
    
    k1zi_val = k1zi(k0, thi, ep1)
    kiy_val = kiy(k0, thi, phi)
    kiz_val = kiz(k0, thi)
    kir_val = kir(k0, thi)
    
    term1 = -(b0HH_val * k1zi_val**2 * kiy_val) / (2 * kir_val)
    term2 = -(bb0HH_val * k1zi_val**2 * kiy_val) / (2 * kir_val)
    term3 = (kiy_val * kiz_val**2) / (2 * kir_val)
    term4 = (a0HH_val * kiy_val * kiz_val**2) / (2 * kir_val)
    
    return term1 + term2 + term3 + term4

def L03TE_j(k0, thi, phi, ep1, ep2, d):
    b0HH_val = b0HH(k0, thi, ep1, ep2, d)
    bb0HH_val = bb0HH(k0, thi, ep1, ep2, d)
    a0HH_val = a0HH(k0, thi, ep1, ep2, d)
    
    k1zi_val = k1zi(k0, thi, ep1)
    k2zi_val = k2zi(k0, thi, ep2)
    kiy_val = kiy(k0, thi, phi)
    kiz_val = kiz(k0, thi)
    kir_val = kir(k0, thi)
    
    term1 = (kiy_val * kiz_val**3) / (2 * k0 * kir_val)
    term2 = -(a0HH_val * kiy_val * kiz_val**3) / (2 * k0 * kir_val)
    term3 = -(b0HH_val * k1zi_val**3 * kiy_val * sqrt(ep1)) / (2 * k0 * kir_val)
    term4 = (bb0HH_val * k1zi_val**3 * kiy_val * sqrt(ep1)) / (2 * k0 * kir_val)
    
    return term1 + term2 + term3 + term4

def L04TE_j(k0, thi, phi, ep1, ep2, d):
    b0HH_val = b0HH(k0, thi, ep1, ep2, d)
    bb0HH_val = bb0HH(k0, thi, ep1, ep2, d)
    a0HH_val = a0HH(k0, thi, ep1, ep2, d)
    
    k1zi_val = k1zi(k0, thi, ep1)
    k2zi_val = k2zi(k0, thi, ep2)
    kix_val = kix(k0, thi, phi)
    kiz_val = kiz(k0, thi)
    kir_val = kir(k0, thi)
    
    term1 = -(kix_val * kiz_val**3) / (2 * k0 * kir_val)
    term2 = (a0HH_val * kix_val * kiz_val**3) / (2 * k0 * kir_val)
    term3 = (b0HH_val * k1zi_val**3 * kix_val * sqrt(ep1)) / (2 * k0 * kir_val)
    term4 = -(bb0HH_val * k1zi_val**3 * kix_val * sqrt(ep1)) / (2 * k0 * kir_val)
    
    return term1 + term2 + term3 + term4

def L05TE_j(k0, thi, phi, ep1, ep2, d):
    bb0HH_val = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_val = b0HH(k0, thi, ep1, ep2, d)
    c0HH_val = c0HH(k0, thi, ep1, ep2, d)
    
    k1zi_val = k1zi(k0, thi, ep1)
    k2zi_val = k2zi(k0, thi, ep2)
    kix_val = kix(k0, thi, phi)
    kir_val = kir(k0, thi)
    
    term1 = (bb0HH_val * k1zi_val**2 * kix_val) / (2 * exp(1j * d * k1zi_val) * kir_val)
    term2 = (b0HH_val * exp(1j * d * k1zi_val) * k1zi_val**2 * kix_val) / (2 * kir_val)
    term3 = -(c0HH_val * exp(1j * d * k2zi_val) * k2zi_val**2 * kix_val) / (2 * kir_val)
    
    return term1 + term2 + term3

def L06TE_j(k0, thi, phi, ep1, ep2, d):
    bb0HH_val = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_val = b0HH(k0, thi, ep1, ep2, d)
    c0HH_val = c0HH(k0, thi, ep1, ep2, d)
    
    k1zi_val = k1zi(k0, thi, ep1)
    k2zi_val = k2zi(k0, thi, ep2)
    kiy_val = kiy(k0, thi, phi)
    kir_val = kir(k0, thi)
    
    term1 = (bb0HH_val * k1zi_val**2 * kiy_val) / (2 * exp(1j * d * k1zi_val) * kir_val)
    term2 = (b0HH_val * exp(1j * d * k1zi_val) * k1zi_val**2 * kiy_val) / (2 * kir_val)
    term3 = -(c0HH_val * exp(1j * d * k2zi_val) * k2zi_val**2 * kiy_val) / (2 * kir_val)
    
    return term1 + term2 + term3

def L07TE_j(k0, thi, phi, ep1, ep2, d):
    bb0HH_val = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_val = b0HH(k0, thi, ep1, ep2, d)
    c0HH_val = c0HH(k0, thi, ep1, ep2, d)
    
    k1zi_val = k1zi(k0, thi, ep1)
    k2zi_val = k2zi(k0, thi, ep2)
    kiy_val = kiy(k0, thi, phi)
    k0_val = k0
    kir_val = kir(k0, thi)
    
    term1 = -((bb0HH_val * k1zi_val**3 * kiy_val) / (exp(1j * d * k1zi_val) * k0_val * kir_val))
    term2 = (b0HH_val * exp(1j * d * k1zi_val) * k1zi_val**3 * kiy_val) / (k0_val * kir_val)
    term3 = -(c0HH_val * exp(1j * d * k2zi_val) * k2zi_val**3 * kiy_val) / (k0_val * kir_val)
    
    return term1 + term2 + term3

def L08TE_j(k0, thi, phi, ep1, ep2, d):
    bb0HH_val = bb0HH(k0, thi, ep1, ep2, d)
    b0HH_val = b0HH(k0, thi, ep1, ep2, d)
    c0HH_val = c0HH(k0, thi, ep1, ep2, d)
    
    k1zi_val = k1zi(k0, thi, ep1)
    k2zi_val = k2zi(k0, thi, ep2)
    kix_val = kix(k0, thi, phi)
    k0_val = k0
    kir_val = kir(k0, thi)
    
    term1 = (bb0HH_val * k1zi_val**3 * kix_val) / (exp(1j * d * k1zi_val) * k0_val * kir_val)
    term2 = -(b0HH_val * exp(1j * d * k1zi_val) * k1zi_val**3 * kix_val) / (k0_val * kir_val)
    term3 = (c0HH_val * exp(1j * d * k2zi_val) * k2zi_val**3 * kix_val) / (k0_val * kir_val)
    
    return term1 + term2 + term3

def Y1TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    a1HHF1_val = a1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VHF1_val = a1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF1_val = b1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF1_val = b1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF1_val = bb1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF1_val = bb1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    
    s0z_val = s0z(k0, sx, sy)
    s1z_val = s1z(k0, sx, sy, ep1)
    ky_val = ky(k0, th, ph)
    sr_val = sr(sx, sy)
    
    term1 = -1j * s0z_val * ((a1HHF1_val * sx) / sr_val + (a1VHF1_val * s0z_val * sy) / (k0 * sr_val))
    term2 = (1j * a1VHF1_val * (ky_val - sy) * sr_val) / k0
    term3 = -1j * s1z_val * ((b1HHF1_val * sx) / sr_val - (b1VHF1_val * s1z_val * sy) / (k0 * sr_val * sqrt(ep1)))
    term4 = 1j * s1z_val * ((bb1HHF1_val * sx) / sr_val + (bb1VHF1_val * s1z_val * sy) / (k0 * sr_val * sqrt(ep1)))
    term5 = -(1j * b1VHF1_val * (ky_val - sy) * sr_val) / (k0 * sqrt(ep1))
    term6 = -(1j * bb1VHF1_val * (ky_val - sy) * sr_val) / (k0 * sqrt(ep1))
    
    return term1 + term2 + term3 + term4 + term5 + term6

def q1TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    L11TE_val = 0
    Y1TE_F1_val = Y1TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    return L11TE_val + Y1TE_F1_val

# -- F2
def Y1TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    a1HHF2_val = a1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VHF2_val = a1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF2_val = b1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF2_val = b1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF2_val = bb1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF2_val = bb1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    
    s0z_val = s0z(k0, sx, sy)
    s1z_val = s1z(k0, sx, sy, ep1)
    ky_val = ky(k0, th, ph)
    sr_val = sr(sx, sy)
    
    term1 = -1j * s0z_val * ((a1HHF2_val * sx) / sr_val + (a1VHF2_val * s0z_val * sy) / (k0 * sr_val))
    term2 = (1j * a1VHF2_val * (ky_val - sy) * sr_val) / k0
    term3 = -1j * s1z_val * ((b1HHF2_val * sx) / sr_val - (b1VHF2_val * s1z_val * sy) / (k0 * sr_val * sqrt(ep1)))
    term4 = 1j * s1z_val * ((bb1HHF2_val * sx) / sr_val + (bb1VHF2_val * s1z_val * sy) / (k0 * sr_val * sqrt(ep1)))
    term5 = -(1j * b1VHF2_val * (ky_val - sy) * sr_val) / (k0 * sqrt(ep1))
    term6 = -(1j * bb1VHF2_val * (ky_val - sy) * sr_val) / (k0 * sqrt(ep1))
    
    return term1 + term2 + term3 + term4 + term5 + term6

def q1TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    L11TE_val = 0
    Y1TE_F2_val = Y1TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    return L11TE_val + Y1TE_F2_val

def Y2TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    a1HHF1_val = a1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VHF1_val = a1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF1_val = b1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF1_val = b1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF1_val = bb1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF1_val = bb1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    
    s0z_val = s0z(k0, sx, sy)
    s1z_val = s1z(k0, sx, sy, ep1)
    kx_val = kx(k0, th, ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1)
    k0sr = k0*sr_val
    prod1 = s1z_val * sx/(k0sr*sq1)
    suma1 = (kx_val - sx)*sr_val
    factor = suma1/(k0*sq1)
    
    term1 = -s0z_val * (-(a1VHF1_val*s0z_val*sx)/k0sr + (a1HHF1_val*sy)/sr_val)
    term2 = -(a1VHF1_val*suma1)/k0
    term3 = -s1z_val*((b1HHF1_val*sy) / sr_val + b1VHF1_val*prod1)
    term4 = s1z_val*((bb1HHF1_val*sy) / sr_val - bb1VHF1_val*prod1)

    return 1j*(term1 + term2 + term3 + term4 + (b1VHF1_val + bb1VHF1_val)*factor)

def Y2TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    a1HHF2_val = a1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VHF2_val = a1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF2_val = b1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF2_val = b1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF2_val = bb1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF2_val = bb1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    
    s0z_val = s0z(k0, sx, sy)
    s1z_val = s1z(k0, sx, sy, ep1)
    kx_val = kx(k0, th, ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1)
    k0sr = k0*sr_val
    
    suma1 = (kx_val - sx)*sr_val
    prod1 = s1z_val*sx/(k0sr*sq1)
    prod2 = sy/sr_val
    factor = suma1/(k0*sq1)
    
    term1 = -s0z_val*(-a1VHF2_val*s0z_val*sx/k0sr + a1HHF2_val*prod2)
    term2 = -a1VHF2_val*suma1/k0
    term3 = b1HHF2_val*prod2 + b1VHF2_val*prod1
    term4 = bb1HHF2_val*prod2 - bb1VHF2_val*prod1
    
    return 1j*(term1 + term2 + s1z_val*(-term3 + term4) + (b1VHF2_val + bb1VHF2_val)*factor)

def q2TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d): 
    L12TE_val = 0
    Y2TE_F1_val = Y2TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    return L12TE_val + Y2TE_F1_val

def q2TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    L12TE_val = 0
    Y2TE_F2_val = Y2TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    return L12TE_val + Y2TE_F2_val

def L13TE_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    suma1 = (ky(k0, th, ph)-sy)/k0
    prod1 = kir(k0, thi)*suma1
    
    return ((b0HH(k0, thi, ep1, ep2, d)-bb0HH(k0, thi, ep1, ep2, d))*k1zi(k0, thi, ep1) + (a0HH(k0, thi, ep1, ep2, d)-1)*kiz(k0, thi))*prod1

def Y3TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    a1VHF1_val = a1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1HHF1_val = a1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF1_val = b1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF1_val = b1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF1_val = bb1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF1_val = bb1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    
    s0z_val = s0z(k0, sx, sy)
    s1z_val = s1z(k0, sx, sy, ep1)
    ky_val = ky(k0, th, ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1)
    k0sr = k0*sr_val
    prod1 = sx/sr_val
    prod2 = s1z_val*sy/k0sr
    suma1 = ky_val-sy
    factor = suma1*sr_val/k0
    
    term1 = -s0z_val*(a1VHF1_val*prod1 - a1HHF1_val*prod2)
    term3 = (b1HHF1_val + bb1HHF1_val)
    term4 =  b1VHF1_val*prod1*sq1 +  b1HHF1_val*prod2
    term5 = bb1VHF1_val*prod1*sq1 - bb1HHF1_val*prod2
    
    return 1j*(term1 - factor*(a1HHF1_val - term3) - s1z_val*(term4 - term5))

def Y3TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    s0z_val = s0z(k0, sx, sy)
    a1VHF2_val = a1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1HHF2_val = a1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF2_val = b1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF2_val = b1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF2_val = bb1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF2_val = bb1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    s1z_val = s1z(k0, sx, sy, ep1)
    ky_val = ky(k0, th, ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1)
    k0sr = k0*sr_val
    prod1 = sx/sr_val
    prod2 = s1z_val*sy/k0sr
    suma1 = ky_val-sy
    factor = suma1*sr_val/k0
    
    
    term1 = -s0z_val*(a1VHF2_val*prod1 - a1HHF2_val*prod2)
    term3 = b1HHF2_val + bb1HHF2_val
    term4 = b1VHF2_val*prod1*sq1 + b1HHF2_val*prod2
    term5 = bb1VHF2_val*prod1*sq1 - bb1HHF2_val*prod2
    
    return 1j*(term1 - factor*(a1HHF2_val - term3) - s1z_val*(term4 - term5))

def q3TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    L13TE_val = L13TE_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    Y3TE_F1_val = Y3TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    return L13TE_val + Y3TE_F1_val


def q3TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    L13TE_val = L13TE_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    Y3TE_F2_val = Y3TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    return L13TE_val + Y3TE_F2_val

def L14TE_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    b0HH_val = b0HH(k0, thi,ep1,ep2,d)
    bb0HH_val = bb0HH(k0, thi,ep1,ep2,d)
    a0HH_val = a0HH(k0, thi,ep1,ep2,d)
    
    kiz_val = kiz(k0,thi)
    k1zi_val = k1zi(k0, thi, ep1)
    kir_val = kir(k0, thi)
    kx_val = kx(k0, th, ph)

    prod1 = kir_val*(kx_val-sx)/k0
    
    return prod1*( (bb0HH_val-b0HH_val)*k1zi_val + (1 -a0HH_val)*kiz_val )

def Y4TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    a1VHF1_val = a1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1HHF1_val = a1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF1_val = b1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF1_val = b1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF1_val = bb1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF1_val = bb1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_val = s0z(k0, sx, sy)
    s1z_val = s1z(k0, sx, sy, ep1)
    kx_val = kx(k0, th,ph)
    ky_val = ky(k0, th,ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1)
    prod1 = s1z_val*sx/(k0*sr_val)
    prod2 = sy/sr_val
    factor = (kx_val-sx)*sr_val/k0
    
    term1 = -s0z_val*(a1HHF1_val*prod1 + a1VHF1_val*prod2)
    term3 = -(b1HHF1_val + bb1HHF1_val)
    term4 = b1VHF1_val*sq1*prod2 - b1HHF1_val*prod1
    term5 = bb1VHF1_val*sq1*prod2 + bb1HHF1_val*prod1
    
    return 1j*(term1 + (a1HHF1_val + term3)*factor - s1z_val*(term4 - term5))

def Y4TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    a1VHF2_val = a1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1HHF2_val = a1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF2_val = b1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF2_val = b1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF2_val = bb1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF2_val = bb1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    s0z_val = s0z(k0, sx, sy)
    s1z_val = s1z(k0, sx, sy, ep1)
    ky_val = ky(k0, th, ph)
    kx_val = kx(k0, th, ph)
    sr_val = sr(sx, sy)
    
    sq1 = sqrt(ep1 )
    suma1 = (kx_val-sx)*sr_val/k0
    prod1 = s1z_val*sx /(k0*sr_val)
    prod2 = sy/sr_val
    
    term1 = -s0z_val*((a1HHF2_val*prod1) + (a1VHF2_val*prod2))
    term4 = b1VHF2_val*prod2*sq1 - b1HHF2_val*prod1
    term5 = bb1VHF2_val*prod2*sq1 + bb1HHF2_val*prod1
    
    return 1j*(term1 + suma1*(a1HHF2_val - (b1HHF2_val+bb1HHF2_val)) + s1z_val*(-term4 + term5))

def q4TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    return L14TE_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d) + Y4TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)

def q4TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d):
    return L14TE_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d) + Y4TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)

def Y5TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1VHF1_val = c1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1HHF1_val = c1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF1_val = b1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF1_val = b1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF1_val = bb1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF1_val = bb1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s1z_val = s1z(k0, sx, sy, ep1)
    s2z_val = s2z(k0, sx, sy, ep2)
    kx_val = kx(k0, th,ph)
    ky_val = ky(k0, th,ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)
    exp1 = exp(1j*d*s1z_val)
    exp2 = exp(1j*d*s2z_val)
    
    k0sr = k0*sr_val
    prod1 = s1z_val*sy/(k0sr*sq1)
    prod2 = sx/sr_val
    suma1 = (ky_val - sy)*sr_val
    
    term1 = exp1*(b1HHF1_val*prod2 - (b1VHF1_val*prod1))
    term2 =     (bb1HHF1_val*prod2 + (bb1VHF1_val*prod1))/exp1
    term3 = ((bb1VHF1_val/exp1 + b1VHF1_val*exp1)*suma1)/(k0*sq1)
    term4 = - exp2*s2z_val*((c1HHF1_val*sx)/sr_val - (c1VHF1_val*s2z_val*sy)/(k0sr*sq2))
    term5 = - (c1VHF1_val*exp2*suma1)/(k0*sq2)
    
    return 1j*(s1z_val*(term1 - term2) + term3 + term4 + term5)

def Y5TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1VHF2_val = c1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1HHF2_val = c1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF2_val = b1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF2_val = b1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF2_val = bb1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF2_val = bb1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    
    s1z_val = s1z(k0, sx, sy, ep1)
    s2z_val = s2z(k0, sx, sy, ep2)
    kx_val = kx(k0, th,ph)
    ky_val = ky(k0, th,ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1)
    sq2 = sqrt(ep2)
    exp1 = exp(1j*d *s1z_val)
    exp2 = exp(1j*d *s2z_val)
    
    k0sr = k0* sr_val
    prod1 = s1z_val* sy/(k0sr*sq1)
    prod2 = sx /sr_val
    suma1 = (ky_val - sy) *sr_val
    
    term1 = exp1*(b1HHF2_val*prod2 - b1VHF2_val*prod1)
    term2 =     (bb1HHF2_val*prod2 + bb1VHF2_val*prod1)/exp1
    term3 = ((bb1VHF2_val/exp1 + b1VHF2_val*exp1)*suma1)/(k0*sq1)
    term4 = - exp2*s2z_val*(c1HHF2_val*prod2 - (c1VHF2_val*s2z_val*sy)/(k0sr*sq2))
    term5 = - (c1VHF2_val*exp2*suma1)/(k0*sq2)
    
    return 1j*(s1z_val*(term1 - term2) + term3 + term4 + term5)

def q5TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L15TE = 0
    return L15TE + Y5TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q5TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L15TE = 0
    return L15TE + Y5TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def Y6TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1VHF1_val = c1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1HHF1_val = c1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF1_val = b1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF1_val = b1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF1_val = bb1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF1_val = bb1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    
    s1z_val = s1z(k0, sx, sy, ep1)
    s2z_val = s2z(k0, sx, sy, ep2)
    kx_val = kx(k0, th,ph)
    ky_val = ky(k0, th,ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j*d*s1z_val)
    exp2 = exp(1j*d*s2z_val)
    prod1 = s1z_val*sx/(k0*sr_val*sq1)
    prod2 = sy/sr_val
    suma1 = (kx_val - sx)*sr_val
    
    term1 = exp1*(b1HHF1_val*prod2 + b1VHF1_val*prod1)
    term2 = (bb1HHF1_val*prod2 - bb1VHF1_val*prod1)/exp1
    term3 = - (bb1VHF1_val/exp1 + b1VHF1_val*exp1)*suma1/(k0*sq1)
    term4 = - s2z_val*(c1HHF1_val*prod2 + c1VHF1_val*s2z_val*sx/(k0*sr_val*sq2))
    term5 = c1VHF1_val*suma1/(k0*sq2)
    
    return 1j*(s1z_val*(term1 - term2) + term3 + exp2*(term4 + term5))

def Y6TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1VHF2_val = c1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1HHF2_val = c1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF2_val = b1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF2_val = b1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF2_val = bb1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF2_val = bb1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    
    s1z_val = s1z(k0, sx, sy, ep1)
    s2z_val = s2z(k0, sx, sy, ep2)
    kx_val = kx(k0, th,ph)
    ky_val = ky(k0, th,ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j*d*s1z_val)
    exp2 = exp(1j*d*s2z_val)
    
    term1 = exp1*s1z_val*((b1HHF2_val*sy)/sr_val + (b1VHF2_val*s1z_val*sx)/(k0*sr_val*sq1))
    term2 = - (s1z_val*((bb1HHF2_val*sy)/sr_val - (bb1VHF2_val*s1z_val*sx)/(k0*sr_val*sq1)))/exp1
    term3 = - ((bb1VHF2_val/exp1 + b1VHF2_val*exp1)*(kx_val - sx)*sr_val)/(k0*sq1)
    term4 = - exp2*s2z_val*((c1HHF2_val*sy)/sr_val + (c1VHF2_val*s2z_val*sx)/(k0*sr_val*sq2))
    term5 = (c1VHF2_val*exp2*(kx_val - sx)*sr_val)/(k0*sq2)
    
    return 1j*(term1 + term2 + term3 + term4 + term5)

def q6TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L16TE = 0
    return L16TE + Y6TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q6TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L16TE = 0
    return L16TE + Y6TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def L17TE_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    b0HH_val = b0HH(k0, thi,ep1,ep2,d)
    bb0HH_val = bb0HH(k0, thi,ep1,ep2,d)
    c0HH_val = c0HH(k0, thi,ep1,ep2,d)

    k1zi_val = k1zi(k0, thi, ep1)
    k2zi_val = k2zi(k0, thi,ep2)
    kir_val = kir(k0, thi)
    ky_val = ky(k0, th, ph)

    prod1 = kir_val*(ky_val - sy)/k0
    exp1 = exp(1j*d*k1zi_val)

    t1 = bb0HH_val/exp1
    t2 = -b0HH_val*exp1
    t3 = c0HH_val*exp(1j*d*k2zi_val)*k2zi_val

    return  ((t1 + t2)*k1zi_val + t3)*prod1

def Y7TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1VHF1_val = c1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1HHF1_val = c1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF1_val = b1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF1_val = b1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF1_val = bb1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF1_val = bb1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s1z_val = s1z(k0, sx, sy, ep1)
    s2z_val = s2z(k0, sx, sy, ep2)
    kx_val = kx(k0, th,ph)
    ky_val = ky(k0, th,ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j*d *s1z_val)
    exp2 = exp(1j*d *s2z_val)

    k0sr = k0*sr_val
    suma1 = (ky_val - sy)*sr_val/k0
    prod1 = sx/sr_val
    prod2 = s1z_val*sy/k0sr

    t1 = c1HHF1_val*exp2
    t2 = bb1HHF1_val/exp1 + b1HHF1_val*exp1
    t3 = exp1*(b1VHF1_val*sq1*prod1 + b1HHF1_val*prod2)
    t4 = (bb1VHF1_val*sq1*prod1 - bb1HHF1_val*prod2)/exp1
    
    t5 = - exp2*s2z_val*(c1VHF1_val*sq2*prod1 + c1HHF1_val*s2z_val*sy/k0sr)

    
    return 1j*((t1 - t2)*suma1 + (t3 - t4)*s1z_val + t5)

def Y7TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1VHF2_val = c1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1HHF2_val = c1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF2_val = b1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF2_val = b1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF2_val = bb1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF2_val = bb1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s1z_val = s1z(k0, sx, sy, ep1)
    s2z_val = s2z(k0, sx, sy, ep2)
    kx_val = kx(k0, th,ph)
    ky_val = ky(k0, th,ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j*d *s1z_val)
    exp2 = exp(1j*d *s2z_val)

    k0sr = k0*sr_val
    suma1 = (ky_val - sy)*sr_val/k0
    prod1 = sx/sr_val
    prod2 = s1z_val*sy/k0sr

    t1 = c1HHF2_val*exp2
    t2 = bb1HHF2_val/exp1 + b1HHF2_val*exp1
    t3 = exp1*(b1VHF2_val*sq1*prod1 + b1HHF2_val*prod2)
    t4 = (bb1VHF2_val*sq1*prod1 - bb1HHF2_val*prod2)/exp1
    
    t5 = - exp2*s2z_val*(c1VHF2_val*sq2*prod1 + c1HHF2_val*s2z_val*sy/k0sr)

    
    return 1j*((t1 - t2)*suma1 + (t3 - t4)*s1z_val + t5)

def q7TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L17TE_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y7TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q7TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L17TE_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y7TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def L18TE_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    b0HH_val = b0HH(k0, thi,ep1,ep2,d)
    bb0HH_val = bb0HH(k0, thi,ep1,ep2,d)
    c0HH_val = c0HH(k0, thi,ep1,ep2,d)
    
    k1zi_val = k1zi(k0, thi, ep1)
    k2zi_val = k2zi(k0, thi,ep2)
    kir_val = kir(k0, thi)
    kx_val = kx(k0, th, ph)

    prod1 = kir_val*(kx_val - sx)/k0
    
    t1 = -bb0HH_val/exp(1j*d*k1zi_val)
    t2 = b0HH_val*exp(1j*d*k1zi_val)
    t3 = - c0HH_val*exp(1j*d*k2zi_val)*k2zi_val
    
    return ((t1 + t2)*k1zi_val +t3)*prod1 

def Y8TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1VHF1_val = c1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1HHF1_val = c1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF1_val = b1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF1_val = b1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF1_val = bb1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF1_val = bb1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s1z_val = s1z(k0, sx, sy, ep1)
    s2z_val = s2z(k0, sx, sy, ep2)
    kx_val = kx(k0, th,ph)
    ky_val = ky(k0, th,ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j*d*s1z_val)
    exp2 = exp(1j*d*s2z_val)
    prod1 = sx/(k0*sr_val)
    prod2 = (kx_val - sx)*sr_val/k0
    prod3 = sy/sr_val

    t1 = -c1HHF1_val*exp2
    t2 = (bb1HHF1_val/exp1 + b1HHF1_val*exp1)
    t3 = exp1*(b1VHF1_val*sq1*prod3 - b1HHF1_val*s1z_val*prod1)
    t4 = - (bb1VHF1_val*sq1*prod3 + bb1HHF1_val*s1z_val*prod1)/exp1
    t5 = - exp2*s2z_val*(c1VHF1_val*sq2*prod3 - c1HHF1_val*s2z_val*prod1)
    
    return  1j*((t1 + t2)*prod2 + s1z_val*(t3 + t4) + t5)

def Y8TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1VHF2_val = c1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1HHF2_val = c1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HHF2_val = b1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VHF2_val = b1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HHF2_val = bb1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VHF2_val = bb1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s1z_val = s1z(k0, sx, sy, ep1)
    s2z_val = s2z(k0, sx, sy, ep2)
    kx_val = kx(k0, th,ph)
    ky_val = ky(k0, th,ph)
    sr_val = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j*d*s1z_val)
    exp2 = exp(1j*d*s2z_val)
    prod1 = sx/(k0*sr_val)
    prod2 = (kx_val - sx)*sr_val/k0
    prod3 = sy/sr_val

    t1 = -c1HHF2_val*exp2
    t2 = (bb1HHF2_val/exp1 + b1HHF2_val*exp1)
    t3 = exp1*(b1VHF2_val*sq1*prod3 - b1HHF2_val*s1z_val*prod1)
    t4 = - (bb1VHF2_val*sq1*prod3 + bb1HHF2_val*s1z_val*prod1)/exp1
    t5 = - exp2*s2z_val*(c1VHF2_val*sq2*prod3 - c1HHF2_val*s2z_val*prod1)
    
    return  1j*((t1 + t2)*prod2 + s1z_val*(t3 + t4) + t5)

def q8TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L18TE_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y8TE_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q8TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L18TE_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y8TE_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

#########################
######## MODO TM ########

# coeficientes que dependen solo de los de orden cero --> no dependen de (sx,sy)
def L01TM_j(k0,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d)

    k1zi_ = k1zi(k0, thi, ep1)
    kiy_ = kiy(k0, thi, phi)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    sq1 = sqrt(ep1)

    t1 = (b0VV_-bb0VV_)*k1zi_**3
    t2 = (-1 + a0VV_)*sq1*kiz_**3
    
    return (t1 + t2)*kiy_/(2*k0*kir_*sq1)

def L02TM_j(k0,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d)

    k1zi_ = k1zi(k0, thi, ep1)
    kix_ = kix(k0, thi, phi)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    sq1 = sqrt(ep1 )

    prod1 = 2*k0*kir_
    
    return ((1 - a0VV_)*kix_*kiz_**3 + (bb0VV_ - b0VV_)*k1zi_**3*kix_/sq1)/prod1

def L03TM_j(k0,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d)

    k1zi_ = k1zi(k0, thi, ep1)
    kix_ = kix(k0, thi, phi)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    sq1 = sqrt(ep1 )

    prod1 = 2*kir_
    
    return ((1 + a0VV_)*kix_*kiz_**2 - (b0VV_ + bb0VV_)*k1zi_**2*kix_*sq1)/prod1

def L04TM_j(k0,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d)

    k1zi_ = k1zi(k0, thi, ep1)
    kiy_ = kiy(k0, thi, phi)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    sq1 = sqrt(ep1 )

    return ((1 + a0VV_)*kiy_*kiz_**2 - (b0VV_ + bb0VV_)*k1zi_**2*kiy_*sq1)/2*kir_

def L05TM_j(k0,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    kiy_ = kiy(k0, thi, phi)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)
    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    k0ir = k0*kir_
    exp1 = exp(1j*d*k1zi_)

    t1 = bb0VV_/exp1
    t2 = - b0VV_*exp1/sq1
    t3 = c0VV_*exp(1j*d*k2zi_)*(k2zi_**3)/sq2
    
    return ((t1 + t2)*k1zi_**3 + t3)*kiy_/(2*k0ir)

def L06TM_j(k0,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)
    
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    kix_ = kix(k0, thi, phi)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    k0ir = k0* kir_
    exp1 = exp(1j* d*k1zi_)

    t1 = -(bb0VV_/(exp1*sq1))*k1zi_**3
    t2 = (b0VV_/sq1)*exp1*k1zi_**3
    t3 = -c0VV_*exp(1j*d*k2zi_)*(k2zi_**3)/sq2
    
    return (t1 + t2 + t3)*kix_/(2*k0ir) 

def L07TM_j(k0,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    kix_ = kix(k0, thi, phi)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j* d*k1zi_)
    
    t1 = (bb0VV_*k1zi_**2)/exp1
    t2 = (b0VV_*exp1*k1zi_**2)
    t3 = -(c0VV_*exp(1j*d*k2zi_)*k2zi_**2*kix_*sq2)
    
    return  ((t1 + t2)*kix_*sq1 + t3)/kir_

def L08TM_j(k0,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)
    kiy_ = kiy(k0, thi, phi)
    kiz_ = kiz(k0, thi)
    kir_ = kir(k0, thi)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j* d*k1zi_)

    t1 = bb0VV_/exp1
    t2 = b0VV_*exp1
    t3 = c0VV_*exp(1j*d*k2zi_)*k2zi_**2*kiy_*sq2
    
    return  ((t1 + t2)*k1zi_**2*kiy_*sq1 + t3)/kir_

def L11TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d)

    k1zi_ = k1zi(k0, thi, ep1)
    
    sq1 = sqrt(ep1)
    prod1 = -kir(k0,thi)*(ky(k0,th,ph) - sy)
    parentesis = (b0VV_ - bb0VV_)*k1zi_ + (-1 + a0VV_)*kiz(k0,thi)*sq1
    
    return prod1*parentesis/(k0*sq1)

def Y1TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    a1HVF1_ = a1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VVF1_ = a1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF1_ = bb1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF1_ = b1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF1_ = bb1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF1_ = b1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    # kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    prod0 = sx/sr_
    prod1 = sy/(k0* sr_)
    prod2 = (ky_ - sy)*sr_/k0
    prod3 = s1z_ *prod1/sq1
    
    t1 = -s0z_*(a1HVF1_*prod0 + a1VVF1_*s0z_*prod1)
    t2 = a1VVF1_*prod2
    t3 = - (b1HVF1_*prod0 - b1VVF1_*prod3)
    t4 = (bb1HVF1_*prod0 + bb1VVF1_*prod3)
    t5 = - (b1VVF1_ + bb1VVF1_)*prod2/sq1 
    
    return 1j*(t1 + t2 + s1z_*(t3 + t4) + t5)

def Y1TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    a1HVF2_ = a1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VVF2_ = a1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF2_ = bb1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF2_ = b1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF2_ = bb1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF2_ = b1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    # kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    prod0 = sx/sr_
    prod1 = sy/(k0* sr_)
    prod2 = (ky_ - sy)*sr_/k0
    prod3 = s1z_ *prod1/sq1
    
    t1 = -s0z_*(a1HVF2_*prod0 + a1VVF2_*s0z_*prod1)
    t2 = a1VVF2_*prod2
    t3 = - (b1HVF2_*prod0 - b1VVF2_*prod3)
    t4 = (bb1HVF2_*prod0 + bb1VVF2_*prod3)
    t5 = - (b1VVF2_ + bb1VVF2_)*prod2/sq1 
    
    return 1j*(t1 + t2 + s1z_*(t3 + t4) + t5)

def q1TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L11TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y1TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q1TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L11TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y1TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def L12TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    a0VV_ = a0VV(k0, thi, ep1, ep2, d)

    sq1 = sqrt(ep1 )
    
    prod1 = kir(k0,thi)*(kx(k0,th,ph) - sx)
    parentesis = ((b0VV_ - bb0VV_)*k1zi(k0,thi,ep1) - (1 - a0VV_)*kiz(k0,thi)*sq1)
    return (prod1*parentesis)/(k0*sq1)

def Y2TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    a1HVF1_ = a1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VVF1_ = a1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF1_ = bb1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF1_ = b1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF1_ = bb1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF1_ = b1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    kx_ = kx(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    suma1 = (kx_ - sx)*sr_ /k0
    prod1 = sy/sr_
    prod2 = sx/(k0*sr_)
    
    t1 = -s0z_*(-a1VVF1_*s0z_*prod2 + a1HVF1_*prod1)
    t2 = - a1VVF1_*suma1
    t4 = (-b1HVF1_+bb1HVF1_)*prod1 - (b1VVF1_ + bb1VVF1_)*s1z_*prod2/sq1
    t5 = (b1VVF1_ + bb1VVF1_)*suma1/sq1
    
    return 1j*(t1 + t2 + s1z_*t4 + t5)

def Y2TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    a1HVF2_ = a1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VVF2_ = a1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF2_ = bb1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF2_ = b1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF2_ = bb1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF2_ = b1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    kx_ = kx(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    suma1 = (kx_ - sx)*sr_ /k0
    prod1 = sy/sr_
    prod2 = sx/(k0*sr_)
    
    t1 = -s0z_*(-a1VVF2_*s0z_*prod2 + a1HVF2_*prod1)
    t2 = - a1VVF2_*suma1
    t4 = (-b1HVF2_+bb1HVF2_)*prod1 - (b1VVF2_ + bb1VVF2_)*s1z_*prod2/sq1
    t5 = (b1VVF2_ + bb1VVF2_)*suma1/sq1
    
    return 1j*(t1 + t2 + s1z_*t4 + t5)

def q2TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L12TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y2TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q2TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L12TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y2TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def Y3TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    a1HVF1_ = a1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VVF1_ = a1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF1_ = bb1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF1_ = b1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF1_ = bb1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF1_ = b1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    ky_ = ky(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    prod1 = s1z_*sy/(k0*sr_)
    suma1 = (ky_ - sy)*sr_/k0
    prod2 = sx/sr_

    t1 = (b1HVF1_ + bb1HVF1_)
    t2 = - s0z_*(a1VVF1_*prod2 - a1HVF1_*prod1)
    t4 = - (b1VVF1_*sq1*prod2 + b1HVF1_*prod1)
    t5 = (bb1VVF1_*sq1*prod2 - bb1HVF1_*prod1)
    
    return 1j*(t2 + (t1 - a1HVF1_)*suma1 + s1z_*(t4 + t5))

def Y3TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    a1HVF2_ = a1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VVF2_ = a1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF2_ = bb1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF2_ = b1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF2_ = bb1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF2_ = b1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    ky_ = ky(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    prod1 = s1z_*sy/(k0*sr_)
    suma1 = (ky_ - sy)*sr_/k0
    prod2 = sx/sr_

    t1 = (b1HVF2_ + bb1HVF2_)
    t2 = - s0z_*(a1VVF2_*prod2 - a1HVF2_*prod1)
    t4 = - (b1VVF2_*sq1*prod2 + b1HVF2_*prod1)
    t5 = (bb1VVF2_*sq1*prod2 - bb1HVF2_*prod1)
    
    return 1j*(t2 + (t1 - a1HVF2_)*suma1 + s1z_*(t4 + t5))

def q3TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L13TM = 0
    return L13TM + Y3TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q3TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L13TM = 0
    return L13TM + Y3TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def Y4TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    a1HVF1_ = a1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VVF1_ = a1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF1_ = bb1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF1_ = b1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF1_ = bb1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF1_ = b1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    kx_ = kx(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    prod1 = s1z_*sx/(k0*sr_)
    prod2 = sy/sr_
    suma1 = (kx_ - sx)*sr_/k0
    
    t1 = (b1HVF1_ + bb1HVF1_)
    t2 = - s0z_*(a1HVF1_*prod1 + a1VVF1_*prod2)
    t4 = - (b1VVF1_*sq1*prod2 - b1HVF1_*prod1)
    t5 = (bb1VVF1_*sq1*prod2 + bb1HVF1_*prod1)
    
    return 1j*(t2 + (a1HVF1_ + t1)*suma1 + s1z_*(t4 + t5))

def Y4TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    a1HVF2_ = a1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    a1VVF2_ = a1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF2_ = bb1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF2_ = b1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF2_ = bb1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF2_ = b1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    kx_ = kx(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    prod1 = s1z_*sx/(k0*sr_)
    prod2 = sy/sr_
    suma1 = (kx_ - sx)*sr_/k0
    
    t1 = (b1HVF2_ + bb1HVF2_)
    t2 = - s0z_*(a1HVF2_*prod1 + a1VVF2_*prod2)
    t4 = - (b1VVF2_*sq1*prod2 - b1HVF2_*prod1)
    t5 = (bb1VVF2_*sq1*prod2 + bb1HVF2_*prod1)
    
    return 1j*(t2 + (a1HVF2_ + t1)*suma1 + s1z_*(t4 + t5))

def q4TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L14TM = 0
    return L14TM + Y4TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q4TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L14TM = 0
    return L14TM + Y4TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def L15TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)
    
    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)

    sq1 = sqrt(ep1 )

    denominador = exp(1j*d*k1zi_)*k0**2*sq1*sqrt(ep2)
    term1 = -c0VV_*exp(1j*d*(k1zi_ + k2zi_))*k0*k2zi_*sq1
    term2 = b0VV_*exp(2j*d*k1zi_)*k0*k1zi_*sqrt(ep2)
    term3 = - bb0VV_*k0*k1zi_*sqrt(ep2)
    
    return (kir(k0,thi)*(ky(k0,th,ph) - sy)*(term1 + term2 + term3))/denominador

def Y5TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1HVF1_ = c1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1VVF1_ = c1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF1_ = bb1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF1_ = b1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF1_ = bb1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF1_ = b1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    s2z_ = s2z(k0, sx, sy, ep2)
    ky_ = ky(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    prod1 = s1z_*sy/(k0*sr_*sq1)
    prod2 = sx/ sr_
    suma1 = (ky_ - sy)*sr_/k0
    exp1 = exp(1j*d*s1z_)
    exp2 = exp(1j*d*s2z_)

    t1 = exp1*(b1HVF1_*prod2 - b1VVF1_*prod1)
    t2 = - (bb1HVF1_*prod2 + bb1VVF1_*prod1)/exp1
    t3 = (bb1VVF1_/exp1 + b1VVF1_*exp1)*suma1/sq1
    t4 = - s2z_*(c1HVF1_*prod2 - c1VVF1_*s2z_*sy/(k0*sr_*sq2))
    t5 = -c1VVF1_*suma1/sq2
    
    return  1j*(s1z_*(t1 + t2) + t3 + exp2*(t4 + t5))

def Y5TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1HVF2_ = c1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1VVF2_ = c1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF2_ = bb1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF2_ = b1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF2_ = bb1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF2_ = b1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    s2z_ = s2z(k0, sx, sy, ep2)
    ky_ = ky(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    prod1 = s1z_*sy/(k0*sr_*sq1)
    prod2 = sx/ sr_
    suma1 = (ky_ - sy)*sr_/k0
    exp1 = exp(1j*d*s1z_)
    exp2 = exp(1j*d*s2z_)

    t1 = exp1*(b1HVF2_*prod2 - b1VVF2_*prod1)
    t2 = - (bb1HVF2_*prod2 + bb1VVF2_*prod1)/exp1
    t3 = (bb1VVF2_/exp1 + b1VVF2_*exp1)*suma1/sq1
    t4 = - s2z_*(c1HVF2_*prod2 - c1VVF2_*s2z_*sy/(k0*sr_*sq2))
    t5 = -c1VVF2_*suma1/sq2
    
    return  1j*(s1z_*(t1 + t2) + t3 + exp2*(t4 + t5))

def q5TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L15TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y5TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q5TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L15TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y5TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def L16TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    b0VV_ = b0VV(k0, thi, ep1, ep2, d)
    bb0VV_ = bb0VV(k0, thi, ep1, ep2, d)
    c0VV_ = c0VV(k0, thi, ep1, ep2, d)

    k1zi_ = k1zi(k0, thi, ep1)
    k2zi_ = k2zi(k0, thi, ep2)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )

    denominador = (exp(1j*d*k1zi_)*k0**2*sq1*sq2)
    term1 = c0VV_*exp(1j*d*(k1zi_ + k2zi_))*k0*k2zi_*sq1
    term2 = - b0VV_*exp(2j*d*k1zi_)*k0*k1zi_*sq2
    term3 = bb0VV_*k0*k1zi_*sq2
    
    return (kir(k0,thi)*(kx(k0,th,ph) - sx)*(term1 + term2 + term3))/denominador

def Y6TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1HVF1_ = c1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1VVF1_ = c1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF1_ = bb1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF1_ = b1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF1_ = bb1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF1_ = b1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    s2z_ = s2z(k0, sx, sy, ep2)
    kx_ = kx(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j* d*s1z_)
    exp2 = exp(1j* d*s2z_)
    prod1 = sx/(k0*sr_)
    prod2 = sy/sr_
    suma1 = (kx_ - sx)*sr_/k0

    t1 = exp1*(b1HVF1_*prod2 + b1VVF1_*s1z_*prod1/sq1)
    t2 = - (bb1HVF1_*prod2 - bb1VVF1_*s1z_*prod1/sq1)/exp1
    t3 = -(bb1VVF1_/exp1 + b1VVF1_*exp1)*suma1/sq1
    t4 = - s2z_*(c1HVF1_*prod2 + c1VVF1_*s2z_*prod1/sq2)
    t5 = c1VVF1_*suma1/sq2
    
    return  1j*(s1z_*(t1 + t2) + t3 + exp2*(t4 + t5))

def Y6TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1HVF2_ = c1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1VVF2_ = c1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF2_ = bb1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF2_ = b1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF2_ = bb1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF2_ = b1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    s2z_ = s2z(k0, sx, sy, ep2)
    kx_ = kx(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j* d*s1z_)
    exp2 = exp(1j* d*s2z_)
    prod1 = sx/(k0*sr_)
    prod2 = sy/sr_
    suma1 = (kx_ - sx)*sr_/k0

    t1 = exp1*(b1HVF2_*prod2 + b1VVF2_*s1z_*prod1/sq1)
    t2 = - (bb1HVF2_*prod2 - bb1VVF2_*s1z_*prod1/sq1)/exp1
    t3 = -(bb1VVF2_/exp1 + b1VVF2_*exp1)*suma1/sq1
    t4 = - s2z_*(c1HVF2_*prod2 + c1VVF2_*s2z_*prod1/sq2)
    t5 = c1VVF2_*suma1/sq2
    
    return  1j*(s1z_*(t1 + t2) + t3 + exp2*(t4 + t5))

def q6TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L16TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y6TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q6TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    return L16TM_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d) + Y6TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def Y7TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1HVF1_ = c1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1VVF1_ = c1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF1_ = bb1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF1_ = b1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF1_ = bb1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF1_ = b1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    s2z_ = s2z(k0, sx, sy, ep2)
    ky_ = ky(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j* d*s1z_)
    exp2 = exp(1j* d*s2z_)
    
    prod1 = sy/ (k0*sr_)
    prod2 = sx/sr_
    suma1 = (ky_ - sy)*sr_/ k0

    t1 = c1HVF1_*exp2
    t2 = - (bb1HVF1_/exp1 + b1HVF1_*exp1)
    t3 = exp1*(b1VVF1_*prod2 + b1HVF1_*s1z_*prod1/sq1)*sq1
    t4 = - (bb1VVF1_*prod2 - bb1HVF1_*s1z_*prod1/sq1)*sq1/exp1
    t5 = - exp2*s2z_*(c1VVF1_*prod2 + c1HVF1_*s2z_*prod1/sq2)*sq2

    return 1j*((t1 + t2)*suma1 + s1z_*(t3 + t4) + t5) 

def Y7TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1HVF2_ = c1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1VVF2_ = c1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF2_ = bb1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF2_ = b1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF2_ = bb1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF2_ = b1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    s2z_ = s2z(k0, sx, sy, ep2)
    ky_ = ky(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j* d*s1z_)
    exp2 = exp(1j* d*s2z_)
    
    prod1 = sy/ (k0*sr_)
    prod2 = sx/sr_
    suma1 = (ky_ - sy)*sr_/ k0

    t1 = c1HVF2_*exp2
    t2 = - (bb1HVF2_/exp1 + b1HVF2_*exp1)
    t3 = exp1*(b1VVF2_*prod2 + b1HVF2_*s1z_*prod1/sq1)*sq1
    t4 = - (bb1VVF2_*prod2 - bb1HVF2_*s1z_*prod1/sq1)*sq1/exp1
    t5 = - exp2*s2z_*(c1VVF2_*prod2 + c1HVF2_*s2z_*prod1/sq2)*sq2

    return 1j*((t1 + t2)*suma1 + s1z_*(t3 + t4) + t5) 

def q7TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L17TM = 0
    return L17TM + Y7TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q7TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L17TM = 0
    return L17TM + Y7TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def Y8TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1HVF1_ = c1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1VVF1_ = c1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF1_ = bb1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF1_ = b1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF1_ = bb1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF1_ = b1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    s2z_ = s2z(k0, sx, sy, ep2)
    kx_ = kx(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j* d*s1z_)
    exp2 = exp(1j* d*s2z_)
    
    prod1 = sx/(k0*sr_)
    prod2 = sy/sr_
    suma1 = (kx_ - sx)*sr_/k0

    t1 = -c1HVF1_*exp2
    t2 = (bb1HVF1_/exp1 + b1HVF1_*exp1)
    t3 = exp1*(b1VVF1_*prod2 - b1HVF1_*s1z_*prod1/sq1)*sq1
    t4 = - (bb1VVF1_*prod2 + bb1HVF1_*s1z_*prod1/sq1)*sq1/exp1
    t5 = - exp2*s2z_*(c1VVF1_*prod2 - c1HVF1_*s2z_*prod1/sq2)*sq2 
    
    return 1j*((t1 + t2)*suma1 + s1z_*(t3 + t4) + t5)

def Y8TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    c1HVF2_ = c1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    c1VVF2_ = c1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1VVF2_ = bb1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1VVF2_ = b1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    bb1HVF2_ = bb1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)
    b1HVF2_ = b1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)

    s0z_ = s0z(k0, sx, sy)
    s1z_ = s1z(k0, sx, sy, ep1)
    s2z_ = s2z(k0, sx, sy, ep2)
    kx_ = kx(k0, th,ph)
    sr_ = sr(sx, sy)

    sq1 = sqrt(ep1 )
    sq2 = sqrt(ep2 )
    exp1 = exp(1j* d*s1z_)
    exp2 = exp(1j* d*s2z_)
    
    prod1 = sx/(k0*sr_)
    prod2 = sy/sr_
    suma1 = (kx_ - sx)*sr_/k0

    t1 = -c1HVF2_*exp2
    t2 = (bb1HVF2_/exp1 + b1HVF2_*exp1)
    t3 = exp1*(b1VVF2_*prod2 - b1HVF2_*s1z_*prod1/sq1)*sq1
    t4 = - (bb1VVF2_*prod2 + bb1HVF2_*s1z_*prod1/sq1)*sq1/exp1
    t5 = - exp2*s2z_*(c1VVF2_*prod2 - c1HVF2_*s2z_*prod1/sq2)*sq2 
    
    return 1j*((t1 + t2)*suma1 + s1z_*(t3 + t4) + t5)

def q8TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    L18TM = 0
    return L18TM + Y8TM_F1_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

def q8TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d): 
    L18TM = 0
    return L18TM + Y8TM_F2_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d)

### Sección Eficaz ###


# **A orden uno es la ecuación (27) del paper de Moghaddam:
# falta escribir bien la correlación entre f1 y f2 - que yo puse gaussiana**

## Espectro de potencias de la superficie ##
# gaussiana
def w(s,l,k1,k2,acf):
    if(acf==1):
        # gaussiana
        return s**2*l**2/(4*np.pi)*np.exp(-0.25*l**2*(k1**2+k2**2))

    if(acf==2):
        # exponencial
        return s**2*l**2/(2*np.pi*(1+(k1**2+k2**2)*l**2)**(3/2))

    if(acf==3):
        # power law
        return s**2*l**2/(2*np.pi)*np.exp(-np.sqrt(k1**2+k2**2)*l)
  
    else:
        print('ACF invalida')


def w_12(s1,l1,s2,l2,k1,k2):
    return 0

def SO1HH_j(k0, th, ep1, ep2, d, s1, l1, s2, l2, acf):
     # = args
    
    thi = th
    phi = 0
    ph = np.pi
    
    k1 = k0 * (np.sin(th) * np.cos(ph) - np.sin(thi) * np.cos(phi))
    k2 = k0 * (np.sin(th) * np.sin(ph) - np.sin(thi) * np.sin(phi))
    
    return (
        abs(a1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)) ** 2 * w(s1, l1, k1, k2, acf) +
        abs(a1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)) ** 2 * w(s2, l2, k1, k2, acf) +
        2 * np.real(a1HHF1_j(k0, thi, phi, th, ph, ep1, ep2, d) * np.conj(a1HHF2_j(k0, thi, phi, th, ph, ep1, ep2, d))) * w_12(s1, l1, s2, l2, k1, k2)
    )

def SO1VH_j(k0, th, ep1, ep2, d, s1, l1, s2, l2, acf):
    # k0, th, ep1, ep2, d, s1, l1, s2, l2, acf = args
    
    thi = th
    phi = 0
    ph = np.pi
    
    k1 = k0 * (np.sin(th) * np.cos(ph) - np.sin(thi) * np.cos(phi))
    k2 = k0 * (np.sin(th) * np.sin(ph) - np.sin(thi) * np.sin(phi))
    
    return (
        abs(a1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d)) ** 2 * w(s1, l1, k1, k2, acf) +
        abs(a1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d)) ** 2 * w(s2, l2, k1, k2, acf) +
        2 * np.real(a1VHF1_j(k0, thi, phi, th, ph, ep1, ep2, d) * np.conj(a1VHF2_j(k0, thi, phi, th, ph, ep1, ep2, d))) * w_12(s1, l1, s2, l2, k1, k2)
    )

def SO1VV_j(k0, th, ep1, ep2, d, s1, l1, s2, l2, acf):
    # k0, th, ep1, ep2, d, s1, l1, s2, l2, acf = args
    
    thi = th
    phi = 0
    ph = np.pi
    
    k1 = k0 * (np.sin(th) * np.cos(ph) - np.sin(thi) * np.cos(phi))
    k2 = k0 * (np.sin(th) * np.sin(ph) - np.sin(thi) * np.sin(phi))
    
    return (
        abs(a1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)) ** 2 * w(s1, l1, k1, k2, acf) +
        abs(a1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)) ** 2 * w(s2, l2, k1, k2, acf) +
        2 * np.real(a1VVF1_j(k0, thi, phi, th, ph, ep1, ep2, d) * np.conj(a1VVF2_j(k0, thi, phi, th, ph, ep1, ep2, d))) * w_12(s1, l1, s2, l2, k1, k2)
    )

def SO1HV_j(k0, th, ep1, ep2, d, s1, l1, s2, l2, acf):
    # k0, th, ep1, ep2, d, s1, l1, s2, l2, acf = args
    
    thi = th
    phi = 0
    ph = np.pi
    
    k1 = k0 * (np.sin(th) * np.cos(ph) - np.sin(thi) * np.cos(phi))
    k2 = k0 * (np.sin(th) * np.sin(ph) - np.sin(thi) * np.sin(phi))
    
    return (
        abs(a1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d)) ** 2 * w(s1, l1, k1, k2, acf) +
        abs(a1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d)) ** 2 * w(s2, l2, k1, k2, acf) +
        2 * np.real(a1HVF1_j(k0, thi, phi, th, ph, ep1, ep2, d) * np.conj(a1HVF2_j(k0, thi, phi, th, ph, ep1, ep2, d))) * w_12(s1, l1, s2, l2, k1, k2)
    )

def SO1HHVV_j(k0, th, ep1, ep2, d, s1, l1, s2, l2, acf):
    # k0, th, ep1, ep2, d, s1, l1, s2, l2, acf = args
    
    thi = th
    phi = 0
    ph = np.pi
    
    k1 = k0 * (np.sin(th) * np.cos(ph) - np.sin(thi) * np.cos(phi))
    k2 = k0 * (np.sin(th) * np.sin(ph) - np.sin(thi) * np.sin(phi))
    
    return a1HHF1_j(k0,thi,phi,th,ph,ep1,ep2,d)*np.conj(a1VVF1_j(k0,thi,phi,th,ph,ep1,ep2,d))*w(s1,l1,k1,k2,acf)+\
            a1HHF2_j(k0,thi,phi,th,ph,ep1,ep2,d)*np.conj(a1VVF2_j(k0,thi,phi,th,ph,ep1,ep2,d))*w(s2,l2,k1,k2,acf)+\
            (a1HHF1_j(k0,thi,phi,th,ph,ep1,ep2,d)*np.conj(a1VVF2_j(k0,thi,phi,th,ph,ep1,ep2,d))+\
             a1HHF2_j(k0,thi,phi,th,ph,ep1,ep2,d)*np.conj(a1VVF1_j(k0,thi,phi,th,ph,ep1,ep2,d)))*w_12(s1,l1,s2,l2,k1,k2)

### Orden dos ###

### HH ###

def L0_11HH_j(k0,th,ph,thi,phi,ep1,ep2,d):
    L01TE_ = L01TE_j(k0, thi, phi, ep1, ep2, d)
    L02TE_ = L02TE_j(k0, thi, phi, ep1, ep2, d)
    L03TE_ = L03TE_j(k0, thi, phi, ep1, ep2, d)
    L04TE_ = L04TE_j(k0, thi, phi, ep1, ep2, d)

    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)

    exp1 = exp(2j*d*k1z_)
    sum1 = k1z_*(kx_*L01TE_+ky_*L02TE_)
    sum2 = k0*(ky_*L03TE_-kx_*L04TE_)

    denom = ((exp1*(k0z_-k1z_)*(k1z_-k2z_)+(k0z_+k1z_)*(k1z_+k2z_))*(kr_**2))
    
    return -kr_*(exp1*(k1z_-k2z_)*(sum1 + sum2)+(-k1z_-k2z_)*(sum1 - sum2))/denom

def L0_22HH_j(k0,th,ph,thi,phi,ep1,ep2,d):
    L05TE_ = L05TE_j(k0, thi, phi, ep1, ep2, d)
    L06TE_ = L06TE_j(k0, thi, phi, ep1, ep2, d)
    L07TE_ = L07TE_j(k0, thi, phi, ep1, ep2, d)
    L08TE_ = L08TE_j(k0, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    exp1 = exp(1j*d*k1z_)
    
    denom = (exp1**2*(k0z_-k1z_)*(k1z_- k2z_)+(k0z_+k1z_)*(k1z_+k2z_))*(kr_**2)
    return 2*exp1*k1z_*kr_* (k2z_*(kx_*L05TE_+ ky_* L06TE_)+k0*(-ky_*L07TE_+kx_*L08TE_))/denom

def L1_11HH_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q1TE = q1TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q2TE = q2TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q3TE = q3TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q4TE = q4TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kr_ = kr(k0, th)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    k12 = (k1z_-k2z_)
    k12m = (k1z_+k2z_)
    
    exp1 = exp(2j*d*k1z_)
    suma1 = (kx_*q1TE+ ky_*q2TE)
    suma2 = (ky_*q3TE - kx_*q4TE)
    
    denom = ((exp1*(k0z_-k1z_)*k12 + (k0z_+k1z_)*k12m)*(kr_**2)) 
    
    
    return -kr_*(exp1*k12*(k1z_*suma1 + k0*suma2)-k12m*(k1z_*suma1 - k0*suma2))/denom

def L1_22HH_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q5TE = q5TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q6TE = q6TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q7TE = q7TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q8TE = q8TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kr_ = kr(k0, th)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    exp1 = exp(1j*d*k1z_)
    
    denom = ((k0z_ - k1z_)*(k1z_ - k2z_)*exp1**2 + (k0z_ + k1z_)*(k1z_ + k2z_))*(kr_**2)
    
    return 2*exp1*k1z_*kr_*(k2z_*(kx_*q5TE + ky_*q6TE) - k0*(ky_*q7TE - kx_*q8TE))/denom

def L1_12HH_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q1TE = q1TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q2TE = q2TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q3TE = q3TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q4TE = q4TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q5TE = q5TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q6TE = q6TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q7TE = q7TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q8TE = q8TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kr_ = kr(k0, th)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    exp1 = exp(1j*d*k1z_)
    suma1 = k1z_*kx_*q1TE + k1z_*ky_*q2TE
    k0q3 = k0*ky_*q3TE
    k0q4 = k0*kx_*q4TE
    
    k12m = (k1z_ + k2z_)
    k12 = (k1z_ - k2z_)
    
    denom = (exp1**2*(k0z_ - k1z_)*k12 + (k0z_ + k1z_)*k12m)*(kx_**2 + ky_**2)
    t1 = -k12m*(suma1 - k0q3 + k0q4)
    t2 = exp1**2*k12*(suma1 + k0q3 - k0q4)
    t3 = - 2*exp1*k1z_*(k2z_*(kx_*q5TE + ky_*q6TE) - k0*(ky_*q7TE + kx_*q8TE))
    
    return -kr_*(t1 + t2 + t3)/denom

def SO2HH_j(k0, th, ep1, ep2, d, s1, l1, s2, l2, acf, n_gauss):
    
    thi = th
    phi = 0
    ph = np.pi
    
    k_lim = 1.5*k0
    X = int_gauss(n_gauss,k_lim)['X_IG']
    Y = int_gauss(n_gauss,k_lim)['Y_IG']
    m_gauss = int_gauss(n_gauss,k_lim)['m_gauss']
    Wt = int_gauss(n_gauss,k_lim)['Wt']
    
    kr_ = kr(k0, th)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    
    L0_11HH_ = L0_11HH_j(k0, th, ph, thi, phi, ep1, ep2, d)
    L0_22HH_ = L0_22HH_j(k0, th, ph, thi, phi, ep1, ep2, d)
    L1_11HH_ = L1_11HH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_12HH_ = L1_12HH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22HH_ = L1_22HH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    L1_11HH_val = L1_11HH_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22HH_val = L1_22HH_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    w1_val = w(s1, l1, kx_-X, ky_-Y, acf)
    w2_val = w(s2, l2, X-kix_, Y-kiy_, acf)
    L1Conj = np.conj(L1_11HH_ + L1_11HH_val)
    L2Conj = np.conj(L1_22HH_ + L1_22HH_val)
    
    
    W1 = w1_val*w(s1,l1,X-kix_,Y-kiy_,acf)
    t0_1 = 2*np.abs(L0_11HH_)**2
    t1_1 = 2*np.real(L0_11HH_*L1Conj)
    t2_1 = L1_11HH_*L1Conj
    out1 =  W1*(t0_1+t1_1+t2_1)
    
    

    W2 = w(s2,l2,kx_-X,ky_-Y,acf)*w2_val
    t0_2 = 2*np.abs(L0_22HH_)**2
    t1_2 = 2*np.real(L0_22HH_*L2Conj)
    t2_2 = L1_22HH_*L2Conj
    out2 = W2*(t0_2+t1_2+t2_2)
    
    
    W12 = w1_val*w2_val
    t12 = np.abs(L1_12HH_)**2
    out12 = W12*t12
    
    aux = out1 + out2 + out12
    
    cuenta = np.real(k_lim**2*np.nansum(np.nansum(Wt*aux.flatten())))
    
    return cuenta

### VH ###

def L0_11VH_j(k0,th,ph,thi,phi,ep1,ep2,d):
    L01TE_ = L01TE_j(k0, thi, phi, ep1, ep2, d)
    L02TE_ = L02TE_j(k0, thi, phi, ep1, ep2, d)
    L03TE_ = L03TE_j(k0, thi, phi, ep1, ep2, d)
    L04TE_ = L04TE_j(k0, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k1e2 = k1z_*ep2
    k2e1 = k2z_*ep1
    ksum = k2e1 + k1e2
    kres = -k2e1 + k1e2
    
    exp1 = exp(2j *d*k1z_)
    sum1 = k1z_*(kx_*L03TE_ + ky_*L04TE_)
    sum2 = k0*ep1*(ky_*L01TE_ - kx_*L02TE_)
    
    denom =  (kx_**2 + ky_**2)*(exp1*(k1z_ - k0z_*ep1)*kres - (k1z_ + k0z_*ep1)*ksum)

    return kr_*(-ksum*(sum1 + sum2) + exp1*kres*(sum1 - sum2))/denom

def L0_22VH_j(k0,th,ph,thi,phi,ep1,ep2,d):
    L05TE_ = L05TE_j(k0, thi, phi, ep1, ep2, d)
    L06TE_ = L06TE_j(k0, thi, phi, ep1, ep2, d)
    L07TE_ = L07TE_j(k0, thi, phi, ep1, ep2, d)
    L08TE_ = L08TE_j(k0, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k1e2 = k1z_*ep2
    k2e1 = k2z_*ep1
    exp1 = exp(1j *d*k1z_)
    
    denom = (kx_**2 + ky_**2)*(exp1**2*(k1z_ - k0z_*ep1)*(-k2e1 + k1e2) - (k1z_ + k0z_*ep1)*(k2e1 + k1e2))
    
    return  -(2*exp1*k1z_*kr_*ep1*(k0*ep2*(ky_*L05TE_ - kx_*L06TE_) + k2z_*(kx_*L07TE_ + ky_*L08TE_)))/denom

def L1_11VH_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q1TE = q1TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q2TE = q2TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q3TE = q3TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q4TE = q4TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kr_ = kr(k0, th)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k0e1 = k0z_*ep1
    k2e1 = k2z_*ep1
    k1e2 = k1z_*ep2
    
    exp1 = exp(2j *d*k1z_)
    suma1 = k0*ep1*(ky_*q1TE - kx_*q2TE)
    suma2 = k1z_*(kx_*q3TE + ky_*q4TE)
    suma3 = -(k2e1 + k1e2)
    
    denom = (kx_**2 + ky_**2)*(exp1*(k1z_ - k0e1)*(-k2e1 + k1e2) - (k1z_ + k0e1)*suma3)
    
    return  kr_*(suma3*(suma1 + suma2) + exp1*suma3*(suma1 - suma2))/denom

def L1_22VH_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q5TE = q5TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q6TE = q6TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q7TE = q7TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q8TE = q8TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kr_ = kr(k0, th)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k2e1 = k2z_*ep1
    k1e2 = k1z_*ep2
    k0e1 = k0z_*ep1
    exp1 = exp(1j *d*k1z_)
    
    denom = (kx_**2 + ky_**2)*(exp1**2*(k1z_ - k0e1)*(-k2e1 + k1e2) - (k1z_ + k0e1)*(k2e1 + k1e2))
    
    return  -2*exp1*k1z_*kr_*ep1*(k0*ep2*(ky_*q5TE - kx_*q6TE) + k2z_*(kx_*q7TE+ ky_*q8TE))/denom

def L1_12VH_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q1TE = q1TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q2TE = q2TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q3TE = q3TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q4TE = q4TE_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q5TE = q5TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q6TE = q6TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q7TE = q7TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q8TE = q8TE_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)

    k0z_ = k0z(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kr_ = kr(k0, th)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k0e1 = k0z_*ep1
    k1e2 = k1z_*ep2
    k2e1 = k2z_*ep1
    k12 = k1e2 - k2e1
    k12m = k2e1 + k1e2
    
    exp1 = exp(1j *d*k1z_)
    suma1 = (k0*ky_*q1TE*ep1 - k0*kx_*q2TE*ep1)
    suma2 = (k1z_*kx_*q3TE + k1z_*ky_*q4TE)
    
    denom = (kx_**2 + ky_**2)*(exp1**2*(k1z_ - k0e1)*k12 - (k1z_ + k0e1)*k12m)
    t1 = exp1**2*(suma2 - suma1)*k12
    t2 = -(suma2 + suma1)*k12m
    t3 = -2*exp1*k1z_*ep1*(k2z_*(kx_*q7TE +  ky_*q8TE) + k0*(ky_*q5TE - kx_*q6TE)*ep2)
    
    return kr_*(t1 + t2 + t3)/denom

def SO2VH_j(k0,th,ep1,ep2,d,s1,l1,s2,l2,acf,n_gauss):
    
    thi = th
    phi = 0
    ph = np.pi
    
    k_lim = 1.5*k0
    X = int_gauss(n_gauss,k_lim)['X_IG']
    Y = int_gauss(n_gauss,k_lim)['Y_IG']
    m_gauss = int_gauss(n_gauss,k_lim)['m_gauss']
    Wt = int_gauss(n_gauss,k_lim)['Wt']
    
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    
    L0_11VH_ = L0_11VH_j(k0, th, ph, thi, phi, ep1, ep2, d)
    L0_22VH_ = L0_22VH_j(k0, th, ph, thi, phi, ep1, ep2, d)
    L1_11VH_ = L1_11VH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_12VH_ = L1_12VH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22VH_ = L1_22VH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    L1_11VH_val = L1_11VH_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22VH_val = L1_22VH_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    w1_val = w(s1, l1, kx_-X, ky_-Y, acf)
    w2_val = w(s2, l2, X-kix_, Y-kiy_, acf)
    L1Conj = np.conj(L1_11VH_ + L1_11VH_val)
    L2Conj = np.conj(L1_22VH_ + L1_22VH_val)
    
    W1 = w1_val*w(s1,l1,X-kix_,Y-kiy_,acf)
    t0_1 = 2*np.abs(L0_11VH_)**2
    t1_1 = 2*np.real(L0_11VH_*L1Conj)
    t2_1 = L1_11VH_*L1Conj
    out1 = W1*(t0_1+t1_1+t2_1)
    
    W2 = w(s2,l2,kx_-X,ky_-Y,acf)*w2_val
    t0_2 = 2*np.abs(L0_22VH_)**2
    t1_2 = 2*np.real(L0_22VH_*L2Conj)
    t2_2 = L1_22VH_*L2Conj
    out2 = W2*(t0_2+t1_2+t2_2)
    
    W12 = w1_val*w2_val
    t12 = np.abs(L1_12VH_)**2
    out12 = W12*t12
    
    aux = out1 + out2 + out12
    
    cuenta = np.real(k_lim**2*np.nansum(np.nansum(Wt*aux.flatten())))
    
    return cuenta

### VV ###

def L0_11VV_j(k0,th,ph,thi,phi,ep1,ep2,d):
    L01TM_ = L01TM_j(k0, thi, phi, ep1, ep2, d)
    L02TM_ = L02TM_j(k0, thi, phi, ep1, ep2, d)
    L03TM_ = L03TM_j(k0, thi, phi, ep1, ep2, d)
    L04TM_ = L04TM_j(k0, thi, phi, ep1, ep2, d)

    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k0e1 = k0z_*ep1
    k1e2 = k1z_*ep2
    k2e1 = k2z_*ep1
    k12m = k2e1 + k1e2
    k12 = -k2e1 + k1e2
    
    exp1 = exp(2j *d*k1z_)
    sum1 = k0*ep1*(ky_*L01TM_ - kx_*L02TM_)
    sum2 = k1z_*(kx_*L03TM_ + ky_*L04TM_)
    
    denom = ((kx_**2 + ky_**2)*(exp1*(k1z_ - k0e1)*k12 - (k1z_ + k0e1)*k12m))
    
    return kr_*(-k12m*(sum1 + sum2) + exp1*k12*(-sum1 + sum2))/denom

def L0_22VV_j(k0,th,ph,thi,phi,ep1,ep2,d):
    L05TM_ = L05TM_j(k0, thi, phi, ep1, ep2, d)
    L06TM_ = L06TM_j(k0, thi, phi, ep1, ep2, d)
    L07TM_ = L07TM_j(k0, thi, phi, ep1, ep2, d)
    L08TM_ = L08TM_j(k0, thi, phi, ep1, ep2, d)

    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k0e1 = k0z_ *ep1
    k1e2 = k1z_ *ep2
    k2e1 = k2z_ *ep1
    k12m = k2e1 + k1e2
    k12 = -k2e1 + k1e2

    exp1 = exp(1j*d*k1z_)
    denom = (kx_**2 + ky_**2)*(exp1**2*(k1z_ - k0e1)*(-k2e1 + k1e2) - (k1z_ + k0e1)*(k2e1 + k1e2))
    return  -2*exp1*k1z_*kr_*ep1*(k0*ep2*(ky_*L05TM_ - kx_*L06TM_) + k2z_*(kx_*L07TM_ + ky_*L08TM_))/denom

def L1_11VV_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q1TM = q1TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q2TM = q2TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q3TM = q3TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q4TM = q4TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kr_ = kr(k0, th)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k0e1 = k0z_ *ep1
    k1e2 = k1z_ *ep2
    k2e1 = k2z_ *ep1
    k12m = k1e2 + k2e1 
    k12 = k1e2 - k2e1
    
    exp1 = exp(2j *d*k1z_)
    sum1 = k0*ep1*(ky_*q1TM - kx_*q2TM)
    sum2 = k1z_*(kx_*q3TM + ky_*q4TM)
    

    denom = (kx_**2 + ky_**2)*(exp1*(k1z_ - k0e1)*k12 - (k1z_ + k0e1)*k12m)
    
    return  (kr_*(-k12m*(sum1 + sum2) + exp1*k12*(-sum1 + sum2)))/denom

def L1_22VV_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q5TM = q5TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q6TM = q6TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q7TM = q7TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q8TM = q8TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kr_ = kr(k0, th)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k0e1 = k0z_ *ep1
    k1e2 = k1z_ *ep2
    k2e1 = k2z_ *ep1

    exp1 = exp(1j *d*k1z_)

    denom = (kx_**2 + ky_**2)*(exp1**2*(k1z_ - k0e1)*(-k2e1 + k1e2) - (k1z_ + k0e1)*(k2e1 + k1e2))
    
    return  -(2*exp1*k1z_*kr_*ep1*(k0*ep2*(ky_*q5TM - kx_*q6TM) + k2z_*(kx_*q7TM+ ky_*q8TM)))/denom

def L1_12VV_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q1TM = q1TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q2TM = q2TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q3TM = q3TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q4TM = q4TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q5TM = q5TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q6TM = q6TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q7TM = q7TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q8TM = q8TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    
    k0z_ = k0z(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    
    kr_ = kr(k0, th)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k0e1 = k0z_ *ep1
    k1e2 = k1z_ *ep2
    k2e1 = k2z_ *ep1
    k12 = -k2e1 + k1e2
    k12m = k2e1 + k1e2

    exp1 = exp(1j *d*k1z_)
    sum1 = k0*ky_*q1TM*ep1 - k0*kx_*q2TM*ep1
    sum2 = k1z_*kx_*q3TM + k1z_*ky_*q4TM
    
    denom = ((kx_**2 + ky_**2)*(exp1**2*(k1z_ - k0e1)*k12 - (k1z_ + k0e1)*k12m))

    return kr_*(exp1**2*(sum2 - sum1)*k12 - (sum2 + sum1)*k12m - 2*exp1*k1z_*ep1*(k2z_*(kx_*q7TM + ky_*q8TM) + k0*(ky_*q5TM*ep2 - kx_*q6TM*ep2)))/denom 

def SO2VV_j(k0,th,ep1,ep2,d,s1,l1,s2,l2,acf,n_gauss):

    thi = th
    phi = 0
    ph = np.pi
    
    k_lim = 1.5*k0
    X = int_gauss(n_gauss,k_lim)['X_IG']
    Y = int_gauss(n_gauss,k_lim)['Y_IG']
    m_gauss = int_gauss(n_gauss,k_lim)['m_gauss']
    Wt = int_gauss(n_gauss,k_lim)['Wt']

    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    
    L0_11VV_ = L0_11VV_j(k0, th, ph, thi, phi, ep1, ep2, d)
    L0_22VV_ = L0_22VV_j(k0, th, ph, thi, phi, ep1, ep2, d)
    L1_11VV_ = L1_11VV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_12VV_ = L1_12VV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22VV_ = L1_22VV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    L1_11VV_val = L1_11VV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22VV_val = L1_22VV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)

    w1_val = w(s1, l1, kx_-X, ky_-Y, acf)
    w2_val = w(s2, l2, X-kix_, Y-kiy_, acf)
    L1Conj = np.conj(L1_11VV_ + L1_11VV_val)
    L2Conj = np.conj(L1_22VV_ + L1_22VV_val)
    
    W1 = w1_val*w(s1,l1,X-kix_,Y-kiy_,acf)
    t0_1 = 2*np.abs(L0_11VV_)**2
    t1_1 = 2*np.real(L0_11VV_*L1Conj)
    t2_1 = L1_11VV_*L1Conj
    out1 = W1*(t0_1+t1_1+t2_1)
    
    W2 = w2_val*w(s2,l2,X-kix_,Y-kiy_,acf)
    t0_2 = 2*np.abs(L0_22VV_)**2
    t1_2 = 2*np.real(L0_22VV_*L2Conj)
    t2_2 = L1_22VV_*L2Conj
    out2 = W2*(t0_2+t1_2+t2_2)
    
    
    W12 = w1_val*w2_val
    t12 = np.abs(L1_12VV_)**2
    out12 = W12*t12
    
    
    aux = out1 + out2 + out12
    cuenta = np.real(k_lim**2*np.nansum(np.nansum(Wt*aux.flatten())))
    
    return cuenta

### HV ###

def L0_11HV_j(k0,th,ph,thi,phi,ep1,ep2,d):
    L01TM_ = L01TM_j(k0, thi, phi, ep1, ep2, d)
    L02TM_ = L02TM_j(k0, thi, phi, ep1, ep2, d)
    L03TM_ = L03TM_j(k0, thi, phi, ep1, ep2, d)
    L04TM_ = L04TM_j(k0, thi, phi, ep1, ep2, d)

    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)

    k12 = k1z_-k2z_
    k12m = k1z_+k2z_
    exp1 = exp(2j *d*k1z_)
    sum1 = k1z_*(kx_*L01TM_ + ky_*L02TM_)
    sum2 = k0*(ky_*L03TM_ - kx_*L04TM_)
    denom = (exp1*(k0z_-k1z_)*k12+(k0z_+k1z_)*k12m)*(kr_**2)

    return -kr_*(exp1*k12*(sum1 + sum2) - k12m*(sum1 - sum2))/denom

def L0_22HV_j(k0,th,ph,thi,phi,ep1,ep2,d):
    L05TM_ = L05TM_j(k0, thi, phi, ep1, ep2, d)
    L06TM_ = L06TM_j(k0, thi, phi, ep1, ep2, d)
    L07TM_ = L07TM_j(k0, thi, phi, ep1, ep2, d)
    L08TM_ = L08TM_j(k0, thi, phi, ep1, ep2, d)

    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)

    exp1 = exp(1j *d*k1z_)
    
    denom = (exp1**2*(k0z_-k1z_)*(k1z_-k2z_)+(k0z_+k1z_)*(k1z_+k2z_))* (kr_**2)
    
    return 2*exp1*k1z_*kr_* (k2z_*(kx_*L05TM_ + ky_* L06TM_) - k0*(ky_*L07TM_ - kx_*L08TM_))/denom

def L1_11HV_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q1TM = q1TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q2TM = q2TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q3TM = q3TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q4TM = q4TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)

    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    
    k12m = k1z_ + k2z_
    k12 = k1z_-k2z_
    
    exp1 = exp(2j *d*k1z_)
    sum1 = k1z_*(kx_*q1TM + ky_*q2TM)
    sum2 = k0*(ky_*q3TM - kx_*q4TM)
    denom = kr_*(exp1*(k0z_ - k1z_)*k12 + (k0z_ + k1z_)*k12m)
    
    return -(exp1*k12*(sum1 + sum2) - k12m*(sum1 - sum2))/denom

def L1_22HV_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q5TM = q5TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q6TM = q6TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q7TM = q7TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q8TM = q8TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)

    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)

    exp1 = exp(1j*d*k1z_)
    denom = (exp1**2*(k0z_ - k1z_)*(k1z_ - k2z_) + (k0z_ + k1z_)*(k1z_ + k2z_))*kr_

    return 2*exp1*k1z_*(k2z_*(kx_*q5TM + ky_*q6TM) - k0*((ky_*q7TM) - kx_*q8TM))/denom

def L1_12HV_j(sx,sy,k0,th,ph,thi,phi,ep1,ep2,d):
    q1TM = q1TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q2TM = q2TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q3TM = q3TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q4TM = q4TM_F2_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q5TM = q5TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q6TM = q6TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q7TM = q7TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)
    q8TM = q8TM_F1_j(sx, sy, k0, th, ph, thi, phi, ep1, ep2, d)

    k0z_ = k0z(k0, th)
    kr_ = kr(k0, th)
    k1z_ = k1z(k0, th, ep1)
    k2z_ = k2z(k0, th, ep2)
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)

    k12 = k1z_ - k2z_
    k12m = k1z_ + k2z_
    
    sum1 = k1z_*kx_*q1TM + k1z_*ky_*q2TM
    sum2 = k0*(ky_*q3TM - kx_*q4TM)
    sum3 = k2z_*(kx_*q5TM + ky_*q6TM)
    sum4 = k0*(ky_*q7TM - kx_*q8TM)
    exp1 = exp(1j*d*k1z_)
    
    denom = (exp1**2*(k0z_ - k1z_)*k12 + (k0z_ + k1z_)*k12m)*(kx_**2 + ky_**2)
    
    return -kr_*(-k12m*(sum1 - sum2) + exp1**2*k12*(sum1 + sum2) - 2*exp1*k1z_*(sum3 - sum4))/denom

def SO2HV_j(k0,th,ep1,ep2,d,s1,l1,s2,l2,acf,n_gauss):
    
    thi = th
    phi = 0
    ph = np.pi
    
    k_lim = 1.5*k0
    X = int_gauss(n_gauss,k_lim)['X_IG']
    Y = int_gauss(n_gauss,k_lim)['Y_IG']
    m_gauss = int_gauss(n_gauss,k_lim)['m_gauss']
    Wt = int_gauss(n_gauss,k_lim)['Wt']
    
    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)
    
    L0_11HV_ = L0_11HV_j(k0, th, ph, thi, phi, ep1, ep2, d)
    L0_22HV_ = L0_22HV_j(k0, th, ph, thi, phi, ep1, ep2, d)
    L1_11HV_ = L1_11HV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_12HV_ = L1_12HV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22HV_ = L1_22HV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    L1_11HV_val = L1_11HV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22HV_val = L1_22HV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    w1_val = w(s1, l1, kx_-X, ky_-Y, acf)
    w2_val = w(s2, l2, X-kix_, Y-kiy_, acf)
    L1Conj = np.conj(L1_11HV_ + L1_11HV_val)
    L2Conj = np.conj(L1_22HV_ + L1_22HV_val)
    
    W1 = w1_val*w(s1,l1,X-kix_,Y-kiy_,acf)
    t0_1 = 2*np.abs(L0_11HV_)**2
    t1_1 = 2*np.real(L0_11HV_*L1Conj)
    t2_1 = L1_11HV_*L1Conj
    out1 = W1*(t0_1+t1_1+t2_1)
    
    W2 = w(s2,l2,kx_-X,ky_-Y,acf)*w2_val
    t0_2 = 2*np.abs(L0_22HV_)**2
    t1_2 = 2*np.real(L0_22HV_*L2Conj)
    t2_2 = L1_22HV_*L2Conj
    out2 = W2*(t0_2+t1_2+t2_2)
    
    W12 = w1_val*w2_val
    t12 = np.abs(L1_12HV_)**2
    out12 = W12*t12
    
    aux = out1 + out2 + out12
    
    cuenta = np.real(k_lim**2*np.nansum(np.nansum(Wt*aux.flatten())))
    
    return cuenta

def SO2HHVV_j(k0,th,ep1,ep2,d,s1,l1,s2,l2,acf,n_gauss):
    
    thi = th
    phi = 0
    ph = np.pi
    
    k_lim = 1.5*k0
    X = int_gauss(n_gauss,k_lim)['X_IG']
    Y = int_gauss(n_gauss,k_lim)['Y_IG']
    m_gauss = int_gauss(n_gauss,k_lim)['m_gauss']
    Wt = int_gauss(n_gauss,k_lim)['Wt']

    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)

    L0_11HH_ = L0_11HH_j(k0, th, ph, thi, phi, ep1, ep2, d)
    L0_11VV_ = L0_11VV_j(k0, th, ph, thi, phi, ep1, ep2, d)
    
    L0_22HH_ = L0_22HH_j(k0, th, ph, thi, phi, ep1, ep2, d)
    L0_22VV_ = L0_22VV_j(k0, th, ph, thi, phi, ep1, ep2, d)
    
    L1_11HH_ = L1_11HH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_11VV_ = L1_11VV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    L1_12HH_ = L1_12HH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_12VV_ = L1_12VV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    L1_22HH_ = L1_12HV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22VV_ = L1_22VV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    L1_11VV_val = L1_11VV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_11HH_val = L1_11HH_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22VV_val = L1_22VV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    L1_22HH_val = L1_22HH_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d)
    
    w1_val = w(s1, l1, kx_-X, ky_-Y, acf)
    w2_val = w(s2, l2, X-kix_, Y-kiy_, acf)
    L0ConjVV = np.conj(L0_11VV_)
    L0ConjVV_2 = np.conj(L0_22VV_)
    L1ConjVV = np.conj(L1_11VV_ + L1_11VV_val)
    L2ConjVV = np.conj(L1_22VV_ + L1_22VV_val)
    
    W1 = w1_val*w(s1,l1,X-kix_,Y-kiy_,acf)

    t0_1 = 2*L0_11HH_*L0ConjVV
    t1_1 = L0_11HH_*L1ConjVV + L0ConjVV*(L1_11HH_ + L1_11HH_val)
    t2_1 = L1_11HH_*L1ConjVV
    out1 =  W1*(t0_1+t1_1+t2_1)

    W2 = w2_val*w(s2,l2,X-kix_,Y-kiy_,acf)
    
    t0_2 = 2*L0_22HH_*L0ConjVV_2
    t1_2 = L0_22HH_*L2ConjVV + L0ConjVV_2*(L1_22HH_ + L1_22HH_val)
    t2_2 = L1_22HH_*L2ConjVV
    out2 = W2*(t0_2+t1_2+t2_2)
    
    
    W12 = w1_val*w2_val
    t12 = L1_12HH_*np.conj(L1_12VV_)

    out12 = W12*t12
    
    aux = out1 + out2 + out12
    cuenta = k_lim**2*np.nansum(np.nansum(Wt*aux.flatten()))
    
    return cuenta

def SO2HHHV_j(k0,th,ep1,ep2,d,s1,l1,s2,l2,acf,n_gauss):
    
    thi = th
    phi = 0
    ph = np.pi
    
    k_lim = 1.5*k0
    X = int_gauss(n_gauss,k_lim)['X_IG']
    Y = int_gauss(n_gauss,k_lim)['Y_IG']
    m_gauss = int_gauss(n_gauss,k_lim)['m_gauss']
    Wt = int_gauss(n_gauss,k_lim)['Wt']

    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)

    L0_11HH_ = L0_11HH_j(k0, th, ph, thi, phi, ep1, ep2, d) #
    L0_11HV_ = L0_11HV_j(k0, th, ph, thi, phi, ep1, ep2, d) #
    
    L0_22HH_ = L0_22HH_j(k0, th, ph, thi, phi, ep1, ep2, d) #
    L0_22HV_ = L0_22HV_j(k0, th, ph, thi, phi, ep1, ep2, d) #
    
    L1_11HH_ = L1_11HH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    L1_11HV_ = L1_11HV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #

    L1_12HH_ = L1_12HH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    L1_12HV_ = L1_12HV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    
    L1_22HH_ = L1_22HH_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    L1_22HV_ = L1_22HV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #

    L1_11HH_val = L1_11HH_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    L1_11HV_val = L1_11HV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d) #

    L1_22HH_val = L1_11HH_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    L1_22HV_val = L1_11HV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d) #

    w1_val = w(s1, l1, kx_-X, ky_-Y, acf)
    w2_val = w(s2, l2, X-kix_, Y-kiy_, acf)

    sum2 = np.conj(L1_11HV_+L1_11HV_val)
    conj1 = np.conj(L0_11HV_)
    W1 = w1_val*w(s1,l1,X-kix_,Y-kiy_,acf)
    
    t0_1 = 2*L0_11HH_*conj1
    t1_1 = L0_11HH_*sum2 + conj1*(L1_11HH_ + L1_11HH_val)
    t2_1 = L1_11HH_*sum2
    out1 =  W1*(t0_1+t1_1+t2_1)
    
    sum1 = np.conj(L1_22HV_ + L1_22HH_val)
    conj2 = np.conj(L0_22HV_)
    W2 = w2_val*w(s2,l2,X-kix_,Y-kiy_,acf)
    
    t0_2 = 2*L0_22HH_*conj2
    t1_2 = L0_22HH_*sum1 + conj2*(L1_22HH_ + L1_22HH_val)
    t2_2 = L1_22HH_*sum1
    out2 = W2*(t0_2+t1_2+t2_2)
    
    
    W12 = w(s1,l1,kx_-X,ky_-Y,acf)*w(s2,l2,X-kix_,Y-kiy_,acf)
    t12 = L1_12HH_*np.conj(L1_12HV_)

    out12 = W12*t12
    
    aux = out1 + out2 + out12
    cuenta = k_lim**2*np.nansum(np.nansum(Wt*aux.flatten()))
    
    return cuenta

def SO2VVHV_j(k0,th,ep1,ep2,d,s1,l1,s2,l2,acf,n_gauss):
    
    thi = th
    phi = 0
    ph = np.pi
    
    k_lim = 1.5*k0
    X = int_gauss(n_gauss,k_lim)['X_IG']
    Y = int_gauss(n_gauss,k_lim)['Y_IG']
    m_gauss = int_gauss(n_gauss,k_lim)['m_gauss']
    Wt = int_gauss(n_gauss,k_lim)['Wt']

    kx_ = kx(k0, th,ph)
    ky_ = ky(k0, th,ph)
    kix_ = kix(k0, thi, phi)
    kiy_ = kiy(k0, thi, phi)

    L0_11VV_ = L0_11VV_j(k0, th, ph, thi, phi, ep1, ep2, d) #
    L0_11HV_ = L0_11HV_j(k0, th, ph, thi, phi, ep1, ep2, d) #
    
    L0_22VV_ = L0_22VV_j(k0, th, ph, thi, phi, ep1, ep2, d) #
    L0_22HV_ = L0_22HV_j(k0, th, ph, thi, phi, ep1, ep2, d) #
    
    L1_11VV_ = L1_11VV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    L1_11HV_ = L1_11HV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #

    L1_12VV_ = L1_12VV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    L1_12HV_ = L1_12HV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    
    L1_22VV_ = L1_22VV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    L1_22HV_ = L1_22HV_j(X, Y, k0, th, ph, thi, phi, ep1, ep2, d) #

    L1_11VV_val = L1_11VV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    L1_11HV_val = L1_11HV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d) #

    L1_22VV_val = L1_11VV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d) #
    L1_22HV_val = L1_11HV_j(kx_+kix_-X, ky_+kiy_-Y, k0, th, ph, thi, phi, ep1, ep2, d) #

    w1_val = w(s1, l1, kx_-X, ky_-Y, acf)
    w2_val = w(s2, l2, X-kix_, Y-kiy_, acf)

    sum2 = np.conj(L1_11HV_+L1_11HV_val)
    conj1 = np.conj(L0_11HV_)
    W1 = w1_val*w(s1,l1,X-kix_,Y-kiy_,acf)
    
    t0_1 = 2*L0_11VV_*conj1
    t1_1 = L0_11VV_*sum2 + conj1*(L1_11VV_ + L1_11VV_val)
    t2_1 = L1_11VV_*sum2
    out1 =  W1*(t0_1+t1_1+t2_1)
    
    sum1 = np.conj(L1_22HV_ + L1_22VV_val)
    conj2 = np.conj(L0_22HV_)
    W2 = w2_val*w(s2,l2,X-kix_,Y-kiy_,acf)
    
    t0_2 = 2*L0_22VV_*conj2
    t1_2 = L0_22VV_*sum1 + conj2*(L1_22VV_ + L1_22VV_val)
    t2_2 = L1_22VV_*sum1
    out2 = W2*(t0_2+t1_2+t2_2)
    
    
    W12 = w(s1,l1,kx_-X,ky_-Y,acf)*w(s2,l2,X-kix_,Y-kiy_,acf)
    t12 = L1_12VV_*np.conj(L1_12HV_)

    out12 = W12*t12
    
    aux = out1 + out2 + out12
    cuenta = k_lim**2*np.nansum(np.nansum(Wt*aux.flatten()))
    
    return cuenta
