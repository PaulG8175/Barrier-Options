import numpy as np
import matplotlib.pyplot as plt

# Box-Muller : génère n*N gaussiennes depuis des uniformes
def bx_mu(n,N):
    U,V = np.random.rand(n,N),np.random.rand(n,N)
    return np.sqrt(-2*np.log(U))*np.cos(2*np.pi*V)

# CDF de la loi normale centrée réduite (Abramowitz & Stegun)
def Abramowitz(x):
    if x<0:
        return 1-Abramowitz(-x)
    else:
        b0 = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        t = 1/(1+b0*x)
        return 1 - np.exp(-x**2 /2) * (b1*t+b2* t**2 +b3* t**3 +b4* t**4 +b5 * t**5)/np.sqrt(2*np.pi)

# Variance empirique sans biais
def var_empi(X):
    return np.sum((X-np.mean(X))**2)/(np.size(X)-1)


# Q1 : S(t) = S0 * exp((r-sig**2 /2)*t+ sig*Wt)

# Q2 : On a : P_euro = -S0*phi(-d1) + K*exp(-r*T)*phi(-d2) avec phi fonction de répartition de N(0,1) et 
# d1 = (ln(S0/K)+(r+0.5*sig**2)*T)/(sig*sqrt(T)) et d2 = d1 - sig*sqrt(T)

# Paramètres du modèle
sig = 0.15
S0=1
r=0.015
T=2
K=1
q_95 = 1.645

# ============================================================
# Q3-Q4) Put européen vanille par MC standard
# P_euro = e^{-rT} * E[(K - S(T))+]
# ============================================================

# W(T) = sqrt(T) * eps avec eps ~ N(0,1) via Box-Muller
def mbs(T,N,eps):
    dt = T/N
    W = np.cumsum(np.sqrt(dt)*eps,axis=1)
    return W

# S(t) = S0 * exp((r - 0.5*sig²)*t + sig*W(t))
def St(t,S0,sig,r,W):
    return S0*np.exp((r-0.5*sig**2)*t + sig*W)

# Estimateur MC du put européen + IC à 90%
def P_MC(T,K,r,sig,S0,eps):
    W = mbs(T,1,eps)
    S = St(T,S0,sig,r,W)
    P_list = np.exp(-r*T) * np.maximum(K-S,0)
    P = np.mean(P_list)
    var = var_empi(P_list)
    IC = np.array([P-q_95*np.sqrt(var/np.size(P_list)),P+q_95*np.sqrt(var/np.size(P_list))])
    return P,IC


n_list = np.array([1000,3000,5000,10000,30000,50000,100000,300000,500000,1000000])

P_euro_list = np.zeros(np.size(n_list))
IC_list = np.zeros((np.size(n_list),2))

# Prix théorique Black-Scholes du put
d1 = (np.log(S0/K) + (r+ 0.5* sig**2)*T)/(sig*np.sqrt(T))
d2 = d1 - sig*np.sqrt(T)
P_theo = -S0 * Abramowitz(-d1) + K*np.exp(-r*T)*Abramowitz(-d2)

for i,n in enumerate(n_list):
    eps = bx_mu(n,1)
    P_euro_list[i],IC_list[i] = P_MC(T,K,r,sig,S0,eps)


plt.figure()
plt.plot(n_list,P_euro_list,label="Monte Carlo")
plt.fill_between(n_list,IC_list[:,0],IC_list[:,1],alpha=0.3,label="IC 90%")
plt.axhline(y=P_theo,color="black",label="Valeur théorique")
plt.legend()
plt.title("Estimateurs de P_euro par Monte Carlo")
plt.xlabel("N")
plt.ylabel("Prix option")
plt.xscale("log")
plt.yscale("log")
plt.show()


# Q5 : on pose Xt = ln(St) = ln(S0)+(r- 0.5* sig**2)*t + sig*Wt, et on obtient min(Su)>= B <=> min(Xu)>=ln(B)
# en remarquant que  1{minS>=B} = 1 - 1{minS<=B}, on peut transformer P_DO, en un put classique et P_DO mais inversé (min(S)<=B)

# ============================================================
# Q6) Put Down-and-Out par MC discret
# P_DO,delta = e^{-rT} * E[(K-S(T))+ * 1_{min_{Ti} S(Ti) >= B}]
# On simule N_delta pas de taille delta=1/52 (hebdomadaire)
# ============================================================

B=0.7
delta = 1/52
N_delta = int(T/delta)
T_i = np.arange(1,N_delta+1)*delta
dt = T/N_delta

def PDO_delta(T,K,r,sig,S0,B,eps):
    W = mbs(T,N_delta,eps)
    S_u = St(T_i,S0,sig,r,W)

    # Payoff nul si le minimum discret des S(Ti) passe sous B
    PDO_list = np.exp(-r*T)*np.maximum(K-S_u[:,-1],0)*(np.min(S_u,axis=1)>=B)
    PDO_delt = np.mean(PDO_list)

    var = var_empi(PDO_list)
    IC = np.array([PDO_delt - q_95*np.sqrt(var/np.size(PDO_list)),PDO_delt + q_95*np.sqrt(var/np.size(PDO_list))])

    return PDO_delt,IC


eps = bx_mu(1,N_delta)
W = mbs(T,N_delta,eps)


# ============================================================
# Q7) Réduction de variance par variables antithétiques
# On moyenne les payoffs de (W) et (-W)
# ============================================================

def PDO_delta_anti(T,K,r,sig,S0,B,eps):
    W = mbs(T,N_delta,eps)
    S_u_plus = St(T_i,S0,sig,r,W)
    S_u_moins = St(T_i,S0,sig,r,-W)

    PDO_list_plus = np.exp(-r*T)*np.maximum(K-S_u_plus[:,-1],0)*(np.min(S_u_plus,axis=1)>=B)
    PDO_list_moins = np.exp(-r*T)*np.maximum(K-S_u_moins[:,-1],0)*(np.min(S_u_moins,axis=1)>=B)
    # Estimateur antithétique : moyenne des deux payoffs
    PDO_list = 0.5*(PDO_list_plus+PDO_list_moins)
    PDO = np.mean(PDO_list)

    var = var_empi(PDO_list)
    IC = np.array([PDO - q_95*np.sqrt(var/np.size(PDO_list)),PDO + q_95*np.sqrt(var/np.size(PDO_list))])

    return PDO,IC


n_list = np.array([1000,3000,5000,10000,30000,50000,100000,300000,500000,1000000])
nbr_tra = np.size(n_list)

PDO_list = np.zeros(nbr_tra)
PDO_anti_list = np.zeros(nbr_tra)
IC_list,IC_anti_list = np.zeros((2,nbr_tra)),np.zeros((2,nbr_tra))

for i,n in enumerate(n_list):
    eps = bx_mu(n,N_delta)
    PDO_list[i],IC_list[:,i] = PDO_delta(T,K,r,sig,S0,B,eps)
    PDO_anti_list[i],IC_anti_list[:,i] = PDO_delta_anti(T,K,r,sig,S0,B,eps)

plt.figure()
plt.title("Estimateurs de PDO par Monte Carlo par rapport à n")
plt.plot(n_list,PDO_list,label="Monte Carlo")
plt.plot(n_list,PDO_anti_list,label="Monte Carlo anti")
plt.fill_between(n_list,IC_list[0,:],IC_list[1,:],alpha=0.3,label="IC 90%")
plt.fill_between(n_list,IC_anti_list[0,:],IC_anti_list[1,:],alpha=0.3,label="IC anti 90%")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Nombre de simulations")
plt.ylabel("prix de l'option")
plt.show()


"""
On voit que les deux méthodes convergent vers une valeur d'environ de 0.046. De plus, on voit que la méthode antithétique 
possède une variance plus faible que la méthode classique ce qui est prévisible dans ce modèle avec Monte Carlo.
De plus, on voit que la valeur de PDO_delta est infèrieur à celle d'un put classique (0.069>0.046). Cela était aussi prévisible
d'après la théorie car (K-S(T))+ * 1{minS(u)>=B} <=(K-S(T))+
"""


# Q8 : N=1e5 → IC 90% de longueur ≈ 0.018% → précision suffisante
# P_DO,delta décroît avec B : plus B est petit, plus le put DO ressemble au put vanille

n=100000
nbr_B = 50
B_list = np.linspace(0.5,1,nbr_B)

PDO_anti_list_B = np.zeros(nbr_B)
eps = bx_mu(n,N_delta)
for i,B in enumerate(B_list):
    PDO_anti_list_B[i] = PDO_delta_anti(T,K,r,sig,S0,B,eps)[0]

plt.figure()
plt.title("Estimateurs de PDO par rapport à B")
plt.plot(B_list,PDO_anti_list_B)
plt.xlabel("B")
plt.ylabel("Valeurs de l'option")
plt.show()

B=0.7
"""
On observe une courbe décroissante qui tend vers 0 lorsque B augmente. De plus, on voit que PDO tend vers un put classique car 
en effet plus B est petit, plus la proba que minS(u)>=B est grande.
"""


# Q9 : effet de sigma sur P_DO pour deux valeurs de S0
# S0=1 : si sigma→0, S reste proche de 1 >> B → P_DO → 0
# S0=0.8 : si sigma→0, S reste proche de 0.8 > B → put ITM mais sans risque de barrière → P_DO > 0

n=10000
B=0.7
nbr_sig = 50
sig_list = np.linspace(0,0.8,nbr_sig)

PDO_anti_list_sig_1 = np.zeros(nbr_sig)
PDO_anti_list_sig_2 = np.zeros(nbr_sig)

eps = bx_mu(n,N_delta)

for i,sig in enumerate(sig_list):
    PDO_anti_list_sig_1[i] = PDO_delta_anti(T,K,r,sig,S0,B,eps)[0]
    PDO_anti_list_sig_2[i] = PDO_delta_anti(T,K,r,sig,0.8,B,eps)[0]

plt.figure()
plt.title("Estimateurs de PDO par rapport à sigma")
plt.plot(sig_list,PDO_anti_list_sig_1,label="S0=1")
plt.plot(sig_list,PDO_anti_list_sig_2,label="S0=0.8")
plt.legend()
plt.xlabel("sigma")
plt.ylabel("Valeurs de l'option")
plt.show()


"""
On observe une grande différence entre les deux courbes. Pour S0=1, il semble que si sigma=0, la valeur de l'option vaut 0
et pour S0=0.8 et sigma=0 on a une valeur du put d'environ de 0.17. C'est prévisible car si S0=1 sans volatilité, la probabilité que la trajectoire
de St aille en dessous de B=0.7 est faible alors que si S0=0.8, la probabilité ets bien plus élevé.
De plus, les deux courbes tendent vers 0 lorsque sigma tend vers 0.8, ce qui explique par le fait qu'avec une volatilité élevé, la proba que la trajectoire St
aille au moins une fois en dessous de B ets élevé.
"""

# ============================================================
# Q10) Correction Brownian Bridge
# P[inf_{u in [t1,t2]} W_u < x* | W_t1, W_t2] = exp(-2*(W_t1-x*)*(W_t2-x*)/(t2-t1))
# On pondère chaque payoff par la probabilité de survie continue entre observations
# ============================================================

sig=0.15

def PDO_10(T,K,r,sig,S0,B,eps):
    W = mbs(T,N_delta,eps)
    S_u = St(T_i,S0,sig,r,W)

    # Probabilité de passage sous B entre deux observations consécutives (pont brownien)
    indicatrice = (S_u[:,1:]>B)&(S_u[:,:-1]>B)
    pi = np.exp(-2*(np.log(S_u[:,1:]/B)*np.log(S_u[:,:-1]/B))/(sig**2 *dt))
    pi_OK = np.where(indicatrice,pi,1.0)
    # Probabilité de survie continue sur [0,T] = produit des probabilités de survie par intervalle
    proba_OK = np.exp(np.sum(np.log(np.clip(1 - pi_OK,1e-300,1.0)),axis=1))
    payoff = np.maximum(K-S_u[:,-1],0.0)
    # Si knock-out discret détecté, payoff nul ; sinon pondéré par la proba de survie continue
    payoff_OK = np.where(np.any(S_u[:,1:]<B,axis=1),0.0,payoff*proba_OK)

    PDO_list = np.exp(-r*T) * payoff_OK
    PDO = np.mean(PDO_list)
    var = var_empi(PDO_list)
    IC = np.array([PDO - q_95*np.sqrt(var/np.size(PDO_list)),PDO + q_95*np.sqrt(var/np.size(PDO_list))])

    return PDO,IC


n_list = np.array([1000,3000,5000,10000,30000,50000,100000,300000,500000,1000000])
nbr_tra = np.size(n_list)

delta = 1/52
N_delta = int(T/delta)
T_i = np.arange(1,N_delta+1)*delta

PDO_list = np.zeros(nbr_tra)
IC_list = np.zeros((2,nbr_tra))

for i,n in enumerate(n_list):
    eps = bx_mu(n,N_delta)
    PDO_list[i],IC_list[:,i] = PDO_10(T,K,r,sig,S0,B,eps)

plt.figure()
plt.title("Estimateurs de PDO par PDO_delta par rapport à n")
plt.plot(n_list,PDO_list,label="PDO_delta")
plt.fill_between(n_list,IC_list[0,:],IC_list[1,:],alpha=0.3,label="IC 90%")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Nombre de simulations")
plt.ylabel("prix de l'option")
plt.show()


# Q11 : convergence de P_DO,delta vers P_DO en fonction de delta
# La correction Brownian Bridge (PDO_10) converge bien plus vite que le MC discret brut

n=100000

delta_list = np.array([1/250,1/52,1/12,1/4,1])

PDO_delta_list = np.zeros(np.size(delta_list))
PDO_anti_list = np.zeros(np.size(delta_list))


for i,delta in enumerate(delta_list):
    N_delta = np.maximum(1,int(T/delta))
    eps = bx_mu(n,N_delta)
    dt = T/N_delta
    T_i = np.arange(1,N_delta+1)*delta
    PDO_delta_list[i] = PDO_10(T,K,r,sig,S0,B,eps)[0]
    PDO_anti_list[i] = PDO_delta(T,K,r,sig,S0,B,eps)[0]

plt.figure()
plt.title("Estimateurs de PDO")
plt.plot(delta_list,PDO_delta_list,label="PDO_delta")
plt.plot(delta_list,PDO_anti_list,label="MC antithétique")
plt.legend()
plt.xlabel("delta")
plt.ylabel("Prix option")
plt.xscale("log")
plt.yscale("log")
plt.show()


# ============================================================
# Q12) Probabilité de non-sortie zeta_DO
# zeta_discret : fraction de trajectoires dont le min discret >= B
# zeta_continue (BB) : corrigé du biais de discrétisation via pont brownien
# Valeur théorique : formule fermée basée sur le réflexion du mouvement brownien
# ============================================================

def zeta_delta(T,r,sig,S0,B,eps,N_delta,T_i):
    W = mbs(T,N_delta,eps)
    S_u = St(T_i,S0,sig,r,W)
    zeta = np.mean((np.min(S_u,axis=1))>=B)
    return zeta

def zeta_delta_10(T,r,sig,S0,B,eps,N_delta,T_i,dt):
    W = mbs(T,N_delta,eps)
    S_u = St(T_i,S0,sig,r,W)

    KO = np.any(S_u<B,axis=1)  # knock-out discret détecté
    indicatrice = (S_u[:, 1:]>B)&(S_u[:,:-1]>B) 
    pi = np.exp(-2*(np.log(S_u[:,1:]/B)*np.log(S_u[:,:-1]/B))/(sig**2 *dt))
    pi_OK = np.where(indicatrice,pi,1.0)
    proba_OK = np.exp(np.sum(np.log(np.clip(1 - pi_OK,1e-300,1.0)),axis=1))
    zeta = np.mean(np.where(KO,0.0,proba_OK))
    return zeta


zeta_delta_list = np.zeros(np.size(delta_list))
zeta_delta_10_list = np.zeros(np.size(delta_list))
# Formule fermée de la probabilité de non-sortie (formule de réflexion du BM)
mu = r - 0.5* sig**2
d1 = (np.log(S0/B) + mu*T)/(sig*np.sqrt(T))
d2 = (np.log(B/S0) + mu*T)/(sig*np.sqrt(T))
lamb = 2*mu/sig**2
zeta_theo = Abramowitz(d1) - (B/S0)**(lamb)*Abramowitz(d2)

for i,delta in enumerate(delta_list):
    N_delta = np.maximum(1,int(T/delta))
    T_i = np.arange(1,N_delta+1)*delta
    dt_i = T/N_delta
    eps = bx_mu(n,N_delta)
    zeta_delta_list[i] = zeta_delta(T,r,sig,S0,B,eps,N_delta,T_i)
    zeta_delta_10_list[i] = zeta_delta_10(T,r,sig,S0,B,eps,N_delta,T_i,dt_i)

plt.figure()
plt.title("Probabilité de sortie zeta_delta en fonction de delta")
plt.plot(delta_list,zeta_delta_list,label="zeta_delta discret")
plt.plot(delta_list,zeta_delta_10_list,label="zeta_delta continue")
plt.axhline(y=zeta_theo,color="black",label="zeta théorique")
plt.legend()
plt.xlabel("delta")
plt.ylabel("Probabilité")
plt.show()


"""
On observe que zeta_delta_discret décroît de 0.953 (delta=1) vers 0.917 (delta=1/250) quand delta diminue, mais reste systématiquement au-dessus de la valeur théorique (~0.912).
 Ce biais positif est attendu : plus delta est grand, plus on rate de passages sous B, et plus on surestime la probabilité de survie.
zeta_delta_continue est quant à lui quasi constant à ~0.912 pour tous les delta et coïncide avec la valeur théorique,
confirmant que la correction Brownian Bridge élimine le biais de discrétisation même avec un delta grossier.
"""


# Q13 : P_DO + P_DI = P_euro car 1{minS>=B} + 1{minS<=B} = 1 p.s.
# => P_DI = P_euro - P_DO (variable de contrôle à espérance connue)

# ============================================================
# Q14) Variable de contrôle : P_DI = P_euro - P_DO
# Z = P_DO + (P_DI - E[P_DI]) = P_euro - P_DI_MC + E[P_DI]
# Efficace quand B petit : P_DI ≈ 0, Var(P_DI) ≈ 0 → forte réduction
# Inefficace quand B ≈ K : P_DO et P_DI ont des variances comparables
# ============================================================

d1 = (np.log(S0/K) + (r+ 0.5* sig**2)*T)/(sig*np.sqrt(T))
d2 = d1 - sig*np.sqrt(T)
P_theo = -S0 * Abramowitz(-d1) + K*np.exp(-r*T)*Abramowitz(-d2)


n=100000
delta = 1/52
N_delta = int(T/delta)
T_i = np.arange(1,N_delta+1)*delta
nbr_B = 30
B_list = np.linspace(0.5,1,nbr_B)

var_MC_list = np.zeros(nbr_B)
var_CV_list = np.zeros(nbr_B)

# Même trajectoires pour les deux estimateurs (variables communes)
eps = bx_mu(n,N_delta)
W = mbs(T,N_delta,eps)
S_u = St(T_i,S0,sig,r,W)
payoff = np.exp(-r*T)*np.maximum(K- S_u[:,-1],0)

for i,B in enumerate(B_list):
    P_DI_list = payoff * (np.min(S_u,axis=1)<=B)

    var_MC_list[i] = var_empi(payoff*(np.min(S_u,axis=1)>=B))
    var_CV_list[i] = var_empi(P_theo - P_DI_list)  # P_DO = P_euro - P_DI

plt.figure()
plt.title("Variance estimateurs P_DO en fonction de B")
plt.plot(B_list,var_MC_list,label="MC")
plt.plot(B_list,var_CV_list,label="Variable controle")
plt.legend()
plt.xlabel("B")
plt.ylabel("Var")
plt.show()


"""
Pour B petit, presque toutes les trajectoires survivent donc P_DI~0 et sa variance est quasi nulle:la variable de contrôle est très efficace.
 A l'inverse, MC classique a une variance élevée car P_DO fluctue beaucoup.
Pour B proche de 1, les trajectoires sont partagées équitablement entre knock-out et survie:var(P_DI)~var(P_DO) et la variable de contrôle n'apporte plus rien.
La réduction de variance est donc d'autant plus efficace que B est petit.
"""
