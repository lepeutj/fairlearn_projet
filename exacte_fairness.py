import numpy as np
import matplotlib
import matplotlib . pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,precision_score
#fairness
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import demographic_parity_ratio
from fairlearn.postprocessing import ThresholdOptimizer 
from fairlearn.postprocessing import plot_threshold_optimizer
from fairlearn.metrics import equalized_odds_difference
from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate, selection_rate, count
from fairlearn.reductions import DemographicParity
#optimiser / solver
import matplotlib as 


######################## Génération des samples ##########################
def sampleexemple(n,p,seed=100): 
    #génération d'un sample de taille n, avec un proportion p sur S, y proportion de Y à 1
    #avec 
    #K=2 (2 classes),
    #m=1(mélange gaussien d'une loi normale),
    #d=2 (2 variables explicatives X1 et X2)
    np.random.seed(seed)
    I1=1
    I2=1
    C0=np.random.uniform(0, 1)
    C1=np.random.uniform(0, 1)
    mu01=np.random.normal(0, I1)
    mu02=np.random.normal(0, I2)
    mu11=np.random.normal(0, I1)
    mu12=np.random.normal(0, I2)
    
    df = pd.DataFrame(0, columns = ['X1', 'X2','S','Y'], index = np.arange(n), dtype = float)   
    df['Y'] = np.random.binomial(1,0.5,n) 
    for i in range(n):
        
        if df.loc[i, 'Y']  == 1:
            df.loc[i, 'X1']  =np.random.normal(C1+mu11, I1)
            df.loc[i, 'X2']  =np.random.normal(C1+mu12, I2)
            df.loc[i, 'S']  =2*(np.random.binomial(1,p))-1
        else:
            df.loc[i, 'X1']  =np.random.normal(C0+mu01, I1)
            df.loc[i, 'X2']  =np.random.normal(C0+mu02, I2)
            df.loc[i, 'S']  = 2*(np.random.binomial(1,(1-p)))-1    

    return df


# Génération de 3 échantillons, 60, 20, 20
def sample_data(df):
    train, val_test = train_test_split(df, test_size=0.4, random_state=42)
    val, ev = train_test_split(val_test, test_size=0.5, random_state=42)
    return train, val, ev



sample=sampleexemple(5000,0.9)
sample=sampleexemple(5000,0.7)
sample=sampleexemple(5000,0.5)


tr, te, ev = sample_data(sample)

#Echantillon d'entrainement
X_train = tr.drop(['Y'],axis=1)
y_train = tr['Y']
#Echantillon test
X_test = te.drop(['Y'],axis=1)
y_test = te['Y']
#Echantillon évaluation
X_eval = ev.drop(['Y'],axis=1)
y_eval = ev['Y']


# Calibration régression logistique
lm = LogisticRegression(max_iter=10000).fit(X_train, y_train)
y_predlm=lm.predict(X_test)
probas_LOGIT = lm.predict_proba(X_test)

# LDA=LinearDiscriminantAnalysis()
# LDA.fit(X_train,y_train)

# y_predLDA=LDA.predict(X_test)
# probas_LDA = LDA.predict_proba(X_test)

# Tirage uniforme pour perturber des probas
zeta_1 = np.random.uniform(0,10**(-5))
zeta_2 = np.random.uniform(0,10**(-5))

# Définition de deux vecteurs, p proba classe 0 et q proba classe 1
p = [x + zeta_1 for x in probas_LOGIT[:,0]]
q = [x + zeta_2 for x in probas_LOGIT[:,1]]

# p = [x + zeta_1 for x in probas_LDA[:,0]]
# q = [x + zeta_2 for x in probas_LDA[:,1]]

# On récupère les indices pour lesquels S = 1 ou S= -1

X_test = X_test.reset_index()
u = X_test['S']==1
v = X_test['S']==-1

# pi_s , pi_p pour S = 1, pi_m pour S = -1
pi_p = u.sum()/len(y_test)
pi_m = v.sum()/len(y_test)

p=np.array(p)
q=np.array(q)

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x*(1/0.005))
    return e_x / e_x.sum(axis=0)

# Fonction à minimiser en lambda pour trouver le classifieur fair
def h(lam):
    term1 = (1/len(u)*sum([max(x) for x in zip(pi_p*p[u]-1*(lam[0]),pi_p*q[u]-1*(lam[1]))])) #S=1
    term2 = (1/len(v)*sum([max(x) for x in zip(pi_m*p[v]+1*(lam[0]),pi_m*q[v]+1*(lam[1]))])) #S=-1
    return term1 + term2 

# def h(lam):
#     term1 = (1/len(u)*sum([sum(softmax(x)*x) for x in zip(pi_p*p[u]-1*(lam[0]),pi_p*q[u]-1*(lam[1]))])) #S=1
#     term2 = (1/len(v)*sum([sum(softmax(x)*x) for x in zip(pi_m*p[v]+1*(lam[0]),pi_m*q[v]+1*(lam[1]))])) #S=-1
#     return term1 + term2 


from scipy.optimize import minimize

# Définition des bornes
bnds = ((10**(-7),10**5),(10**(-7),10**5))
# Minimisation selon SLSQP
#res = minimize(h,[np.random.uniform(0,1000),np.random.uniform(0,1000)], bounds = bnds , method = "SLSQP")
res = minimize(h,[np.random.uniform(0,500),np.random.uniform(0,500)], bounds = bnds , method = "SLSQP")
#res = minimize(h,[0,0], bounds = bnds , method = "SLSQP")

res.x[0]-res.x[1]
probas_LOGIT_eval = lm.predict_proba(X_eval)

p_ev = probas_LOGIT_eval[:,0] 
q_ev = probas_LOGIT_eval[:,1] 

# probas_LDA = LDA.predict_proba(X_eval)

# p = probas_LDA[:,0] 
# q = probas_LDA[:,1] 

# Génération de lois uniforme pour perturbation : 
t = np.random.uniform(0,10**(-5),2)

# Nouveau classifieur 
fair = []
for i in range(len(y_eval)):
    if X_eval.reset_index()['S'][i] == 1:
        fair.append(np.argmax([pi_p*(p_ev[i]+t[0])-1*(res.x[0]),pi_p*(q_ev[i]+t[1])-1*(res.x[1])]))
    else :
        fair.append(np.argmax([pi_m*(p_ev[i]+t[0])+1*(res.x[0]),pi_m*(q_ev[i]+t[1])+1*(res.x[1])]))

# Observation par rapport à nos prédictions initiales :

X_eval = X_eval.reset_index()
X_eval = X_eval.drop('index',axis=1)
X_eval = X_eval.drop('level_0',axis=1)

fair= np.array(fair)
#p_esti(g_fair(x,s)=1|s=1)
np.mean(fair[X_eval['S']==1]==1) - np.mean(fair[X_eval['S']==-1]==1)
#p_esti(g(x,s)=1|s=1)

y_eval_lm = lm.predict(X_eval)
np.mean(y_eval_lm[X_eval['S']==1]) - np.mean(y_eval_lm[X_eval['S']==-1])




#Accuracy algo fair
np.diag(confusion_matrix(y_eval,fair)).sum()/len(y_eval)
#Accuracy algo
np.diag(confusion_matrix(y_eval,lm.predict(X_eval))).sum()/len(y_eval)
res.x