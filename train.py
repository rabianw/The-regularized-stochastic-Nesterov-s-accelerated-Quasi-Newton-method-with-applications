import numpy as np
import pandas as pd
import time

'''Input data'''
data = pd.read_csv("your_data", 
                    header=None, 
                    delim_whitespace=False)


x = data.values[:, :-1]
t = data.values[:, -1]
t[t == 0] = -1


''' 5-fold cross validation '''

from sklearn.model_selection import KFold
n_folds = 5
cv = KFold(n_splits=n_folds, shuffle=True, random_state=12345678)


''' setting matrix '''
acc_train_RESNAQ = np.empty((n_folds))
acc_test_RESNAQ = np.empty((n_folds))
processingtime_RESNAQ = np.empty((n_folds))
df_objresnaq = pd.DataFrame(columns=["fold1", "fold2","fold3","fold4","fold5"])
df_mu = pd.DataFrame(columns=["fold1", "fold2","fold3","fold4","fold5"])
df_time_resnaq = pd.DataFrame(columns=["fold1", "fold2","fold3","fold4","fold5"])



''' Import evaluation package '''
import time
from sklearn.metrics import accuracy_score, confusion_matrix


''' Import models '''
from res_naq_method import ResNaq



''' setting hyperparameter '''
max_epochs_    = 50
n_batches_sgd_ = 1
n_batches_     = 5
C_     = 0.0001 
delta_ = 0.0001 
gamma_ = 0.0001 
mu__   = 0.99
time_  = 8000
timesgd_ = 40000
epsilon_0_ = 1e-3      #<---step size
tau_0_ = 4*(10**8)     #<---step size




''' Testing Cross Validation '''
for i, (train, test) in enumerate(cv.split(x)):
    
    '''  choose model  '''   
    model = ResNaq(C = C_, delta = delta_, gamma = gamma_, mu = mu__, 
                    max_epochs = max_epochs_, n_batches = n_batches_, time = time_,
                    epsilon_0 = epsilon_0_, tau_0 = tau_0_)
    # steamp time
    start = time.time()
    
    # modeling
    model.fit(x[train], t[train])
    
    # cpu time used
    processingtimes_RESNAQ = time.time() - start
    processingtime_RESNAQ[i] = time.time() - start
    
    # prediction
    y_train = model.predict(x[train])
    y_test = model.predict(x[test])
    


    
    # evaluation model
    print('Fold: %2d' % (i+1))
    print(df_objresnaq)
    print('Final iteration = %.d' % model.final_iter)
    print('CPU time = %.4f' % (processingtimes_RESNAQ))
    print("Training accuracy = %.2f %% " % (accuracy_score(t[train],y_train)*100))
    print("Test accuracy     = %.2f %% " % (accuracy_score(t[test],y_test)*100))
    
    
    acc_train_RESNAQ[i]= accuracy_score(t[train], y_train)
    acc_test_RESNAQ[i] = accuracy_score(t[test], y_test)


print("Using RESNAQ")
print('CPU time = %.4f and its S.D. = %.4f' % (np.mean(processingtime_RESNAQ), np.std(processingtime_RESNAQ)))
print("Training accuracy = %.4f and its S.D. = %.4f" % (np.mean(acc_train_RESNAQ), np.std(acc_train_RESNAQ)))
print("Test accuracy     = %.4f and its S.D. = %.4f" % (np.mean(acc_test_RESNAQ), np.std(acc_test_RESNAQ)))
print('#'*75)


