import numpy as np
import time

class sgd():

    def __init__(self, C = 1, max_epochs = 1, n_batches = 5, time = 100,
                epsilon_0 = 0.0001, tau_0 = 0.0001):
        assert C > 0
        self.C = C
        assert epsilon_0 > 0
        self.epsilon_0 = epsilon_0
        assert tau_0 > 0
        self.tau_0 = tau_0
        assert max_epochs > 0 and isinstance(max_epochs, int)
        self.max_epochs=max_epochs
        assert n_batches > 0 and isinstance(n_batches, int)
        self.n_batches=n_batches
        assert time > 0 
        self.time = time

    '''gradient of squareHingeloss'''
    def gradient(self, w, x, t):
        u = (np.maximum(0, 1-((t[:, np.newaxis]*x).dot(w))))
        p = -2 * u * (t[:,np.newaxis]*x)
        grad = self.C * w + p
        return u,p,grad
                

    ''' cost function (squareHingeloss SVM) '''
    def cost_function(self,w,x,t):
        loss = 1/len(t) *  (np.sum((np.maximum(0, 1-((t[:, np.newaxis]*x).dot(w)))**2)))
        cost = self.C/2 * np.linalg.norm(w)**2  + loss
        return loss, cost

    def fit(self, x, t):
        # checks for labels
        self.classes_ = np.unique(t)
        t[t==0] = -1

        # initail variables k, w_0
        k = 0
        w = np.ones(len(x[0]))                
        #w = np.zeros(len(x[0]))
        obj_sgd = []
        iter_sgd = []

        # set up matrices
        time_sgd = np.empty(self.time)

        for epoch in range(self.max_epochs):
            idx = np.random.permutation(len(t))
            print("Epoch: %d" %(epoch+1), idx)

            for i in range(len(t)):
                r = idx[i]
                if r.size==0: break 

                k = k + 1
                iter_sgd.append(k)

                # stamp time
                start_time = time.perf_counter()

                # compute cost function
                loss, cost = self.cost_function(w,x,t)
                obj_sgd.append(cost)
                
                print("----Iteration: %d" %(i+1), r)
                
                # compute gradient 
                grad = -2*(np.maximum(0, 1-((t[r, np.newaxis]*x[r,:]).dot(w)))) * (t[r, np.newaxis]*x[r,:]) + self.C * w

                # step size
                #self.eta = ((1e-3) * (4*10**8))/((4*10**8)+k)
                eta =((self.epsilon_0) * (self.tau_0))/((self.tau_0)+k)
                
                # update weight
                w -= eta*grad
                print("----w: ",w)

                # cpu time used
                stop_time = time.perf_counter()
                time_sgd[i] = stop_time - start_time
                
                        
        self.final_iter = k
        self._coef = w
        self.obj_sgd = obj_sgd
        self.iter_sgd = iter_sgd
        self.time_sgd = time_sgd

        return self

    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef))
        p[p==0] = 1
        return p.astype(int)

class nag():

    def __init__(self, C = 1, mu = 0.9, epsilon_0 = 0.0001, tau_0 = 0.0001,
                max_epochs = 1, n_batches = 5, time = 100):
        assert C > 0
        self.C = C
        assert epsilon_0 > 0
        self.epsilon_0 = epsilon_0
        assert tau_0 > 0
        self.tau_0 = tau_0
        assert 0 < mu < 1
        self.mu = mu
        assert max_epochs > 0 and isinstance(max_epochs, int)
        self.max_epochs=max_epochs
        assert n_batches > 0 and isinstance(n_batches, int)
        self.n_batches=n_batches
        assert time > 0 
        self.time = time

    '''gradient of squareHingeloss'''
    def gradient(self, w, x, t):
        u = np.maximum(0, 1-((t[:, np.newaxis]*x).dot(w)))
        p = -2/len(t)*u.dot(t[:,np.newaxis]*x)
        grad = self.C * w + p
        return u,p,grad
                

    ''' cost function (squareHingeloss SVM) '''
    def cost_function(self,w,x,t):
        loss = 1/len(t) *  (np.sum((np.maximum(0, 1-((t[:, np.newaxis]*x).dot(w)))**2)))
        cost = self.C/2 * np.linalg.norm(w)**2  + loss
        return loss, cost

    def fit(self, x, t):
        # checks for labels
        self.classes_ = np.unique(t)
        t[t==0] = -1

        # initail variables k, w_0
        k = 0
        w = np.ones(len(x[0]))                
        #w = np.zeros(len(x[0]))
        z = np.zeros(len(x[0]))
        obj_nag = []
        iter_nag = []

        # set up matrices
        time_nag = np.empty(self.time)

        for epoch in range(self.max_epochs):
            idx = np.random.permutation(len(t))
            print("Epoch: %d" %(epoch+1), idx)
            for i in range(len(t)):
                
                r = idx[i*self.n_batches:(i+1)*self.n_batches]
                if r.size==0: break 

                k = k + 1
                iter_nag.append(k)

                # stamp time
                start_time = time.perf_counter()

                # compute cost function
                loss, cost = self.cost_function(w,x,t)
                obj_nag.append(cost)
                
                print("----Iteration: %d" %(i+1), r)
                
                
                # compute nesterov'gradient 
                _, _, g_naq = self.gradient(w+(self.mu*z),x[r,:],t[r])
                

                # step size
                #self.eta = ((1e-3) * (4*10**8))/((4*10**8)+k)
                eta =((self.epsilon_0) * (self.tau_0))/((self.tau_0)+k)
                
                # compute nesterov'direction
                z_new = self.mu*z - eta * g_naq
                
                
                # update weight
                w_new = w + z_new
                print("----w: ",w)

                # update step
                w = w_new
                z = z_new

                # cpu time used
                stop_time = time.perf_counter()
                time_nag[i] = stop_time - start_time
                
                        
        self.final_iter = k
        self._coef = w
        self.obj_nag = obj_nag
        self.iter_nag = iter_nag
        self.time_nag = time_nag

        return self

    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef))
        p[p==0] = 1
        return p.astype(int)

class batchsgd():

    def __init__(self, C = 1, max_epochs = 1, n_batches = 5, time = 100,
                epsilon_0 = 0.0001, tau_0 = 0.0001):
        assert C > 0
        self.C = C
        assert epsilon_0 > 0
        self.epsilon_0 = epsilon_0
        assert tau_0 > 0
        self.tau_0 = tau_0
        assert max_epochs > 0 and isinstance(max_epochs, int)
        self.max_epochs=max_epochs
        assert n_batches > 0 and isinstance(n_batches, int)
        self.n_batches=n_batches
        assert time > 0 
        self.time = time

    '''gradient of squareHingeloss'''
    def gradient(self, w, x, t):
        u = np.maximum(0, 1-((t[:, np.newaxis]*x).dot(w)))
        p = -2/len(t)*u.dot(t[:,np.newaxis]*x)
        grad = self.C * w + p
        return u,p,grad
                

    ''' cost function (squareHingeloss SVM) '''
    def cost_function(self,w,x,t):
        loss = 1/len(t) *  (np.sum((np.maximum(0, 1-((t[:, np.newaxis]*x).dot(w)))**2)))
        cost = self.C/2 * np.linalg.norm(w)**2  + loss
        return loss, cost

    def fit(self, x, t):
        # checks for labels
        self.classes_ = np.unique(t)
        t[t==0] = -1

        # initail variables k, w_0
        k = 0
        w = np.ones(len(x[0]))                
        #w = np.zeros(len(x[0]))
        obj_batchsgd = []
        iter_batchsgd = []

        # set up matrices
        time_batchsgd = np.empty(self.time)

        for epoch in range(self.max_epochs):
            idx = np.random.permutation(len(t))
            print("Epoch: %d" %(epoch+1), idx)
            for i in range(len(t)):
                
                r = idx[i*self.n_batches:(i+1)*self.n_batches]
                if r.size==0: break

                k = k + 1
                iter_batchsgd.append(k)

                # stamp time
                start_time = time.perf_counter()

                # compute cost function
                loss, cost = self.cost_function(w,x,t)
                obj_batchsgd.append(cost)
                
                print("----Iteration: %d" %(i+1), r)
                
                # compute gradient 
                _, _, grad = self.gradient(w,x[r,:],t[r])
                
                
                # step size
                #self.eta = ((1e-3) * (4*10**8))/((4*10**8)+k)
                eta =((self.epsilon_0) * (self.tau_0))/((self.tau_0)+k)
                
                # update weight
                w -= eta*grad
                print("----w: ",w)

                # cpu time used
                stop_time = time.perf_counter()
                time_batchsgd[i] = stop_time - start_time
                
                        
        self.final_iter = k
        self._coef = w
        self.obj_batchsgd = obj_batchsgd
        self.iter_batchsgd = iter_batchsgd
        self.time_batchsgd = time_batchsgd

        return self

    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef))
        p[p==0] = 1
        return p.astype(int)


