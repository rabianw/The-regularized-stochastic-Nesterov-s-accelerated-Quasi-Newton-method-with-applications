import numpy as np
import time

#RES_NAQ for SVM 

class ResNaq():
    def __init__(self, C = 1, delta = 0.0001, gamma = 0.0001,
                epsilon_0 = 0.0001, tau_0 = 0.0001,
                mu = 0.99, max_epochs = 1, n_batches = 5, time = 100):
        assert C > 0
        self.C = C
        assert epsilon_0 > 0
        self.epsilon_0 = epsilon_0
        assert tau_0 > 0
        self.tau_0 = tau_0
        assert delta > 0
        self.delta = delta
        assert gamma > 0
        self.gamma = gamma
        assert 0 <= mu < 1
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

    ''' approximate ResNaq_BFGS matrix '''
    def Update_RES_BFGS(self,B, dw, dg, I):
        dg_t =  dg[:, np.newaxis]
               
        Bdw = np.dot(B, dw)
        Bdw = np.dot(B, dw)
        dw_t_B = np.dot(dw, B)
        dwBdw = np.dot(np.dot(dw, B), dw)

        p = dg_t*dg
        u = Bdw[:, np.newaxis] * dw_t_B

        B_new = B + p / np.dot(dg, dw) - u / dwBdw + self.delta * I
        return p, u, B_new

    def fit(self, x, t):
        #initail variables k, stop, w_0
        k = 0
        w = np.ones(len(x[0])) 
        #w = np.zeros(len(x[0]))                
        z = np.zeros(len(x[0]))
        
        #set up matrices
        obj_resnaq = []
        objnat_resnaq = []
        iter_resnaq = []
        mu_ = []       
        time_resnaq = np.empty(self.time)
        B = np.identity(len(x[0]))  #inverse hessian
        I = np.identity(len(x[0]))  #identity 

        
        for epoch in range(self.max_epochs):
            idx = np.random.permutation(len(t))
            print("Epoch: %d" %(epoch+1), idx)
            for i in range(len(t)):
                
                r = idx[i*self.n_batches:(i+1)*self.n_batches]
                if r.size==0: break
                
                k = k + 1
                iter_resnaq.append(k)

                # stamp time
                start_time = time.perf_counter()
          
                # compute cost function at w
                _, cost = self.cost_function(w,x,t)
                obj_resnaq.append(cost)

                # compute cost function at w+(mu*z)
                _, cost_naq = self.cost_function(w+(self.mu*z),x,t)
                objnat_resnaq.append(cost_naq)


                print("----Iteration: %d" %(i+1), r)
                # print(X[r,], t[r]) # <-- used to train ml models.
                
                if cost < cost_naq:
                    mu_t = 0 
                    print('----mu:', mu_t)
                    mu_.append(mu_t)

                    # compute nesterov'gradient 
                    _, _, g_naq = self.gradient(w+(mu_t*z),x[r,:],t[r])
                
                    #if np.linalg.norm(g_naq)==0: break 
                  
                    # update hessian matrix
                    H = np.linalg.inv(B)
                    H_RES = H + self.gamma*I
                  
                    # step size
                    #self.eta = ((1e-3) * (4*10**8))/((4*10**8)+k)
                    eta =((self.epsilon_0) * (self.tau_0))/((self.tau_0)+k)


                    # compute nesterov'direction
                    z_new = mu_t*z - eta * np.matmul(H_RES,g_naq)

                    # update weight
                    w_new = w + z_new
                
                    # compute new gradient 
                    _, _, g_new = self.gradient(w_new,x[r,:],t[r])        

                    # get difference of values   
                    dw = w_new - (w + mu_t * z) 
                    # get difference of gradients 
                    dg = g_new - g_naq - (self.delta * dw)
                    #print('--dg',dg) 

                    _, _, B_new = self.Update_RES_BFGS(B, dw, dg, I)

                    # update step
                    B = B_new
                    w = w_new
                    z = z_new
                    
                    #print("----w: ",w)
                    #print("----k: ",k)
                
                else:
                    mu_t = self.mu 
                    print('----mu:', mu_t)
                    mu_.append(mu_t)

                    # compute nesterov'gradient 
                    _, _, g_naq = self.gradient(w+(mu_t*z),x[r,:],t[r])
                
                    #if np.linalg.norm(g_naq)==0: break 
                  
                    # update hessian matrix
                    H = np.linalg.inv(B)
                    H_RES = H + self.gamma*I
                  
                    # step size
                    #self.eta = ((1e-3) * (4*10**8))/((4*10**8)+k)
                    eta =((self.epsilon_0) * (self.tau_0))/((self.tau_0)+k)


                    # compute nesterov'direction
                    z_new = mu_t*z - eta * np.matmul(H_RES,g_naq)

                    # update weight
                    w_new = w + z_new
                
                    # compute new gradient 
                    _, _, g_new = self.gradient(w_new,x[r,:],t[r])        

                    # get difference of values   
                    dw = w_new - (w + mu_t * z) 
                    # get difference of gradients 
                    dg = g_new - g_naq - (self.delta * dw)
                    #print('--dg',dg) 

                    _, _, B_new = self.Update_RES_BFGS(B, dw, dg, I)

                    # update step
                    B = B_new
                    w = w_new
                    z = z_new
                    
                    #print("----w: ",w)
                    #print("----k: ",k)

                # cpu time used
                stop_time = time.perf_counter()
                time_resnaq[i] = stop_time - start_time
                 
        
        self.final_iter = k
        self._coef = w
        self.obj_resnaq = obj_resnaq
        self.iter_resnaq = iter_resnaq
        self.objnat_resnaq = objnat_resnaq
        self.mu_ = mu_
        self.time_resnaq = time_resnaq

        return self

    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef))
        p[p==0] = 1
        return p.astype(int)