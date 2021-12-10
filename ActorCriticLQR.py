"""
Actor-critic method to solve LQR
@author: mo zhou
"""
import json
import os
from scipy.stats import multivariate_normal as normal
from scipy.linalg import solve_discrete_are as dscrt_rct
from absl import app
from absl import flags
import numpy as np
# import matplotlib.pyplot as plt

flags.DEFINE_string('config_path', 'cfg2.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', None,
                    """The name of numerical experiments, prefix for logging""")
FLAGS = flags.FLAGS

def main(argv):
    del argv
    if FLAGS.exp_name is None: # use config name as exp_name
        FLAGS.exp_name = os.path.splitext(os.path.basename(FLAGS.config_path))[0]
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)
    # config = munch.munchify(config)
    A = np.array(config['A'])
    B = np.array(config['B'])
    Q = np.array(config['Q'])
    R = np.array(config['R'])
    D_xi = np.array(config['D_xi'])
    sigma = config['sigma']
    alpha = config['alpha']
    beta = config['beta']
    T = config['T']
    N = config['N']
    N0 = config['N0']
    N1 = config['N1']
    log_freq = config['log_freq']
    D_ep = D_xi + sigma**2 * np.matmul(B, np.transpose(B))
    d, k = np.shape(B)
    
    # initialization
    theta = np.zeros([d+k, d+k])
    K = np.zeros([k, d])
    train_history = []
    
    def solve(init, A_BK, d):
        # matrix equation: ans = init + A_BK * ans * A_BK^T, d is the dim
        tens = np.kron(A_BK, A_BK)
        ans_vec = np.linalg.solve(np.identity(d**2)-tens, np.matrix.flatten(init))
        return np.reshape(ans_vec, [d,d])
    
    D_ep = D_xi + sigma**2 * np.matmul(B, np.transpose(B))
    P_K_star = dscrt_rct(A,B,Q,R)
    K_star = np.linalg.solve(R+np.linalg.multi_dot([np.transpose(B),P_K_star,B]), np.linalg.multi_dot([np.transpose(B),P_K_star,A]))
    J_K_star = np.trace(np.matmul(D_ep, P_K_star))
    
    max_norm = [0, 0, 0, 0] # record the max norm, for assump 1
    
    # training process
    for t in range(T+1):
        A_BK = A - np.matmul(B,K) # dxd
        # compute and record the errors 
        if t % log_freq == 0:
            P_K = solve(Q+np.linalg.multi_dot([np.transpose(K),R,K]) , np.transpose(A_BK), d)
            # D_K = solve(D_ep , A_BK, d)
            temp1 = np.concatenate((Q+np.linalg.multi_dot([np.transpose(A),P_K,A]), np.linalg.multi_dot([np.transpose(A),P_K,B])), axis = 1)
            temp2 = np.concatenate((np.linalg.multi_dot([np.transpose(B),P_K,A]), R+ np.linalg.multi_dot([np.transpose(B),P_K,B])), axis = 1)
            theta_K = np.concatenate((temp1, temp2), axis=0)
            rel_err_theta = np.linalg.norm(theta - theta_K)**2
            J_K = np.trace(np.matmul(D_ep, P_K))
            err_actor = J_K - J_K_star
            err_K = np.linalg.norm(K - K_star)
            train_history.append([t, rel_err_theta, err_actor, err_K])
            print("errors", rel_err_theta, err_actor, err_K)
        
        # compute the max norms in order to check assumption 1
        norm_A_BK = np.linalg.norm(A_BK, ord=2)
        E = np.matmul(np.concatenate((np.identity(d),-K), axis=0), np.concatenate((A,B), axis=1))
        norm_E = np.linalg.norm(E, ord=2)
        norm_K = np.linalg.norm(K)
        norm_theta = np.linalg.norm(theta)
        max_norm = np.maximum(max_norm, [norm_A_BK, norm_E, norm_K, norm_theta])
        
        # critic
        # observe N0 trajectories
        x_N0 = np.zeros([N, d]) #Nxd
        for _ in range(N0):
            x_N0 = np.tensordot(x_N0, A_BK, axes=([1],[1]))
            x_N0 += np.reshape(normal.rvs(mean=np.zeros([d]), cov=D_ep, size=(N)), [N,d])
        u_N0 = - np.tensordot(x_N0, K, axes=([1],[1])) + sigma**2 * np.reshape(normal.rvs(size=[N,k]), [N,k]) #Nxk
        xu_N0 = np.concatenate((x_N0, u_N0), axis=1)
        
        # observe hat_psi_j
        x_prime = np.tile(np.tensordot(x_N0, A, axes=([1],[1])), (N1,1,1)) #N1xNxd
        x_prime += np.tile(np.tensordot(u_N0, B, axes=([1],[1])), (N1,1,1)) #N1xNxd
        x_prime += np.reshape(normal.rvs(mean=np.zeros([d]), cov=D_xi, size=(N1, N)), [N1,N,d]) #N1xNxd
        u_prime = - np.tensordot(x_prime, K, axes=([2],[1])) + sigma**2 * np.reshape(normal.rvs(size=[N1, N, k]), [N1, N, k])#N1xNxk
        xu_prime = np.concatenate((x_prime, u_prime), axis=2) #N1xNx(d+k)
        phi_N0 = np.reshape(xu_N0, [N,d+k,1]) * np.reshape(xu_N0, [N,1,d+k])#Nx(d+k)x(d+k)
        psi = np.reshape(xu_prime, [N1,N,d+k,1]) * np.reshape(xu_prime, [N1,N,1,d+k]) - np.reshape(phi_N0, [1,N,d+k,d+k]) #N1xNx(d+k)x(d+k)
        
        # compute critic gradient
        bar_psi = np.mean(psi, axis=0, keepdims=True) #1xNx(d+k)x(d+k)
        c_N0 = np.sum(np.matmul(x_N0, Q) * x_N0, axis=1, keepdims = True)\
            + np.sum(np.matmul(u_N0, R) * u_N0, axis=1, keepdims = True) #Nx1
        f1 = np.mean(psi * np.reshape(c_N0, [1,N,1,1]), axis=0) 
        f2 = np.mean(np.reshape(np.tensordot(psi, theta, axes=([2,3],[0,1])), [N1,N,1,1])* psi, axis=0)
        f3 = np.sum(np.reshape(np.tensordot(psi-bar_psi, theta, axes=([2,3],[0,1])), [N1,N,1,1])* (psi-bar_psi), axis=0) / (N1-1)
        critic_grad = np.mean(f1+f2-f3, axis=0)
        
        # actor update
        natural_grad = np.matmul(theta[d:,:], np.concatenate((-np.identity(d),K), axis=0))
        K -= beta * natural_grad
        
        # critic update
        theta -= alpha * critic_grad
    
    train_history.append(max_norm)
    # plt.plot(np.log10(train_history[:,1]))
    # plt.plot(np.log10(train_history[:,2]))
    print('max norm', max_norm)
    np.savetxt('{}.csv'.format(FLAGS.config_path),
                train_history,
                fmt=['%.5e', '%.5e', '%.5e', '%.5e'],
                delimiter=",",
                header='step, err_critic, err_actor, err_K',
                )
    
if __name__ == '__main__':
    app.run(main)