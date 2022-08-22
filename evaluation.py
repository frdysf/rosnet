#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging

from network import Network
from L2_eval_bounds import L2_eval_bounds


def _init_logger():
    'Private method, returns custom setup for logger.'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(name)s: %(funcName)s: %(message)s')
    
    # TO-DO: ideally wipe evaluation.log first, if it exists

    file_handler = logging.FileHandler('evaluation.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    return logger

class Evaluation:
    
    def __init__(self, x, nf, NNo, nr, NNr):
        '''
        Initialise reduced-order synthesis problem with variables and parameters.
        
        Parameters
        ----------
        x: np.ndarray
            Input data.
                
        nf: int
            (Maximum) dimension of full-order network's hidden layers.
                
        NNo: Network (network.py)
            Full-order network.
            
        nr: int
            (Maximum) dimension of pruned (or reduced-order) network's hidden layers.
                
        NNr: Network (network.py)
            Pruned or otherwise reduced-order network to be evaluated. Originates from full-order network.
        '''
        self.logger = _init_logger() # create logger for debug messages
        
        self.x = np.atleast_2d(x)
        
        if not all(isinstance(n, int) for n in [nf, nr]):
            raise TypeError('nf and nr must be integers. Use help() for more info.')
        else:
            self.nf = nf
            self.nr = nr
        
        if not all(isinstance(N, Network) for N in [NNo, NNr]):
            raise TypeError('NNo and NNr must be Network objects. See network.py.')
        else:
            self.NNo = NNo
            self.NNr = NNr
        
        self.Nsamples = self.x.shape[1]
        
        self.z = np.zeros(self.Nsamples) 
        self.z_r = np.zeros(self.Nsamples)
        
        self.gamma = 0
        self.eps = 0
        
        self.error = np.zeros(self.Nsamples)
        self.error_sq = np.zeros(self.Nsamples)
        self.error_bound = np.zeros(self.Nsamples)
        self.diff = np.zeros(self.Nsamples)
        
    def output(self, func1, func2):
        '''
        Generates and stores output of NNo.
        Current default is feedforward mapping. (See funcmap() in network.py.)
        
        Parameters
        ----------
        func1: method
            Activation function applied to hidden layer preceding output layer, e.g. ReLU, sigmoid.
            
        func2: method
            Activation function applied to output layer, e.g. ReLU, sigmoid.
        '''
        try:
            self.z = self.NNo.funcmap(self.x, func1, func2)
            
        except Exception as e:
            self.logger.error('Failed to generate output from NNo. Error raised: {0}'.format(e))
            raise RuntimeError('Failed to generate output from NNo. Check log for details.')
            
        self.logger.info('Output generated from NNo.')
        
    def eval_L2(self, c1, c2, func1, func2):
        '''
        Runs L2_eval_bounds and generates error bounds for NNr. Also calculates various error metrics.
        
        Parameters
        ----------
        eps: float
            Constant term in error bound: ||e||^2 = gamma*||x||^2 + eps.
            
        func1: method
            Activation function applied to hidden layer preceding output layer, e.g. ReLU, sigmoid.
            
        func2: method
            Activation function applied to output layer, e.g. ReLU, sigmoid.
        '''
        
        if not self.z.any():
            raise RuntimeError('Missing output of full-order network. Run output() first.')
        
        if len(self.NNo.NN) != 4 or len(self.NNr.NN) != 4:
            raise RuntimeError('NNo and NNr must each have only ONE hidden layer to run eval_L2(). NNo has {0} implicit layers. NNr has {1} implicit layers.'.format(len(self.NNo.NN)/2 - 1, len(self.NNr.NN)/2 - 1))
        
        else:
            
            gamv, epsv, status = L2_eval_bounds(self.NNo, self.NNr, self.nr, c1, c2)

            if status not in ['optimal']:
                self.logger.warning('Solution status returns {1}'.format(status))

            self.gamma = gamv
            self.eps = epsv
                               
            try:
                self.z_r = self.NNr.funcmap(self.x, func1, func2)
            
            except Exception as e:
                self.logger.error('Failed to generate output from NNr. Error raised: {0}'.format(e))
                raise RuntimeError('Failed to generate output from NNr. Check log for details.')
            
            self.logger.info('Output generated from NNr.')

            for i in range(self.Nsamples):
                self.error[i] = np.linalg.norm(self.z[i] - self.z_r[i], ord=np.inf)
                self.error_sq[i] = self.error[i]**2
                self.error_bound[i] = self.gamma*(np.linalg.norm(self.x[:,i], ord=2)**2) + self.eps
                self.diff[i] = self.error_bound[i] - self.error_sq[i]
                
    def plot(self, size=(10, 7.5), line=True):
        '''
        Plots a) outputs of original and pruned (or reduced) NNs, b) all error bounds against input x. Thin wrapper around matplotlib.pyplot.
        Use for nx = 1, i.e. unidimensional input data x.
        
        Parameters
        ----------
        size: tuple, optional
            Specifies figsize for both figures/plots.
            
        line: bool, optional
            Line (True) or scatter (False) plot. Default True.
        '''
        
        if len(size) != 2:
            raise ValueError('size should be a tuple of length 2.')
            
        if not all(isinstance(i, (int, float)) for i in size):
                raise ValueError('size must contain int or float only.')
        
        # a) plot outputs of full and pruned (or reduced) NNs
        plt.figure('outputs', figsize=size)
        
        if line:
            plt.plot(self.x.T, self.z, label=r'$f(x)$ - original network')
            plt.plot(self.x.T, self.z_r, label=r'$g(x)$ - pruned (or reduced-order) network')
            
        else:
            plt.plot(self.x.T, self.z, marker='o', label=r'$f(x)$ - original network')
            plt.plot(self.x.T, self.z_r, marker='x', label=r'$g(x)$ - pruned (or reduced-order) network')
        
        plt.legend()
        plt.grid(linewidth=0.5)
        plt.xlabel(r'input $x$')
        plt.ylabel(r'neural network output, $z$')
        plt.title('Output of original vs. pruned neural networks [$f(x)$, $g(x)$]')
        plt.show()
        
        # b) plot error bounds with actual squared error for the pruned (or reduced) NN, across its output
        plt.figure('bounds', figsize=size)
        
        if line:
            plt.semilogy(self.x.T, self.error_bound, 'b', label=r'$\gamma\Vert x\Vert_{2}^{2} + \epsilon$') # log scaling on y axis
            plt.semilogy(self.x.T, self.error_sq, 'k', label=r'$\Vert f(x) - g(x)\Vert_{2}^{2}$') # log scaling on y axis
        
        else:
            plt.semilogy(self.x.T, self.error_bound, 'bx', label=r'$\gamma\Vert x\Vert_{2}^{2} + \epsilon$') # log scaling on y axis
            plt.semilogy(self.x.T, self.error_sq, 'kx', label=r'$\Vert f(x) - g(x)\Vert_{2}^{2}$') # log scaling on y axis
        
        plt.legend()
        plt.grid(which='both', linewidth=0.5)
        plt.xlabel(r'input $x$')
        plt.ylabel(r'approximation error')
        plt.title(r'Error bound vs. actual squared error for input $x$')
        plt.show()
        
    def multiplot(self, size=(10, 7.5), line=False):
        '''
        Plots a) outputs of original and pruned (or reduced) NNs, b) all error bounds against input x. Thin wrapper around matplotlib.pyplot.
        Use for nx > 1, i.e. multidimensional input data x. Uses PCA to represent input with principal component x_1.
        
        Parameters
        ----------
        size: tuple, optional
            Specifies figsize for both figures/plots.
            
        line: bool, optional
            Line (True) or scatter (False) plot. Default False.
        '''
        
        if len(size) != 2:
            raise ValueError('size should be a tuple of length 2.')
            
        if not all(isinstance(i, (int, float)) for i in size):
                raise ValueError('size must contain int or float only.')
        
        # a) plot outputs of full and pruned (or reduced) NNs
        plt.figure('outputs', figsize=size)
        
        # perform PCA on input x, reducing to 2 components
        pca = PCA(n_components=2)
        pca.fit(self.x.T)
        x_pca = pca.transform(self.x.T) # column indexing of new components x_i
        
        print(u'Explained variance ratio, n_components = 2 (x\u2081, x\u2082): {0}'.format(pca.explained_variance_ratio_))
        
        # if output z is multidimensional (nz > 1), perform PCA
        if self.z.shape[1] > 1:
            pca = PCA(n_components=2)
            pca.fit(self.z)
            
            z_pca = pca.transform(self.z)
            z_r_pca = pca.transform(self.z_r)
        
            print(u'Explained variance ratio, n_components = 2 (z\u2081, z\u2082): {0}'.format(pca.explained_variance_ratio_))
        
        if line:
            if self.z.shape[1] > 1:
                plt.plot(x_pca[:,0], z_pca[:,0], label=r'$f(x)$ - original network')
                plt.plot(x_pca[:,0], z_r_pca[:,0], label=r'$g(x)$ - pruned (or reduced-order) network')
                plt.ylabel(r'principal component $z_1$ from PCA on input $z$, where nz > 1')
            else:
                plt.plot(x_pca[:,0], self.z, label=r'$f(x)$ - original network')
                plt.plot(x_pca[:,0], self.z_r, label=r'$g(x)$ - pruned (or reduced-order) network')
                plt.ylabel(r'neural network output, $z$')
        else:
            if self.z.shape[1] > 1:
                plt.plot(x_pca[:,0], z_pca[:,0], 'o', label=r'$f(x)$ - original network')
                plt.plot(x_pca[:,0], z_r_pca[:,0], 'x', label=r'$g(x)$ - pruned (or reduced-order) network')
                plt.ylabel(r'principal component $z_1$ from PCA on input $z$, where nz > 1')
            else:
                plt.plot(x_pca[:,0], self.z, 'o', label=r'$f(x)$ - original network')
                plt.plot(x_pca[:,0], self.z_r, 'x', label=r'$g(x)$ - pruned (or reduced-order) network')
                plt.ylabel(r'neural network output, $z$')
        
        plt.legend()
        plt.grid(linewidth=0.5)
        plt.xlabel(r'principal component $x_1$ from PCA on input $x$, where nx > 1')
        plt.title(r'Output of original vs. pruned neural networks [$f(x)$, $g(x)$]')
        plt.show()
        
        # b) plot error bounds with actual squared error for the pruned (or reduced) NN, across its output
        plt.figure('bounds', figsize=size)
        
        if line:
            plt.semilogy(x_pca[:,0], self.error_bound, 'b', label=r'$\gamma\Vert x\Vert_{2}^{2} + \epsilon$') # log scaling on y axis
            plt.semilogy(x_pca[:,0], self.error_sq, 'k', label=r'$\Vert f(x) - g(x)\Vert_{2}^{2}$') # log scaling on y axis
            
        else:
            plt.semilogy(x_pca[:,0], self.error_bound, 'bx', label=r'$\gamma\Vert x\Vert_{2}^{2} + \epsilon$') # log scaling on y axis
            plt.semilogy(x_pca[:,0], self.error_sq, 'kx', label=r'$\Vert f(x) - g(x)\Vert_{2}^{2}$') # log scaling on y axis
        
        plt.legend()
        plt.grid(which='both', linewidth=0.5)
        plt.xlabel(r'input $x$')
        plt.ylabel(r'approximation error')
        plt.title(r'Error bound vs. actual squared error for input $x$')
        plt.show()
        
        
    def multiplot3D(self, size=(10, 7.5)):
        '''
        Plots a) outputs of original and pruned (or reduced) NNs, b) all error bounds against input x. Thin wrapper around matplotlib.pyplot.
        Use for nx > 1, i.e. multidimensional input data x. Uses PCA to represent input with principal components x_1, x_2.
        
        Parameters
        ----------
        size: tuple, optional
            Specifies figsize for both figures/plots.
        '''
        
        if len(size) != 2:
            raise ValueError('size should be a tuple of length 2.')
            
        if not all(isinstance(i, (int, float)) for i in size):
                raise ValueError('size must contain int or float only.')
        
        # a) plot outputs of full and reduced NNs
        fig = plt.figure('outputs', figsize=size)
        ax = fig.add_subplot(projection='3d')
        
        # perform PCA on input x, reducing to 2 components
        pca = PCA(n_components=2)
        pca.fit(self.x.T)
        x_pca = pca.transform(self.x.T) # column indexing of new components
        
        print(u'Explained variance ratio, n_components = 2 (x\u2081, x\u2082): {0}'.format(pca.explained_variance_ratio_))
        
        # if output z is multidimensional (nz > 1), perform PCA
        if self.z.shape[1] > 1:
            pca = PCA(n_components=2)
            pca.fit(self.z)
            
            z_pca = pca.transform(self.z)
            z_r_pca = pca.transform(self.z_r)
        
            print(u'Explained variance ratio, n_components = 2 (z\u2081, z\u2082): {0}'.format(pca.explained_variance_ratio_))
            
            ax.scatter(x_pca[:,0], x_pca[:,1], z_pca[:,0], marker='o', label=r'$f(x)$ - original network')
            ax.scatter(x_pca[:,0], x_pca[:,1], z_r_pca[:,0], marker='x', label=r'$g(x)$ - pruned or (reduced-order) network')
            ax.set(zlabel=r'principal component $z_1$ from PCA on input $z$, where nz > 1')
            
        else:
            ax.scatter(x_pca[:,0], x_pca[:,1], self.z, marker='o', label=r'$f(x)$ - original network')
            ax.scatter(x_pca[:,0], x_pca[:,1], self.z_r, marker='x', label=r'$g(x)$ - pruned or (reduced-order) network')
            ax.set(zlabel=r'neural network output, $z$')
        
        plt.title(r'Output of original vs. pruned neural networks [$f(x)$, $g(x)$]')
        ax.legend()
        ax.grid(linewidth=0.5)
        ax.set(xlabel=r'principal component $x_1$ from PCA on input $x$, where nx > 1',
               ylabel=r'principal component $x_2$')
        plt.show()
        
        # b) plot error bounds with actual squared error for the pruned (or reduced) NN, across its output
        plt.figure('bounds', figsize=size)
        
        plt.semilogy(x_pca[:,0], self.error_bound, 'bx', label=r'$\gamma\Vert x\Vert_{2}^{2} + \epsilon$') # log scaling on y axis
        plt.semilogy(x_pca[:,0], self.error_sq, 'kx', label=r'$\Vert f(x) - g(x)\Vert_{2}^{2}$') # log scaling on y axis
        
        plt.legend()
        plt.grid(which='both', linewidth=0.5)
        plt.xlabel(r'input $x$')
        plt.ylabel(r'approximation error')
        plt.title(r'Error bound vs. actual squared error for input $x$')
        plt.show()


def main_math():
    x = np.linspace(-10, 10, 100)

    weights = np.array([np.array([-0.634700665707495, 1.656731026589558, 1.124487323633357, -0.795741403939511, -1.168109939313002, 0.869874974297699, -1.008855879224341, 1.914651700028183, 1.240317284056857, -1.420252553244148]).reshape(10, 1),
                       np.array([-0.258472107285207, -0.680982998070259, 1.560623889702786, -1.220971595926017, -0.027188941462718, -1.010335979063966, 0.835793863154799, 1.222569086520831, 0.259056187128039, 0.245551285836084]).reshape(10, 1),
                       np.array([0.082854200378300, 0.992706291627412, -1.256284879106125, 0.554918709010267, 0.317351602368993, 1.250828758360099, 1.679813903514902, -0.287094733070243, 1.701671362974561, -0.829936846466298]),
                       np.array([1.864581839050232])], dtype='object')

    NNo = Network()
    NNo.load(2, weights)

    reduced_weights = np.array([np.array([0, 1.656731026589558, 1.124487323633357, 0, -1.168109939313002, 0.869874974297699, -1.008855879224341, 1.914651700028183, 1.240317284056857, -1.420252553244148]).reshape(10, 1),
                       np.array([-0.258472107285207, -0.680982998070259, 1.560623889702786, -1.220971595926017, -0.027188941462718, -1.010335979063966, 0.835793863154799, 1.222569086520831, 0.259056187128039, 0.245551285836084]).reshape(10, 1),
                       np.array([0.082854200378300, 0.992706291627412, -1.256284879106125, 0.554918709010267, 0, 1.250828758360099, 1.679813903514902, 0, 1.701671362974561, -0.829936846466298]),
                       np.array([1.864581839050232])], dtype='object')

    NNr = Network()
    NNr.load(2, reduced_weights)

    prob = Evaluation(x, 10, NNo, 10, NNr)
    prob.output(activations.relu, activations.null)

    c1, c2 = 1.0, 0.0005  # choice of c1, c2 seem to affect the correctness of bounds (if c2 is too large, eps is not sufficiently large)
    prob.eval_L2(c1, c2, activations.relu, activations.null)
    prob.plot()

def main_diabetes():
    x = np.array([[6,148,72,35,0,33.6,0.627,50],  # original train/test data from diabetes.csv
                  [1,85,66,29,0,26.6,0.351,31],
                  [8,183,64,0,0,23.3,0.672,32],
                  [1,89,66,23,94,28.1,0.167,21],
                  [0,137,40,35,168,43.1,2.288,33],
                  [5,116,74,0,0,25.6,0.201,30],
                  [3,78,50,32,88,31,0.248,26],
                  [10,115,0,0,0,35.3,0.134,29],
                  [2,197,70,45,543,30.5,0.158,53],
                  [8,125,96,0,0,0,0.232,54]])

    x = x.T # column indexing of input data

    W0 = np.array([[0.319330543,-0.064024068,-0.12958169,0.053546514,-0.16652295,0.424845994,0.281637579,0.463299185,0.41607216,-0.362394065,0.117396057,-0.115828782,0.02417052,-0.093473181,-0.033018291,0.340984523,-0.430841088,0.282699287,-0.174354658,-0.327419668,-0.425565124,0.089084774,0.209613413,0.215300351,0.029289126],
                  [0.060385201,-0.307317913,0.290631652,0.164571777,0.043761898,-0.244602799,-0.179488897,-0.231704101,-0.042899083,-0.384876609,-0.038089514,0.308912098,-0.256682724,0.381159484,-0.335326999,-0.04961988,-0.416980743,-0.308168411,0.012801125,0.16613692,0.351306528,0.036453165,0.306259722,-0.21805951,-0.015378118],
                  [-0.079051293,-0.127091154,0.096157998,-0.207436576,-0.18794015,-0.165974349,0.319373965,-0.199821487,-0.238665953,0.286627412,-0.300753355,-0.293435097,0.222713411,0.13364692,-0.322380036,0.056780335,0.013730234,0.111230694,0.116473801,-0.233298987,-0.394313961,0.251653463,0.17598027,0.00401052,-0.061955899],
                  [0.03535784,0.01237071,0.042771801,-0.416875362,-0.398908228,0.181106031,0.142381296,-0.120942213,0.079152577,-0.16076991,-0.102625996,0.164235398,-0.220229328,-0.016303798,-0.108958811,-0.051054373,0.27026701,0.381220609,-0.17959097,-0.112234637,-0.009297519,-0.05388635,0.366193682,0.201444998,-0.351042628],
                  [0.326576203,0.386269867,-0.337754548,-0.294614315,0.053369574,-0.377952576,-0.380502552,0.383208185,0.340806603,0.075724252,-0.079203039,-0.234530643,0.057954129,-0.404509962,-0.079749286,-0.228150129,0.136709854,-0.234567031,0.092377476,0.337934583,0.019169254,0.329208672,-0.258823991,-0.372488737,-0.069464862],
                  [0.004284321,0.121035874,0.245985344,0.307337523,-0.078217342,0.335836798,0.241838291,0.123912677,-0.058150098,0.031053482,0.242122054,0.416289151,-0.179632962,0.086220279,-0.269981295,-0.105384499,-0.218318611,0.073712669,-0.06532003,-0.23324953,0.242176294,0.158564076,0.453506917,0.242155299,-0.099511266],
                  [-0.218930051,-0.250733525,0.141308591,0.587304473,-0.194021061,-0.299740225,-0.509302795,0.355344862,0.224369243,-0.102556191,-0.255802184,0.102736801,0.357320815,-0.582347035,-0.32347554,0.08099708,0.273869932,0.087297425,-0.236969888,0.489243299,-0.347979486,-0.430328786,0.127037898,0.257815361,-0.223205581],
                  [-0.066540658,-0.463751853,0.067296103,-0.196394593,0.181782559,0.127613172,0.292350173,-0.198134363,0.101130165,-0.140528634,-0.212750256,-0.392510474,0.114727616,-0.076889455,-0.013871998,0.209182099,0.061868381,-0.282851577,-0.278421909,-0.268992513,-0.418144166,0.252622873,-0.105547152,-0.091992907,0.006925523]])

    b0 = np.array([0.385025948, 0.203247383, -0.435936451, -0.384230286, 0.26093635, 0.065291144, 0.263380855, -0.213543728, -0.364733875, 0.010182882, 0, -0.384915113, -0.006400483, 0.418469876, 0, -0.249839008, -0.004327881, 0.068022244, 0.320145249, -0.369324774, 0.532851279, 0.400793076, -0.434162915, 0.061749421, 0]).reshape(-1, 1)

    W1 = np.array([-0.05639274, -0.212818667, 0.118844204, 0.177333325, -0.437093645, 0.444800556, -0.241538703, 0.226674169, 0.04753435, -0.323705137, -0.025351912, 0.062733993, 0.054641746, -0.099165536, 0.437024713, 0.361994803, -0.438521206, 0.266100854, -0.359393835, 0.333090693, -0.247340187, -0.134647533, 0.133005962, 0.389749646, -0.27149421]).reshape(-1, 1)

    b1 = np.array([-0.42961809])

    weights = np.array([W0.T, b0, W1.T, b1], dtype=object)

    NNo = Network()
    NNo.load(2, weights)

    W0r = np.array([[ 0.32898635,  0.        , -0.08142337,  0.        ,  0.1121863 ,
             0.60852474,  0.3379216 ,  0.39201936,  0.40094614, -0.54057175,
             0.11139488, -0.11782964, -0.        , -0.0995153 , -0.        ,
             0.26683232, -0.28083757,  0.4532513 , -0.09626577, -0.2850118 ,
            -0.3569746 ,  0.10892044,  0.19673488,  0.3902965 ,  0.        ],
           [-0.        , -0.2986591 ,  0.25856525,  0.15550525, -0.        ,
            -0.26861182, -0.13649717, -0.22266772,  0.        , -0.39553842,
            -0.        ,  0.31357753, -0.25348628,  0.36753556, -0.335327  ,
             0.        , -0.4010823 , -0.3081684 , -0.        ,  0.14210199,
             0.37991273,  0.        ,  0.29233763, -0.28646043, -0.        ],
           [-0.        , -0.10798831,  0.08719976, -0.16818489, -0.17862326,
            -0.17081957,  0.3262849 , -0.19682045, -0.23920175,  0.28086284,
            -0.30075336, -0.29454187,  0.21563998,  0.12470833, -0.32238004,
             0.        ,  0.        ,  0.10662506,  0.16237175, -0.2385796 ,
            -0.3777177 ,  0.24389295,  0.18588956,  0.        , -0.        ],
           [-0.        ,  0.        , -0.        , -0.27959895, -0.3902605 ,
             0.14527294,  0.16392347, -0.10645112,  0.        , -0.13391997,
            -0.102626  ,  0.16864859, -0.23723695, -0.        , -0.10895881,
            -0.        ,  0.3348577 ,  0.34807396, -0.13386628, -0.11527086,
             0.        ,  0.        ,  0.35757342,  0.16478802, -0.35104263],
           [ 0.3235537 ,  0.41553122, -0.30847457, -0.23481533, -0.        ,
            -0.37795258, -0.32663527,  0.36185983,  0.3424986 ,  0.        ,
            -0.        , -0.22981061,  0.        , -0.42603546, -0.        ,
            -0.24438477,  0.13016897, -0.25027937,  0.        ,  0.3314496 ,
             0.        ,  0.3303226 , -0.23886968, -0.3950496 , -0.        ],
           [-0.        ,  0.06342336,  0.272285  ,  0.39878523, -0.        ,
             0.34100625,  0.21731974,  0.19963197,  0.        , -0.        ,
             0.23611757,  0.4215798 , -0.19168402, -0.        , -0.2699813 ,
            -0.11854812, -0.16515572,  0.        , -0.        , -0.18724896,
             0.18697129,  0.10471443,  0.5190722 ,  0.24574484, -0.09951127],
           [-0.21763997, -0.1829633 ,  0.20501639,  0.60647094, -0.23018838,
            -0.31155312, -0.660749  ,  0.25876597,  0.21570045, -0.10523744,
            -0.26172638,  0.11273324,  0.34704182, -0.5969039 , -0.32347554,
             0.        ,  0.46998742,  0.        , -0.37650624,  0.38356563,
            -0.30602098, -0.4305604 ,  0.17119661,  0.24580888, -0.22320558],
           [-0.        , -0.39965746,  0.        , -0.2754033 ,  0.24333766,
             0.16418533,  0.33775812, -0.23537464,  0.        , -0.18854412,
            -0.21875447, -0.3945756 ,  0.12928835, -0.        , -0.        ,
             0.1605258 ,  0.        , -0.24897116, -0.23508275, -0.253873  ,
            -0.38179648,  0.25479475, -0.12460133, -0.05844583,  0.        ]])

    b0r = np.array([[ 0.40143678,  0.35796377, -0.709049  , -0.6795594 ,  0.44224095,
           -0.00141956,  0.507188  , -0.3492456 , -0.3770488 ,  0.06582427,
           -0.00595833, -0.40203238, -0.01463025,  0.42690778,  0.        ,
           -0.4981922 ,  0.06264249,  0.00657516,  0.70461965, -0.6325912 ,
            0.86853176,  0.69590694, -0.7374417 , -0.00447432,  0.        ]])

    W1r = np.array([[-0.        ],
           [-0.23543309],
           [ 0.07786088],
           [ 0.24573803],
           [-0.47365907],
           [ 0.48040658],
           [-0.23243384],
           [ 0.18790376],
           [-0.        ],
           [-0.35482234],
           [ 0.        ],
           [ 0.        ],
           [ 0.        ],
           [ 0.        ],
           [ 0.4370247 ],
           [ 0.2861908 ],
           [-0.34721765],
           [ 0.22663319],
           [-0.3680076 ],
           [ 0.30601966],
           [-0.25001344],
           [-0.1338576 ],
           [ 0.13985643],
           [ 0.38377875],
           [-0.2714942 ]])

    b1r = np.array([-0.73059076])

    reduced_weights = np.array([W0r.T, b0r.T, W1r.T, b1r], dtype=object)

    NNr = Network()
    NNr.load(2, reduced_weights)

    prob = Evaluation(x, 25, NNo, 25, NNr)
    prob.output(activations.relu, activations.sigmoid)

    c1, c2 = 1.0, 0.0005  # choice of c1, c2 seem to affect the correctness of bounds (if c2 is too large, eps is not sufficiently large)
    prob.eval_L2(c1, c2, activations.relu, activations.sigmoid)

    prob.multiplot()
    prob.multiplot3D()

if __name__ == "__main__":
    import activations
    main_math()
    main_diabetes()