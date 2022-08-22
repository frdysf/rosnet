#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging

from network import Network
from L2_min_sector import L2_min_sector


def _init_logger():
    'Private method, returns custom setup for logger.'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(name)s: %(funcName)s: %(message)s')
    
    # TO-DO: ideally wipe synthesis.log first, if it exists

    file_handler = logging.FileHandler('synthesis.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    return logger

class Synthesis:
    
    def __init__(self, x, nf, NNo, m1_range):
        '''
        Initialise reduced-order synthesis problem with variables and parameters.
        
        Parameters
        ----------
        x: np.ndarray
            Input data.
                
        nf: int
            (Maximum) dimension of full-order network's hidden layers. 
                
        NNo: Network (network.py)
            Full-order network to be approximated.
                
        m1_range: sequence
            Integer parameter values for dimension of hidden layers in reduced-order networks. A reduced-order network is synthesised for each value.
        '''
        self.logger = _init_logger() # create logger for debug messages
        
        self.x = np.atleast_2d(x)
        
        if not isinstance(nf, int):
            raise TypeError('nf must be an integer. Use help() for more info.')
        else:
            self.nf = nf
        
        if not isinstance(NNo, Network):
            raise TypeError('NNo must be a Network object. See network.py.')
        else:
            self.NNo = NNo
        
        if not all((isinstance(i, int)) for i in m1_range):
            raise TypeError('Values in m1_range must be integers only. Use help() for more info.')        
        else:
            self.m1_range = m1_range
        
        self.Nsamples = self.x.shape[1]
        self.Nm1 = len(self.m1_range)
        
        self.nz = self.NNo.NN['b1'].shape[0]
        self.z = np.zeros(self.Nsamples) 
        self.z_r = np.zeros((self.Nm1, self.Nsamples, self.nz))
        
        self.gamma = np.zeros(self.Nm1)
        self.eps = np.zeros(self.Nm1)
        
        self.error = np.zeros((self.Nm1, self.Nsamples))
        self.error_sq = np.zeros((self.Nm1, self.Nsamples))
        self.error_bound = np.zeros((self.Nm1, self.Nsamples))
        self.diff = np.zeros((self.Nm1, self.Nsamples))
        
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
        
    def reduce_L2(self, c1, c2, func1, func2):
        '''
        Runs L2_min_sector for each value m1 in m1_range and generates NNr output. Also calculates error & error bounds for each NNr.
        
        L2_min_sector for reduced-order synthesis where l = 1 ONLY (i.e. single hidden layer) for NNo, NNr. Only verified on ReLU activation.
        
        Parameters
        ----------
        c1, c2: int/float
            Scalar factors in objective function: min c1*gamv + c2*epsv.
            
        func1: method
            Activation function applied to hidden layer preceding output layer, e.g. ReLU, sigmoid.
            
        func2: method
            Activation function applied to output layer, e.g. ReLU, sigmoid.
        '''
        
        if not self.z.any():
            raise RuntimeError('Missing output of full-order network. Run output() first.')
        
        if len(self.NNo.NN) != 4:
            raise RuntimeError('NNo must have only ONE hidden layer to run reduce_L2(). NNo has {0} implicit layers.'.format(len(self.NNo.NN)/2 - 1))
        
        else:
            
            self.NNrs = dict.fromkeys(self.m1_range)
            
            j = 0  # independent iterator over m1_range
            
            for g in self.m1_range:
                NNr, gamv, epsv, status = L2_min_sector(self.NNo, g, c1, c2)
                
                if status not in ['optimal']:
                    self.logger.warning('Solution status for g = {0} (n_l) returns {1}'.format(g, status))

                self.gamma[j] = gamv
                self.eps[j] = epsv

                for i in range(self.Nsamples):  # NB: Does not currently work with multidimensional output z, z_r!
                    self.z_r[j][i] = NNr.funcmap(self.x, func1, func2)[i]
                    self.error[j][i] = np.linalg.norm(self.z[i] - self.z_r[j][i], ord=2)
                    self.error_sq[j][i] = self.error[j][i]**2
                    self.error_bound[j][i] = self.gamma[j]*(np.linalg.norm(self.x[:,i], ord=2)**2) + self.eps[j]
                    self.diff[j][i] = self.error_bound[j][i] - self.error_sq[j][i]

                self.NNrs[g] = NNr              
                j += 1
            
    def plot(self, size=(10, 7.5), line=True):
        '''
        Plots a) outputs of full and reduced NNs, b) error bound against m1 for each reduced NN. Thin wrapper around matplotlib.pyplot.
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
        
        # a) plot outputs of full and reduced NNs
        plt.figure('outputs', figsize=size)
        
        if line:
            plt.plot(self.x.T, self.z, label=r'$f(x)$')
            for j in range(self.Nm1):
                plt.plot(self.x.T, self.z_r[j], label=r'$g(x)$ with $m1 = {0}$'.format(self.m1_range[j]))
                
        else:
            plt.plot(self.x.T, self.z, marker='o', label=r'$f(x)$')
            for j in range(self.Nm1):
                plt.plot(self.x.T, self.z_r[j], marker='x', label=r'$g(x)$ with $m1 = {0}$'.format(self.m1_range[j]))
        
        plt.legend()
        plt.grid(linewidth=0.5)
        plt.xlabel(r'input $x$')
        plt.ylabel(r'neural network output, $z$')
        plt.title('Outputs of full- and reduced-order neural networks [$f(x)$, $g(x)$]')
        plt.show()
        
        # b) plot error bound against m1 for each reduced NN
        plt.figure('bounds', figsize=size)
        
        sup_bounds = np.array([])
        sup_errors = np.array([])
        
        for k in range(self.Nm1):
            sup_errors = np.append(sup_errors, np.max(self.error_sq[k]))
            i = np.argmax(self.error_sq[k])
            sup_bounds = np.append(sup_bounds, np.max(self.error_bound[k,i]))
        
        plt.semilogy(self.m1_range, sup_bounds, 'bs', label=r'$\gamma\Vert x\Vert_{2}^{2} + \epsilon$') # log scaling on y axis
        plt.semilogy(self.m1_range, sup_errors, 'kx', label=r'$sup_x \Vert f(x) - g(x)\Vert_{2}^{2}$') # log scaling on y axis
        
        plt.legend()
        plt.grid(which='both', linewidth=0.5)
        plt.xlabel(r'reduced-order networks, hidden layer dimension $m_1$')
        plt.ylabel(r'approximation error')
        plt.title(r'Error bounds against $m_1$')
        plt.show()
        
    def multiplot(self, size=(10, 7.5), line=False):
        '''
        Plots a) outputs of full and reduced NNs, b) error bound against m1 for each reduced NN. Thin wrapper around matplotlib.pyplot.
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
        
        # a) plot outputs of full and reduced NNs
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
            
            z_r_pca = np.zeros((self.Nm1, self.Nsamples, 2))  # n_components = 2
            for j in range(self.Nm1):
                    z_r_pca[j] = pca.transform(self.z_r[j])
            
            print(u'Explained variance ratio, n_components = 2 (z\u2081, z\u2082): {0}'.format(pca.explained_variance_ratio_))
        
        if line:
            if self.z.shape[1] > 1:
                plt.plot(x_pca[:,0], z_pca[:,0], label=r'$f(x)$')
                for j in range(self.Nm1):
                    plt.plot(x_pca[:,0], z_r_pca[j][:,0], label=r'$g(x)$ with $m1 = {0}$'.format(self.m1_range[j]))
                    plt.ylabel(r'principal component $z_1$ from PCA on input $z$, where nz > 1')
            else:
                plt.plot(x_pca[:,0], self.z, label=r'$f(x)$')
                for j in range(self.Nm1):
                    plt.plot(x_pca[:,0], self.z_r[j], label=r'$g(x)$ with $m1 = {0}$'.format(self.m1_range[j]))
                    plt.ylabel(r'neural network output, $z$')
                
        else:
            if self.z.shape[1] > 1:
                plt.plot(x_pca[:,0], z_pca[:,0], 'o', label=r'$f(x)$')
                for j in range(self.Nm1):
                    plt.plot(x_pca[:,0], z_r_pca[j][:,0], 'x', label=r'$g(x)$ with $m1 = {0}$'.format(self.m1_range[j]))
                    plt.ylabel(r'principal component $z_1$ from PCA on input $z$, where nz > 1')
            else:
                plt.plot(x_pca[:,0], self.z, 'o', label=r'$f(x)$')
                for j in range(self.Nm1):
                    plt.plot(x_pca[:,0], self.z_r[j], 'x', label=r'$g(x)$ with $m1 = {0}$'.format(self.m1_range[j]))
                    plt.ylabel(r'neural network output, $z$')
        
        plt.legend()
        plt.grid(linewidth=0.5)
        plt.xlabel(r'principal component $x_1$ from PCA on input $x$, where nx > 1')
        plt.title('Outputs of full- and reduced-order neural networks [$f(x)$, $g(x)$]')
        plt.show()
        
        # b) plot error bound against m1 for each reduced NN
        plt.figure('bounds', figsize=size)
        
        sup_bounds = np.array([])
        sup_errors = np.array([])
        
        for k in range(self.Nm1):
            sup_errors = np.append(sup_errors, np.max(self.error_sq[k]))
            i = np.argmax(self.error_sq[k])
            sup_bounds = np.append(sup_bounds, np.max(self.error_bound[k,i]))
        
        plt.semilogy(self.m1_range, sup_bounds, 'bs', label=r'$\gamma\Vert x\Vert_{2}^{2} + \epsilon$') # log scaling on y axis
        plt.semilogy(self.m1_range, sup_errors, 'kx', label=r'$sup_x \Vert f(x) - g(x)\Vert_{2}^{2}$') # log scaling on y axis
        
        plt.legend()
        plt.grid(which='both', linewidth=0.5)
        plt.xlabel(r'reduced-order networks, hidden layer dimension $m_1$')
        plt.ylabel(r'approximation error')
        plt.title(r'Error bounds against $m_1$')
        plt.show()
        
    def multiplot3D(self, size=(10, 7.5)):  # NB: Does not currently work with multidimensional output z, z_r!
        '''
        Plots a) outputs of full and reduced NNs, b) error bound against m1 for each reduced NN. Thin wrapper around matplotlib.pyplot.
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
            
            z_r_pca = np.zeros((self.Nm1, self.Nsamples, 2))  # n_components = 2
            for j in range(self.Nm1):
                    z_r_pca[j] = pca.transform(self.z_r[j])
            
            print(z_r_pca)
            print(u'Explained variance ratio, n_components = 2 (z\u2081, z\u2082): {0}'.format(pca.explained_variance_ratio_))
            
            ax.scatter(x_pca[:,0], x_pca[:,1], z_pca[:,0], marker='o', label=r'$f(x)$')
            
            for j in range(self.Nm1):
                ax.scatter(x_pca[:,0], x_pca[:,1], z_r_pca[j][:,0], marker='x', label=r'$g(x)$ with $m1 = {0}$'.format(self.m1_range[j]))
                
            ax.set(zlabel=r'principal component $z_1$ from PCA on input $z$, where nz > 1')
        
        else:
            ax.scatter(x_pca[:,0], x_pca[:,1], self.z, marker='o', label=r'$f(x)$')
            
            for j in range(self.Nm1):
                ax.scatter(x_pca[:,0], x_pca[:,1], self.z_r[j], marker='x', label=r'$g(x)$ with $m1 = {0}$'.format(self.m1_range[j]))
                
            ax.set(zlabel=r'neural network output, $z$')
        
        plt.title(r'Output of full-order vs. pruned neural networks [$f(x)$, $g(x)$]')
        ax.legend()
        ax.grid(linewidth=0.5)
        ax.set(xlabel=r'principal component $x_1$ from PCA on input $x$, where nx > 1',
               ylabel=r'principal component $x_2$')
        plt.show()
        
        # b) plot error bound against m1 for each reduced NN
        plt.figure('bounds', figsize=size)
        
        sup_bounds = np.array([])
        sup_errors = np.array([])
        
        for k in range(self.Nm1):
            sup_errors = np.append(sup_errors, np.max(self.error_sq[k]))
            i = np.argmax(self.error_sq[k])
            sup_bounds = np.append(sup_bounds, np.max(self.error_bound[k,i]))
        
        plt.semilogy(self.m1_range, sup_bounds, 'bs', label=r'$\gamma\Vert x\Vert_{2}^{2} + \epsilon$') # log scaling on y axis
        plt.semilogy(self.m1_range, sup_errors, 'kx', label=r'$sup_x \Vert f(x) - g(x)\Vert_{2}^{2}$') # log scaling on y axis
        
        plt.legend()
        plt.grid(which='both', linewidth=0.5)
        plt.xlabel(r'reduced-order networks, hidden layer dimension $m_1$')
        plt.ylabel(r'approximation error')
        plt.title(r'Error bounds against $m_1$')
        plt.show()

def main():
    import activations

    x = np.linspace(-10, 10, 100)

    weights = np.array([np.array([-0.634700665707495, 1.656731026589558, 1.124487323633357, -0.795741403939511, -1.168109939313002, 0.869874974297699, -1.008855879224341, 1.914651700028183, 1.240317284056857, -1.420252553244148]).reshape(10, 1),
                       np.array([-0.258472107285207, -0.680982998070259, 1.560623889702786, -1.220971595926017, -0.027188941462718, -1.010335979063966, 0.835793863154799, 1.222569086520831, 0.259056187128039, 0.245551285836084]).reshape(10, 1),
                       np.array([0.082854200378300, 0.992706291627412, -1.256284879106125, 0.554918709010267, 0.317351602368993, 1.250828758360099, 1.679813903514902, -0.287094733070243, 1.701671362974561, -0.829936846466298]),
                       np.array([1.864581839050232])], dtype='object')

    NNo = Network()
    NNo.load(2, weights)

    prob = Synthesis(x, 10, NNo, range(1, 10))
    prob.output(activations.relu, activations.null)

    c1, c2 = 1.0, 1.0
    prob.reduce_L2(c1, c2, activations.relu, activations.null)
    prob.plot()
    
if __name__ == "__main__":
	main()