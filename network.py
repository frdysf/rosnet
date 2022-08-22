#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import defaultdict

from activations import relu, null

class Network:
    
    def __init__(self):
        self.NN = defaultdict(lambda: np.ndarray(0)) # weights and biases stored with keys W0, b0, W1, b1, etc.
    
    def load(self, n, weights):
        '''
        Load NN structure with given weights.
        
        Parameters
        ----------
        n: int
            Number of pairs of weights and biases.
            
        weights: np.ndarray
            Array containing weight and bias vectors (as lists) in ascending order: W0, b0, W1, b1...
        '''
        
        if n != len(weights) / 2:
            raise ValueError('n is inconsistent with number of weight vectors. Use help() for more info.')
        
        else:
            self.n = n
            
            self.NN.clear()
            
            for i in range(0, 2*n, 2):
                self.NN['W{0}'.format(i//2)] = np.atleast_2d(weights[i])
                self.NN['b{0}'.format(i//2)] = np.atleast_2d(weights[i+1])
    
    def loadf(self, n, path, frmt):
        '''
        Load NN structure with weights read from file.
        
        Parameters
        ----------
        n: int
            Number of pairs of weights and biases.
            
        path: str
            Relative path to file.

        frmt: str
            'csv' or 'txt'.
        '''
        raise NotImplementedError('This method has not yet been implemented.')
    
    def funcmap(self, x, func1, func2):
        '''
        Generate feedforward function mapping from loaded weights.
        
        Parameters
        ----------
        x: np.ndarray
            Input data.
            
        func1: method
            Activation function applied to hidden layer preceding output layer. See activations.py.
            
        func2: method
            Activation function applied to output layer. See activations.py.
        
        Returns
        -------
        z: np.ndarray
            Output data.
        '''
        
        if not self.NN['W0'].any():
            self.NN.clear()
            raise RuntimeError('Weights have not been loaded. Run load() first.')

        else:

            x = np.atleast_2d(x)
			
            Nsamples = x.shape[1]
            m = self.NN['b{0}'.format(self.n - 1)].shape[0] # dimension of nth (final) bias vector, hence of output
            
            z = np.zeros((Nsamples, m))
            
            for i in range(Nsamples):
                
                y = x[:,i]
                y = np.atleast_2d(y)
                y = y.T
                
                for j in range(self.n - 1):  
                    w = self.NN['W{0}'.format(j)]
                    b = self.NN['b{0}'.format(j)]
                    y = np.dot(w, y) + b
                
                phi = func1(y)
                w = self.NN['W{0}'.format(self.n - 1)]
                b = self.NN['b{0}'.format(self.n - 1)]
                z[i:,] = func2(np.dot(w, phi) + b).T
                
            return z

def main():
	# Test with Example 1

	x = np.linspace(-10, 10, 100)
	weights = np.array([np.array([-0.634700665707495, 1.656731026589558, 1.124487323633357, -0.795741403939511, -1.168109939313002, 0.869874974297699, -1.008855879224341, 1.914651700028183, 1.240317284056857, -1.420252553244148]).reshape(10, 1),
                   np.array([-0.258472107285207, -0.680982998070259, 1.560623889702786, -1.220971595926017, -0.027188941462718, -1.010335979063966, 0.835793863154799, 1.222569086520831, 0.259056187128039, 0.245551285836084]).reshape(10, 1),
                   np.array([0.082854200378300, 0.992706291627412, -1.256284879106125, 0.554918709010267, 0.317351602368993, 1.250828758360099, 1.679813903514902, -0.287094733070243, 1.701671362974561, -0.829936846466298]),
                   np.array([1.864581839050232])], dtype='object')

	test = Network()
	test.load(2, weights)

	if not test.NN['W0'].any(): # test if NN has been loaded correctly
		print('False')
	else:
		print('True')

	print(getattr(test, 'NN'))
	print(test.funcmap(x, relu, null))
	
if __name__ == "__main__":
	main()