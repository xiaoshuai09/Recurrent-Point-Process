import abc
import scipy.stats
import numpy as np

class Intensity(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def getValue(self, t):
        return
    
class IntensitySumGaussianKernel(Intensity):

    def __init__(self, k=2, centers=[2, 4], stds=[1, 1], coefs= [1, 1]):
        self.k = k
        self.centers = centers
        self.stds = stds
        self.coefs = coefs
        
    def getValue(self, t):
        inten = 0
        for i in range(self.k):
            inten += self.coefs[i] * scipy.stats.norm.pdf(t, self.centers[i], self.stds[i])
        return inten
    
    def getUpperBound(self, from_t, to_t):
        max_val = max(self.getValue(from_t), self.getValue(to_t))
        for i in range(self.k):
            max_val = max(max_val, self.getValue(self.centers[i]))
        for i in range(self.k-1):
            point = (self.coefs[i]*self.centers[i]/self.stds[i] + self.coefs[i+1]*self.centers[i+1]/self.stds[i+1])/\
                (self.coefs[i]/self.stds[i] + self.coefs[i+1]/self.stds[i+1])
            max_val = max(max_val, self.getValue(point))    
        return max_val
    
class IntensityHomogenuosPoisson(Intensity):

    def __init__(self, lam):
        self.lam = lam
        
    def getValue(self, t):
        return self.lam
    
    def getUpperBound(self, from_t, to_t):
        return self.lam
    

def generate_sample(intensity, T, n):
    Sequnces = []
    i = 0
    while True:
        seq = []
        t = 0
        while True:
            intens1 = intensity.getUpperBound(t,T)
            dt = np.random.exponential(1/intens1)
            new_t = t + dt
            if new_t > T:
                break
                
            intens2 = intensity.getValue(new_t)
            u = np.random.uniform()
            if intens2/intens1 >= u:
                seq.append(new_t)
            t = new_t
        if len(seq)>1:
            Sequnces.append(seq) 
            i+=1
        if i==n:
            break
    return Sequnces



class MarkedIntensity(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def getValue(self, t, inds=1):
        return
    
class MarkedIntensityIndepenent(MarkedIntensity):
    
    def __init__(self, dim=1):
        self.dim = dim
        self.intensities = [None]*dim
        
    
    def initialize(self, intensity, dim=1):
        self.intensities[dim] = intensity
        
    def getValue(self, t, inds=1):
        l = len(inds)
        inten = [0]*l
        for i in range(l):
            inten[i] +=  1+self.intensities[inds[i]].getValue(t)
        return inten
    
    def getUpperBound(self, from_t, to_t, inds=1):
        l = len(inds)
        inten = [0]*l
        for i in range(l):
            inten[i] =  self.intensities[inds[i]].getUpperBound(from_t, to_t)
        return inten
    
class MarkedIntensityHomogenuosPoisson(Intensity):

    def __init__(self, dim=1):
        self.dim = dim
        self.lam = [None]*dim
    
    def initialize(self, lam, dim=1):
        self.lam[dim] = lam
        
    def getValue(self, t, inds):
        l = len(inds)
        inten = [0]*l
        for i in range(l):
            inten[i] =  self.lam[i]
        return inten
    
    def getUpperBound(self, from_t, to_t, inds):
        l = len(inds)
        inten = [0]*l
        for i in range(l):
            inten[i] =  self.lam[i]
        return inten
    
def generate_samples_marked(intensity, T, n):
    U = intensity.dim
    Sequences = []
    inds = np.arange(U)
    for i in range(n):
        seq = []
        t = 0
        while True:
            intens1 = intensity.getUpperBound(t,T,inds)
            #print(intens1)
            dt = np.random.exponential(1/sum(intens1))
            #print(dt)
            new_t = t + dt
            #print(new_t)
            if new_t > T:
                break
            intens2 = intensity.getValue(new_t, inds)
            #print(intens2)
            u = np.random.uniform()
            if sum(intens2)/sum(intens1) > u:
                #print(intens2)
                x_sum = sum(intens2)
                norm_i = [ x/x_sum for x in intens2]
                #print(norm_i)
                dim = np.nonzero(np.random.multinomial(1, norm_i))
                seq.append([np.asscalar(dim[0]),new_t])
            t = new_t
        if len(seq)>1:
            Sequences.append(seq) 
    return Sequences
    
