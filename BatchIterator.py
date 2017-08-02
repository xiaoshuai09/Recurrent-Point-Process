import numpy as np
import random 


class SimpleDataIterator():
    def __init__(self, df,T,MARK,DIFF=False):
        self.df = df
        self.T = T
        self.MARK = MARK
        self.DIFF = DIFF
        self.size = len(self.df)
        self.length = [len(item) for item in self.df]
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.df)
        self.length = [len(item) for item in self.df]
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df[self.cursor:self.cursor+n]
        seqlen = self.length[self.cursor:self.cursor+n]
        self.cursor += n
        return res,seqlen
      

class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df[self.cursor:self.cursor+n]
        seqlen = self.length[self.cursor:self.cursor+n]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(seqlen)
        #x = np.zeros([n, maxlen,1], dtype=np.float32)
        if self.MARK:
            x = np.ones([n, maxlen,2], dtype=np.float32)*self.T
        else:
            x = np.ones([n, maxlen,1], dtype=np.float32)*self.T
        for i, x_i in enumerate(x):
            if self.MARK:
                x_i[:seqlen[i],:] = res[i] # asarray
            else:
                x_i[:seqlen[i],0] = res[i] 
        
        if self.DIFF==True:
            if self.MARK:
                  xt = np.concatenate([x[:,0:1,0:1],np.diff(x[:,:,0:1],axis=1)],axis=1)
                  x = np.concatenate([xt,x[:,:,1:]],axis=2)
            else:
                  x = np.concatenate([x[:,0:1,:],np.diff(x,axis=1)],axis=1)
        return x, np.asarray(seqlen)
      

class BucketedDataIterator():
    def __init__(self, df, T, MARK,DIFF=False, num_buckets = 5):
        self.df = df
        self.length = [len(item) for item in self.df]
        
        temp_ = sorted(zip(self.df,self.length),key= lambda x:x[1])
        self.df = [item[0] for item in temp_]
        self.length = [item[1] for item in temp_]
        
        self.T = T
        self.MARK = MARK
        self.DIFF = DIFF
        
        self.size = len(df) / num_buckets
         
 
        self.dfs = []
        self.lengths = []
        for bucket in range(num_buckets):
            self.dfs.append(self.df[bucket*self.size: (bucket+1)*self.size])
            self.lengths.append( [len(item) for item in self.dfs[bucket]] )
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            random.shuffle(self.dfs[i])
            self.lengths[i] = [len(item) for item in self.dfs[i]]
            self.cursor[i] = 0

    def next_batch(self, n):
        if np.any(self.cursor+n+1 > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0,self.num_buckets)

        res = self.dfs[i][self.cursor[i]:self.cursor[i]+n]
        seqlen = self.lengths[i][self.cursor[i]:self.cursor[i]+n]
        self.cursor[i] += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(seqlen)
        #x = np.zeros([n, maxlen,1], dtype=np.float32)
        if self.MARK:
            x = np.ones([n, maxlen,2], dtype=np.float32)*self.T
        else:
            x = np.ones([n, maxlen,1], dtype=np.float32)*self.T
        for i, x_i in enumerate(x):
            if self.MARK:
                x_i[:seqlen[i],:] = res[i] # asarray
            else:
                x_i[:seqlen[i],0] = res[i] 
        
        if self.DIFF==True:
            if self.MARK:
                  xt = np.concatenate([x[:,0:1,0:1],np.diff(x[:,:,0:1],axis=1)],axis=1)
                  x = np.concatenate([xt,x[:,:,1:]],axis=2)
            else:
                  x = np.concatenate([x[:,0:1,:],np.diff(x,axis=1)],axis=1)
        return x, np.asarray(seqlen)

