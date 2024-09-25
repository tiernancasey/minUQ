import numpy as np
import scipy.optimize as opt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#polynomial chaos expansion class
class pce :
  
    def __init__(self, dim=0, order=0, basisType='legendre') :
        self.dim=dim
        self.order=order
        self.numTerms=self.setNumPCETerms()
        self.mi=self.genMultiIndex()
        self.coefficients=None
        self.coefficientsSet=False
        self.basisType=basisType
        self.basisInnerProd=self.computeBasisInnerProducts()
        #scaling parameters
        self.x_mins=None
        self.x_maxs=None
        self.y_mins=None
        self.y_maxs=None
        
    def legendre(self, n: int, x: float):

        if n<0 :
            print(f"Error: polynomial order can not be < 0")
            return
        if n==0 :
            return x*0 + 1 #to match the dimension of x
        elif n==1 :
            return x  
        else :
            #Bonnet's recursion formula
            return (  (2.*n-1)*x*self.legendre(n-1,x) -(n-1)*self.legendre(n-2,x) )/n


    def setNumPCETerms(self) :
        #number of PC terms = (dim + order)!/dim!order!
        return int( np.math.factorial(self.dim+self.order)/(np.math.factorial(self.dim) * np.math.factorial(self.order) ) )


    def genMultiIndex(self) :

        #create zero array
        a=[]
        for i in range(self.dim):
            a.append(0)

        #append to multi-index matrix
        miArray=[a]

        #build for each order
        for i in range(0,self.order+1) :
            _=self.fillMI(miArray,[],a,i,i)
    
        return miArray


    def fillMI(self,miArray: list, b: list, a: list, remaining_order: int, global_order: int ) :
    
        #base case, if order remaining is zero remaining entries will all be zero
        if remaining_order==0 or len(a)==0 :
            return a
        else :

            #iterate over index
            for i in range(remaining_order,-1,-1):
            
                a[0]=i
                #fill the remaining 
                a[1:]=self.fillMI(miArray,b+a[0:1],a[1:],remaining_order-i, global_order)
                #stamp into main array
                if sum(b+a) ==global_order :
                    miArray.append(b+a)

        return a


    def evaluatePolynomialTerms(self,x) :

        #expect x to be an  (ndata x dim) array
        assert x.shape[1]==self.dim ,f"Incorrect dimension for input data (x), {self.dim} expected but received {x.shape[1]}" 

        #number of data points
        nData=x.shape[0]

        #loop over terms
        terms=np.ones( ( nData, self.numTerms  ) )
        for i in range(0,self.numTerms) :

            #loop over dimensions
            #terms.append(1)
            for k in range(0,self.dim) :
                #print(j,)
                #terms[i]*=legendre( self.mi[i][k], x[0,k])
                if self.basisType=='legendre':  
                    terms[:,i] = np.multiply( terms[:,i],  self.legendre( self.mi[i][k], x[:,k] ) )
                elif self.basisType ==  'hermite':
                    terms[:,i] = np.multiply( terms[:,i],  self.hermite( self.mi[i][k], x[:,k] ) )
        return terms
    


    def pcePredict(self,x) :

        assert x.shape[1]==self.dim ,f"Incorrect dimension for input data (x), {self.dim} expected but received {x.shape[1]}" 
        
        #rescale input to [-1,1]
        x_scaled= x - self.x_mins #shifts to [0, (max-min)]
        x_scaled=x_scaled/ ( (self.x_maxs-self.x_mins)/2. ) #scales to [0, 2]
        x_scaled-=1 #[shifts to [-1,1]

        #basis = self.evaluatePolynomialTerms(x)
        basis = self.evaluatePolynomialTerms(x_scaled)

        y=np.dot(self.coefficients.T,basis.T)

        return y



    def fitPCE(self,train_x,train_y,method='LSQ') :
        
        #rescale inputs and output to [-1,1] and store scaling parameters 
        self.x_mins=np.min(train_x, axis=0)
        self.x_maxs=np.max(train_x, axis=0)

        train_x_scaled=train_x - self.x_mins
        train_x_scaled=train_x_scaled/ ( (self.x_maxs-self.x_mins)/2. )
        train_x_scaled-=1

        if method=='LSQ':
            basis = self.evaluatePolynomialTerms(train_x_scaled)
            #basis = self.evaluatePolynomialTerms(train_x)

            #least squares fit

            #X_ij = X_{pt,dim}
            #c_i = (X_ij.T X_ij )^-1 X_ij.T Y_i
            s =   np.dot(  np.linalg.inv ( np.dot(  basis.T,basis )  ), basis.T)

            self.coefficients =    np.dot(s,train_y) 

        else :
            print("ERROR: unknown method")
            return

        #coefficients are defined
        self.coefficientsSet=True

        return 



    def computeBasisInnerProducts(self):

        #number of quadrature points in each dimension
        nQuadPts=1000
        x_quad=np.linspace(-1.,1.,nQuadPts)

        InnerProducts = np.ones(self.numTerms)
        #loop over terms in the PCE
        for i in range(0,self.numTerms) :
            #loop over dimensions within each polynomial term
            for k in range(0,self.dim) :
                
                if self.basisType=='legendre':
                    #compute basis function for this dimension  
                    a =  np.array(self.legendre( self.mi[i][k], x_quad ) )
                    #inner product (note the 0.5 weight)
                    InnerProducts[i]*=0.5*np.trapz(a**2.,x_quad)
                    
        return InnerProducts


    def computeVariances(self):

        if not self.coefficientsSet:
            print("Error: PCE coefficients not defined")
            return -1
        
    
        totalVar=0
        varFracs=[]
        varFracs=np.zeros(self.numTerms-1)
        for i in range(1,self.numTerms):
            #skip the constant term at index 0

            #varFracs.append(self.coefficients[i]**2. * self.basisInnerProd[i])
            #totalVar+=self.coefficients[i]**2. * self.basisInnerProd[i]
            #compute variance fraction for each term
            varFracs[i-1] = self.coefficients[i]**2. * self.basisInnerProd[i]
            #compute total variance
            totalVar+=varFracs[i-1]
        varFracs=np.array(varFracs) #dont normalize by total variance here

        return totalVar, varFracs
    

    def computeSobolIndices(self):

        if not self.coefficientsSet :
            print(f"Error: PCE coefficients undefined")
            return -1
        
        totalVar, varFracs=self.computeVariances()

        #======== compute Sobol indices =========

        sobolMain=np.zeros(self.dim)
        sobolTotal=np.zeros(self.dim)
        #loop over dimensions
        for i in range(0,self.dim) :

            #loop over all PCE terms (except for the constant term)
            for j in range(1,self.numTerms) :

                #MAIN EFECT INDICES (accumulate variances for terms containing variation in this dimension only)
                if sum(self.mi[j]) - self.mi[j][i] !=0 : #(i.e. there are non-zeros in the other dimensions for this term)
                    pass
                    #ignore this term
                else: 
                    #add this term's contribution to the main Sobol index for this dimension
                    sobolMain[i]+=varFracs[j-1] #j-1 because the varFracs array is of size numTerms-1
                    
                if self.mi[j][i]!=0: #(i.e. this dimension contributes to this term)
                    sobolTotal[i]+=varFracs[j-1]
        #=======================================

        return sobolMain/totalVar, sobolTotal/totalVar
