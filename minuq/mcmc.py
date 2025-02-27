import numpy as np
from typing import Callable
import scipy as sp

class mcmcChain() :
        
    def __init__(self, log_targetPDF: Callable, PDF_args: list, initial_state : np.array, chainLength: int, mode: str) :

        #set current state to initial state
        self.current_state = np.copy(initial_state)
        
        #set current state to initial state
        self.proposed_state = np.copy(initial_state)

        #set number of parameters to infer
        self.numParams = self.current_state.shape[0]

        #length of MCMC chain
        self.chainLength=chainLength

        #container for the Markov chain
        self.chain=np.zeros( (self.chainLength, self.numParams) )

        #number of accepted proposals
        self.accepted=0

        #number of chain reports to print
        self.chainReport=20

        #target PDF function (function which returns log of the pdf)
        self.targetPDF = log_targetPDF

        #some target PDF arguments (other than the state, e.g. data for a Bayesian likelihood, a function handle for a forward model, meta-parameters, constants etc.)
        self.PDF_args = PDF_args

        #compute PDF at initial state
        self.f_current = self.targetPDF( self.current_state, self.PDF_args )

        #proposal standard deviations 
        self.proposalSigmas=np.copy(initial_state)*1.0

        #mode
        self.mode=mode

        if self.mode=="MH":
            print("MCMC using Metropolis-Hastings")
        else :
            print("Error: invalid mode specifier")
        return
    
    def __str__(self) :
        return "mcmcChain object"
    

    # proposal sampling normal distribution using inverse of CDF
    def norm1d_samp(self,mu_, sig_) :
        
        #random number in [0,1]
        y_ = np.random.rand()
        #inverse of normal CDF using inverse error function
        x_ =sig_*np.sqrt(2.)*sp.special.erfinv(y_*2. - 1.)    + mu_
        
        return x_   


    def proposal(self):

        #Metropolis-Hastings, i.e. from a symmetric distribution (i.e. p( x1 | x2 ) = p( x2 | x1 )
        for j in range(self.numParams) :
            self.proposed_state[j] = self.norm1d_samp(self.current_state[j], self.proposalSigmas[j]) 

        return
    
    
    def acceptReject(self, i, f_proposal):

        #compute probability ratio
        #alpha_ = np.exp(f_proposal)/np.exp(self.f_current)
        alpha_ = np.exp(f_proposal -self.f_current)


        if alpha_ >= np.random.rand() :
            #accept the proposed state as the next sample (always accept samples with higher probability)
            self.accepted+=1
            if (i % int(self.chainLength/self.chainReport) ==0  or i==1 ):
                print(i,':', 'x_current=',self.current_state, 'x_proposed=', self.proposed_state, 'fp=',self.f_current, 'fn=',f_proposal, 'alpha=',alpha_, 'ACCEPT', 'AcceptRatio=', self.accepted/i, "Proposal stds", self.proposalSigmas)
            self.current_state = np.copy(self.proposed_state)
            #note -1 for zero indexing
            self.chain[i-1,:] = self.current_state

            #update the current target PDF evaluation
            self.f_current =f_proposal

            #update the posterior mode
            #self.mode = 

        else :
            #reject the proposed state, reuse previous state as next sample
            if (i % int(self.chainLength/self.chainReport) ==0 or i==1) :
                print(i,':', 'x_current=',self.current_state, 'x_proposed=', self.proposed_state, 'fp=',self.f_current, 'fn=',f_proposal, 'alpha=',alpha_, 'REJECT', 'AcceptRatio=', self.accepted/i, "Proposal stds", self.proposalSigmas)
            #note -1 for zero indexing
            self.chain[i-1,:] = self.current_state

        return


    def runChain(self):

        #run chain
        for i in range(1,self.chainLength+1) :
            
            #generate a proposal at the current state
            self.proposal()

            #compute target PDF for proposal
            f_proposal = self.targetPDF( self.proposed_state, self.PDF_args )

            #compute probability ratio
            self.acceptReject(i, f_proposal)
          
        return


if __name__=="__main__" :
    pass


