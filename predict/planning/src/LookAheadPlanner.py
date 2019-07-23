'''
Created on Oct 22, 2017

@author: Sara
'''
from gurobipy import *
import numpy as np
import time
from PatrolProblem import PatrolProblem



#file1='./QENP_AnimalNonCom/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_detect_2.csv'
#file1='/Users/Sara/Documents/Euler/PathPlanning/mfnp_animalnoncom-1/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_t hreshold7.5_6years_blackBoxFunction_detect.csv'
file1='../QENP_AnimalNonCom/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_attack.csv'
file2='../QENP_AnimalNonCom/PatrolPosts.csv'
filep='../QENP_AnimalNonCom/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_detect_2.csv'

#file1="/Users/Sara/Documents/Euler/PathPlanning/mfnp_animalnoncom-1/archive/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_attack.csv"
# file1="../MFNP_AnimalNonCom/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_attack.csv"
# file2='../MFNP_AnimalNonCom/PatrolPosts.csv'
# filep='../MFNP_AnimalNonCom/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_detect.csv'


class LookAheadPlanner(object):

    def __init__(self, graph=None, days=15):
        self.r = 10
        self.c = 10
        self.n = self.r*self.c
        self.r_s = self.r //2
        self.c_s = self.c//2
        self.T = 10
        self.days = days
        self.count_patrols=1
     
    def genSOSVars(self, m, x_vals, y_vals, x, y):
        wx =[[0 for i in range(len(x_vals[j]))] for j in range(self.n)]
        for k in range(self.n):
            for i in range(len(x_vals[k])):
                wx[k][i] = m.addVar(lb=0, ub = 1.0, vtype=GRB.CONTINUOUS, name="wx%d-%d" %(k,i))
        wy =[[0 for i in range(len(y_vals[j]))] for j in range(self.n)]
        for k in range(self.n):    
            for i in range(len(y_vals[k])):
                wy[k][i] = m.addVar(lb=0, ub = 1.0, vtype=GRB.CONTINUOUS, name="wy%d-%d" % (k,i))
        wxy=[[[0 for k in range(len(y_vals[j]))] for i in range(len(x_vals[j]))] for j in range(self.n)]
        for k in range(self.n):    
            for i in range(len(x_vals[k])):
                for j in range(len(y_vals[k])):
                    wxy[k][i][j] = m.addVar(lb=0, ub = 1.0, vtype=GRB.CONTINUOUS, name="wyx%d-%d-%d" % (k,i,j))
        m.update() 
        
        
        for i in range(self.n):
            m.addConstr(1 == quicksum( wx[i][j] for j in range(len(x_vals[i]))))
            m.addConstr(x[i] == quicksum( wx[i][j]*x_vals[i][j] for j in range(len(x_vals[i]))))
               
        for i in range(self.n):
            m.addConstr(1 == quicksum( wy[i][j] for j in range(len(y_vals[i]))))
            m.addConstr(y[i] == quicksum( wy[i][j]*y_vals[i][j] for j in range(len(y_vals[i]))))
        for k in range(self.n):
            for i in range(len(x_vals[k])):
                m.addConstr(wx[k][i] == quicksum( wxy[k][i][j] for j in range(len(y_vals[k]))))
        
            for j in range(len(y_vals[k])):
                m.addConstr(wy[k][j] == quicksum( wxy[k][i][j] for i in range(len(x_vals[k]))))
        for k in range(self.n):
            if len(x_vals)>1:
                m.addSOS(GRB.SOS_TYPE2, wx[k])#,x_vals)
                m.addSOS(GRB.SOS_TYPE2, wy[k])#, y_vals)
        m.update() 
    
        return wx, wy, wxy
        
    def genPatrolVars(self, m, e):
        x = {}
        f = [[[0 for t in range(self.T)] for j in range(self.n)] for i in range(self.n)]
        for i in range(self.n):
            ub = e[i][-1]
            x[i] = m.addVar(lb=0, ub = ub, vtype=GRB.CONTINUOUS, name="x%d-%d" % (self.count_patrols,i))
        for i in range(self.n):
            for j in range(self.n):
                for t in range(self.T):
                        f[i][j][t] = m.addVar(lb=0, ub = 1.0, vtype=GRB.CONTINUOUS, name="f%d-%d-%d-%d" % (self.count_patrols,i,j,t))
        
        m.update()
        flat_list = []
        for sf in f:
            for ssf in sf:
                for item in ssf:
                    flat_list.append(item)
    
        m.addConstr(self.T >= quicksum(flat_list))
        
        # Add constraints
        m.addConstr(self.T*self.days >= quicksum( x[j] for j in range(self.n)))
        for i in range(self.r):
            for j in range(self.c):
                k = i*self.c+j
                fl = []
                if j<(self.c-1): fl.append(f[i*self.c+j+1][k])
                if j>0: fl.append(f[i*self.c+j-1][k])
                if i<(self.r-1): fl.append(f[(i+1)*self.c+j][k])
                if i>0: fl.append(f[(i-1)*self.c+j][k])
                flat_fl = [val for sublist in fl for val in sublist]
                m.addConstr(x[k] == self.days*quicksum( flat_fl[p] for p in range(len(flat_fl))))
                
                for t in range(1,self.T):    
                    flin = []
                    if j<(self.c-1): flin.append(f[i*self.c+j+1][k][t-1])
                    if j>0: flin.append(f[i*self.c+j-1][k][t-1])
                    if i<(self.r-1): flin.append(f[(i+1)*self.c+j][k][t-1])
                    if i>0: flin.append(f[(i-1)*self.c+j][k][t-1])
                    flin.append(f[k][k][t-1])
                    
                    flout=[]
                    if j<self.c-1: flout.append(f[k][i*self.c+j+1][t])
                    if j>0: flout.append(f[k][i*self.c+j-1][t])
                    if i<self.r-1: flout.append(f[k][(i+1)*self.c+j][t])
                    if i>0: flout.append(f[k][(i-1)*self.c+j][t])
                    flout.append(f[k][k][t])
                   
                    m.addConstr(quicksum( flout[p] for p in range(len(flout))) == quicksum( flin[p] for p in range(len(flin))))
        for i in range(self.r):
            for j in range(self.c):
                d=0
                k = i*self.c+j
                if i==self.r_s and j==self.c_s: 
                    d=1
                flin = []
                if j<(self.c-1): flin.append(f[k][i*self.c+j+1][0])
                if j>0: flin.append(f[k][i*self.c+j-1][0])
                if i<(self.r-1): flin.append(f[k][(i+1)*self.c+j][0])
                if i>0: flin.append(f[k][(i-1)*self.c+j][0])
                flin.append(f[k][k][0])
                m.addConstr(d == quicksum( flin[j] for j in range(len(flin))))
        
                flout=[]
                if j<self.c-1: flout.append(f[i*self.c+j+1][k][self.T-1])
                if j>0: flout.append(f[i*self.c+j-1][k][self.T-1])
                if i<self.r-1: flout.append(f[(i+1)*self.c+j][k][self.T-1])
                if i>0: flout.append(f[(i-1)*self.c+j][k][self.T-1])
                flout.append(f[k][k][self.T-1])
                m.addConstr(quicksum( flout[j] for j in range(len(flout))) == d)
        fl = []
        k = self.r_s*self.c+self.c_s
        print k
        if self.c_s<self.c-1: fl.append(f[k][self.r_s*self.c+self.c_s+1][0])
        if self.c_s>0: fl.append(f[k][self.r_s*self.c+self.c_s-1][0])
        if self.r_s<self.r-1: fl.append(f[k][(self.r_s+1)*self.c+self.c_s][0])
        if self.r_s>0: fl.append(f[k][(self.r_s-1)*self.c+self.c_s][0])
        m.addConstr(quicksum( fl[j] for j in range(len(fl))) == 1)
    
        
        flin=[]
        flout=[]
        if self.c_s<self.c-1: flout.append(f[self.r_s*self.c+self.c_s+1][k][self.T-1])
        if self.c_s>0: flout.append(f[self.r_s*self.c+self.c_s-1][k][self.T-1])
        if self.r_s<self.r-1: flout.append(f[(self.r_s+1)*self.c+self.c_s][k][self.T-1])
        if self.r_s>0: flout.append(f[(self.r_s-1)*self.c+self.c_s][k][self.T-1])
        m.addConstr(quicksum( flout[j] for j in range(len(flout))) == 1)
        
                
        m.update() 
        return x, f
    def genPatrolVarsfromGraph(self, m, graph, e):
        x = {}
        f = [[[0 for t in range(self.T)] for j in range(self.n)] for i in range(self.n)]
        for i in range(self.n):
            ub = e[i]
            ub = ub[-1]
            x[i] = m.addVar(lb=0, ub = ub, vtype=GRB.CONTINUOUS, name="x%d-%d" % (self.count_patrols,i))
        for i in range(self.n):
            for j in range(self.n):
                for t in range(self.T):
                        f[i][j][t] = m.addVar(lb=0, ub = 1.0, vtype=GRB.CONTINUOUS, name="f%d-%d-%d-%d" % (self.count_patrols,i,j,t))
        
        m.update()
        flat_list = []
        for sf in f:
            for ssf in sf:
                for item in ssf:
                    flat_list.append(item)
    
        m.addConstr(self.T >= quicksum(flat_list))
        
        # Add constraints
        m.addConstr(self.T*self.days >= quicksum( x[j] for j in range(self.n)))
        for k in range(self.n):
            fl = []
            fl.append(f[k][k])
            for i in graph.neighbors(k):
                fl.append(f[i][k])
                
    
            flat_fl = [val for sublist in fl for val in sublist]
            m.addConstr(x[k] == self.days*quicksum( flat_fl[p] for p in range(len(flat_fl))))
                
            for t in range(1,self.T):    
                flin = []
                flout=[]
                for i in graph.neighbors(k):
                    flin.append(f[i][k][t-1])
                    flin.append(f[k][k][t-1])
                    
                    flout.append(f[k][i][t])
                flin.append(f[k][k][t-1])
                flout.append(f[k][k][t])
                   
                m.addConstr(quicksum( flout[p] for p in range(len(flout))) == quicksum( flin[p] for p in range(len(flin))))
    
            d=0
            if k==graph.source:
                d=1
            flin = []
            flout=[]
            flin.append(f[k][k][0])
            flout.append(f[k][k][self.T-1])
            
            for i in graph.neighbors(k):
                flin.append(f[k][i][0])
                flout.append(f[i][k][self.T-1])  
                     
            m.addConstr(d == quicksum( flin[j] for j in range(len(flin))))
            m.addConstr(quicksum( flout[j] for j in range(len(flout))) == d)
            
        fl = []
        k = graph.source
        for i in graph.neighbors(k):
            fl.append(f[k][i][0])
        #m.addConstr(quicksum( fl[j] for j in range(len(fl))) == 1)
    
        
        flin=[]
        flout=[]
        for i in graph.neighbors(k):
            flout.append(f[i][k][self.T-1])
        #m.addConstr(quicksum( flout[j] for j in range(len(flout))) == 1)
        
                
        m.update() 
        return x, f
    
    def getSol(self, m, x, f):
        sol=[0 for i in range(self.n)]
        solf=[[[0 for t in range(self.T)] for j in range(self.n)] for i in range(self.n)]
        if m.status == GRB.Status.OPTIMAL:
            obj = m.getAttr('ObjVal')
            solx = m.getAttr('x', x)
            for i in range(self.n):
                for j in range(self.n):
                    for t in range(self.T):
                        solf[i][j][t] =  f[i][j][t].getAttr('x')
        for i in range(self.n):
            #name = "x%d" % i
            sol[i]=solx[i]
        return sol, solf, obj   
    def setObjectiveandSolve(self, m, data, wxy, x2, effortx1, effortx2,datapast=None): 
        obj = []
        d1 = [[0 for i in range(len(effortx1[j]))] for j in range(self.n)]
        for k in range(self.n):    
            for i in range(len(effortx1[k])):
                d1[k][i]=data[k][i][0]
                for j in range(len(effortx2[k])):
                    obj.append(wxy[k][i][j]*data[k][i][j])
        m.update() 
        
        # Set objective
        #m.setObjective( 0.0, GRB.MAXIMIZE)
        m.setObjective( quicksum(obji for obji in obj), GRB.MAXIMIZE)
        
        
        for i in range(self.n):
            #print d1[i],  len(effortx1)
            if len(effortx1[i]) >1:
                #print x2[i], effortx1[i], d1[i]
                m.setPWLObj(x2[i], effortx2[i], datapast[i][:len(effortx2[i])])
        m.update()
    def printDatafiles(self, g, file1, file2):
        levels = 2
        for i in range(20):
            m = Model()
            data, effortx1, effortx2 = g.loadFile2Step(file1, file2, i, points=levels) #
            self.n=g.n
            
    def runExperiments(self, g, file1, file2, filep):
        levels = 3
        for i in range(16,17):
            m = Model()
            datapast, effortx = g.loadFile1D(filep, file2, i, points=levels)
            
            data, effortx1, effortx2 = g.loadFile2D(file1, file2, i, points=levels) #
            self.n=g.n
    
    
            # Add variables
            #times step 2
            x1,f1 = self.genPatrolVarsfromGraph(m, g.graph, effortx1)
            self.count_patrols+=1
            
            #time step 1
            x2,f2 = self.genPatrolVarsfromGraph(m, g.graph, effortx2)
            self.count_patrols+=1            
            
            wx, wy, wxy = self.genSOSVars(m, effortx1, effortx2, x1, x2)
            
            obj = self.setObjectiveandSolve(m, data, wxy, x2, effortx1, effortx2, datapast=datapast)
            
            start = time.time()

            m.write("out.lp")
            m.optimize()
            #m.write("out.sol")
            end = time.time()
            totaltime = end-start
            
            

            solx1, solf1, obj = self.getSol(m, x1, f1)
            solx2, solf2, obj = self.getSol(m, x2, f2)
            
             
            soldata1 = [i, totaltime, obj]
            soldata1.extend(solx1)
            
            soldata2 = [i, totaltime, obj]
            soldata2.extend(solx2)
            g.graph.writeSolution("lookahead_15_mnfp.csv", soldata1,soldata2)    


            g.graph.plotHeatGraph(solx1, i, name="Round2")    
            g.graph.samplePatrols(solf1, self.days, post=i, name="Round2")   
            
            g.graph.plotHeatGraph(solx2, i, name="Round1")     
            g.graph.samplePatrols(solf2, self.days, post=i, name="Round1")  
            
            
r = 20
c = 20
n = r*c
r_s = r //2
c_s = c//2
T = 20
days = 12

# Add variables
z = {}
g = PatrolProblem(T, r, c, r_s, c_s,obj="min")
g.max_effort=T*days
g.days=days

planner = LookAheadPlanner(days=days)    
#planner.printDatafiles(g, file1, file2)
planner.runExperiments(g, file1, file2, filep)

"""
    #.get(GRB.DoubleAttr.X);
g.graph.plotHeatGraph(solx1)
g.graph.plotHeatGraph(solx2)

g.plotHeatMap(solx1)
g.plotHeatMap(solx2)
#g.plotPurePatrols(solf1)
g.plotPatrol(solf1)
"""
