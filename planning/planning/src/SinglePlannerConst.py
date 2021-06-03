'''
Created on Oct 22, 2017
Updated on March 19, 2018

@author: Sara
'''
from gurobipy import *
import numpy as np
import time
from PatrolProblem import PatrolProblem


file1='../QENP_AnimalNonCom/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_detect_2.csv'
#file1='/Users/Sara/Documents/Euler/PathPlanning/MFNP_AnimalNonCom/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_detect.csv'
#'/Users/Sara/Documents/Euler/PathPlanning/QENP_AnimalNonCom/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_attack.csv'
file2='../QENP_AnimalNonCom/PatrolPosts.csv'
#file2='./MFNP_AnimalNonCom/PatrolPosts.csv'

"""
r = 10
c = 10
n = r*c
r_s = r //2
c_s = c//2
T = 10
days = 15

start = time.time()
m = Model()

count_patrols=1
"""
class SinglePlannerConst(object):
    def __init__(self, graph=None, days=15, T=10):
        self.r = 10
        self.c = 10
        self.n = self.r*self.c
        self.r_s = self.r //2
        self.c_s = self.c//2
        self.T = T
        self.days = days
        self.count_patrols=1



    
    #Generates Flow Variables and Constraints for a nxn grid graph
    def genPatrolVars(self, m, effort):
        x = {}
        for i in range(self.n):
            x[i] = m.addVar(lb=0, ub = 1, vtype=GRB.INTEGER, name="x%d-%d" % (self.count_patrols,i))
        
        m.update()
        
        
        # Add constraints
        print(self.T)
        print(self.days)
       # print(x[0]*5)
       # for i in range (self.n):
       #     a=[10,20,30,40]
       #     b=5
       #     sumValue = quicksum(x[i]*a)
       #     print(sumValue)
        m.addConstr(self.T*self.days >= quicksum(x[j]*effort[j] for j in range(self.n)))
        m.update()

        return x
    
    
    def getSol(self, m, x):
        sol=[0 for i in range(self.n)]
        solf=[[[0 for t in range(self.T)] for j in range(self.n)] for i in range(self.n)]
        if m.status == GRB.Status.OPTIMAL:
            obj = m.getAttr('ObjVal')
            solx = m.getAttr('x', x)
        for i in range(self.n):
            sol[i]=solx[i]
        return sol, obj
    
    def setFixedObjectiveandSolve(self, m, data, x):
        obj = []
        m.setObjective( 0.0, GRB.MAXIMIZE)
        
        for i in range(self.n):
            obj.append( (x[i]*data[i]) )
        m.setObjective( quicksum(obj), GRB.MAXIMIZE)

        m.update()
        #m.write("out.lp")
        m.optimize()
        #m.write("out.sol")
        
    def setObjectiveandSolve(self, m, data, effort_level, x):
        obj = []
        m.setObjective( 0.0, GRB.MAXIMIZE)
        
        for i in range(self.n):
            obj.append( (x[i]*data[i][effort_level]) )
        m.setObjective( quicksum(obj), GRB.MAXIMIZE)

        m.update()
        #m.write("out.lp")
        m.optimize()
        #m.write("out.sol")
     
    def solvePost(self, g, file1, file2, i, vmax=None):   
        start = time.time()
        m = Model()
        data, effortx = g.loadFile1D(file1, file2, i)
        self.n=g.n
    
    
        # Add variables
        #x = self.genPatrolVars(m, g.graph)
        x= self.genPatrolVars(m,effortx)
        m.update() 
            
        obj = self.setObjectiveandSolve(m, data, effortx, x)
        end = time.time()
        totaltime = end-start
            
        solx1 , obj = self.getSol(m, x)
        soldata = [i, totaltime, obj]
        soldata.extend(solx1)
        g.graph.writeSolution("runMFP.csv", soldata)
        g.graph.plotHeatGraph(solx1, i, vmax=vmax)    
        #g.graph.piecewisefn(solx1, effortx, data, i) 
     
    def solveFixedPost(self, g, data, effortx, i, vmax=None):   
        start = time.time()
        m = Model()
    
        self.n=g.n
    
        # Add variables
        x = self.genPatrolVars(m, effortx)
        m.update() 
            
        obj = self.setFixedObjectiveandSolve(m, data, x)
        end = time.time()
        totaltime = end-start
            
        solx1 , obj = self.getSol(m, x)
        soldata = [i, totaltime, obj]
        soldata.extend(solx1)
        solx1 = [sol*effortx for sol in solx1]
        g.graph.plotDataHeatGraph(solx1, data, i, name="cons_no_paths"+str(effortx))
        g.graph.plotHeatGraph(solx1, i, name="const_no_paths"+str(effortx))
        g.graph.plotUniformDataPredictions(data, i, effortx)
        
    
        return obj
    def solvePostFromSol(self, g, data, elevel, effortx, i, vmax=None):   
        start = time.time()
        m = Model()
    
        self.n=g.n
    
        # Add variables
        x = self.genPatrolVars(m, effortx[0][elevel])
        m.update() 
            
        obj = self.setObjectiveandSolve(m, data, elevel, x)
        end = time.time()
        totaltime = end-start
            
        solx1 , obj = self.getSol(m, x)
        soldata = [i, totaltime, obj]
        soldata.extend(solx1)
        solx1 = [sol*effortx[0][elevel] for sol in solx1]
        g.graph.plotObjHeatGraph(solx1, effortx, data, i, name="cons_no_paths"+str(effortx[0][elevel]))
        g.graph.plotHeatGraph(solx1, i, name="const_no_paths"+str(effortx[0][elevel]), vmax=vmax)
        g.graph.plotUniformPredictions(data, i, elevel, effortx)
        
    
        return obj
    
    def runExperiments(self, g, file1, file2):
        
        elevel = 3
        for i in range(1):
            m = Model()
            data, effortx = g.loadFile1D(file1, file2, i, points=6)
            self.n=g.n
    
    
            # Add variables
            x = self.genPatrolVars(m, effortx[0][elevel])
            m.update() 
            start = time.time()
            print len(effortx[0])

            obj = self.setObjectiveandSolve(m, data, elevel, x)
            end = time.time()
            totaltime = end-start
            solx1 ,  obj = self.getSol(m, x)
            soldata = [i, totaltime, obj]
            soldata.extend(solx1)

            g.graph.plotObjHeatGraph(solx1, effortx, data, i, name="cons_no_paths"+str(effortx[elevel]))
            g.graph.plotHeatGraph(solx1, i, name="const_no_paths"+str(effortx[elevel]))
    

            m.terminate()

if __name__ == "__main__":
        
    r = 20
    c = 20
    n = r*c
    r_s = r //2
    c_s = c//2
    T = 10
    days = 15
      
    g = PatrolProblem(T, r, c, r_s, c_s,obj="max")
    g.max_effort=T*days
    g.days=days
    
    planner = SinglePlannerConst(days=days)    
    planner.solvePost(i=17,g=g,file1=file1,file2=file2)
    planner.runExperiments(g, file1, file2)
    
    """
    g = PatrolProblem(T, r, c, r_s, c_s,obj="max")
    g.max_effort=T*days
    g.days=days
    
    #data, effortx = g.loadFile1(file1, file2, 3)
    data, effortx = g.loadFileGrid(file1, file2, 3)
    #g.plotData(r_s*c+c_s)
    n=g.n
    
    
    # Add variables
    x,f = genPatrolVars(m)
    #x,f = genPatrolVarsfromGraph(m, g.graph)
    m.update() 
    
    
    #wx = genSOSVars(effortx, x)
    setObjectiveandSolve(m, effortx, x)
    
    end = time.time()
    print "Time: ", end-start
    
    solx1 , solf1 = getSol(m, x, f)
    g.writeSolution("test.csv", solx1)
    g.graph.plotHeatGraph(solx1)
    g.plotHeatMap(solx1)
    g.plotPatrol(solf1)
    """


