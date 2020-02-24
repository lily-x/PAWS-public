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
file3='./queen_var_detect.csv'
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
class SinglePlannerNoPath(object):
    def __init__(self, graph=None, days=15, T=10):
        self.r = 10
        self.c = 10
        self.n = self.r*self.c
        self.r_s = self.r //2
        self.c_s = self.c//2
        self.T = T
        self.days = days
        self.count_patrols=1


    #Generate all the variables and constraints for the optimization
    def genSOSVars(self, m, x_vals, x):
        wx =[[0 for i in range(len(x_vals[j]))] for j in range(self.n)]
        for k in range(self.n):
            for i in range(len(x_vals[k])):
                wx[k][i] = m.addVar(lb=0, ub = 1.0, vtype=GRB.CONTINUOUS, name="wx%d-%d" %(k,i))
        m.update() 
        
        
        for i in range(self.n):
            m.addConstr(x[i] == quicksum( wx[i][j]*x_vals[i][j] for j in range(len(x_vals[i]))))
            m.addConstr(1 == quicksum( wx[i][j] for j in range(len(x_vals[i]))))
    
        for k in range(self.n):
            if len(x_vals)>1:
                m.addSOS(GRB.SOS_TYPE2, wx[k])#,x_vals)
        m.update() 
    
        return wx
    
    #Generates Flow Variables and Constraints for a nxn grid graph
    def genPatrolVars(self, m):
        x = {}
        f = [[[0 for t in range(self.T)] for j in range(self.n)] for i in range(self.n)]
        for i in range(self.n):
            x[i] = m.addVar(lb=0, ub = 10000, vtype=GRB.CONTINUOUS, name="x%d-%d" % (self.count_patrols,i))
        
        
        m.update()
        
        
        # Add constraints
        m.addConstr(self.T*self.days >= quicksum( x[j] for j in range(self.n)))
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
    
    def setConstantObjectiveandSolve(self, m, data, effort_level, x):
        obj = []
        m.setObjective( 0.0, GRB.MAXIMIZE)
        
        for i in range(self.n):
            obj.append( (x[i]*data[i][effort_level]) )
        m.setObjective( quicksum(obj), GRB.MAXIMIZE)

        m.update()
        #m.write("out.lp")
        m.optimize()
        #m.write("out.sol")
    
    def setObjectiveandSolve(self, m, data, effortx, x):
        obj = []
        #for k in range(self.n):    
        #    for i in range(len(effortx[k])):
        #       pass#obj.append(wx[k][i]*data[k][i])
        # Set objective
        #m.setObjective( quicksum(obji for obji in obj), GRB.MAXIMIZE)
        m.setObjective( 0.0, GRB.MAXIMIZE)
        
        for i in range(self.n):
            if len(effortx[i]) >1:
                m.setPWLObj(x[i], effortx[i], data[i][:len(effortx[i])])
        
        m.update()
        #m.write("out.lp")
        m.optimize()
        #m.write("out.sol")
    
    def solvePost(self, g, file1, file2, i, vmax=None):   
        start = time.time()
        m = Model()
        data, effortx = g.loadFile1D(file1, file2, i) #changed loadFile1 to loadFile1D
        self.n=g.n
    

        # Add variables
        x = self.genPatrolVars(g.graph,m) 
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
        
    def solvePostFromSol(self, g, data, effortx, i, vmax=None):   
        start = time.time()
        m = Model()
    
        self.n=g.n
    
        # Add variables
        x = self.genPatrolVars(m)
        m.update() 
            
        obj = self.setObjectiveandSolve(m, data, effortx, x)
        end = time.time()
        totaltime = end-start
            
        solx1 , obj = self.getSol(m, x)
        soldata = [i, totaltime, obj]
        soldata.extend(solx1)
        
        g.graph.writeSolution("qeen_4.csv", soldata)
        g.graph.plotObjHeatGraph(solx1, effortx, data, i, name="no_paths", vmax=vmax, objective=obj)
        g.graph.plotHeatGraph(solx1, i, name="no_paths", vmax=vmax)
        #g.graph.piecewisefn(solx1, effortx, data, i) 
        
        return obj
    
    def runExperiments(self, g, file1, file2):
        
        for i in range(1):
            m = Model()
            data, effortx = g.loadFile1D(file1, file2, i, points=4)
            self.n=g.n
    
    
            # Add variables
            x = self.genPatrolVars(m)
            m.update() 
            start = time.time()
            print(len(effortx[0]))

            obj = self.setObjectiveandSolve(m, data, effortx, x)
            end = time.time()
            totaltime = end-start
            solx1 ,  obj = self.getSol(m, x)
            soldata = [i, totaltime, obj]
            soldata.extend(solx1)

            g.graph.writeSolution("qeen_4.csv", soldata)
            g.graph.plotObjHeatGraph(solx1, effortx, data, i, name="no_paths")
            g.graph.plotHeatGraph(solx1, i, name="no_paths")
            g.graph.plotUniformPredictions(data, i, 0, effortx)
            g.graph.plotUniformPredictions(data, i, 1, effortx)
            g.graph.plotUniformPredictions(data, i, 2, effortx)
            g.graph.plotUniformPredictions(data, i, 3, effortx)  

            m.terminate()

if __name__ == "__main__":
        
    r = 10
    c = 10
    n = r*c
    r_s = r //2
    c_s = c//2
    T = 10
    days = 15
      
    g = PatrolProblem(T, r, c, r_s, c_s,obj="max")
    g.max_effort=T*days
    g.days=days
    i=16
    elevel = 4
    
    data, effortx = g.loadFile1D(file1, file2, i, points=10,ceiling=False)
    var, effortx = g.loadFile1D(file3, file2, i, points=10,ceiling=False)

    planner = SinglePlannerNoPath(days=days)    
    planner.solvePostFromSol(i=16, g=g, data=data, effortx=effortx)
    #planner.solvePost(i=16,file1=file1,file2=file2,g=g)
    #planner.runExperiments(g, file1, file2)
    
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


