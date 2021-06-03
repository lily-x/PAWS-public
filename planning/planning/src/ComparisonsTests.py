'''
Created on Apr 7, 2018

@author: Sara
'''

from SinglePlannerConst import SinglePlannerConst
from SinglePlannerNoPath import SinglePlannerNoPath
from SinglePlanner import SinglePlanner
from PatrolProblem import PatrolProblem

if __name__ == "__main__":

    file1='../QENP_AnimalNonCom/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_detect_2.csv'
    file1='./GP_detect_data.csv'
    file2='../QENP_AnimalNonCom/PatrolPosts.csv'

    r = 12
    c = 12
    n = r*c
    r_s = r //2
    c_s = c//2
    T = 10
    days = 12
      
    g = PatrolProblem(T, r, c, r_s, c_s,obj="max")
    g.max_effort=T*days
    g.days=days
    i=16
    elevel = 4
    
    data, effortx = g.loadFile1D(file1, file2, i, points=10,ceiling=False)
    g.graph.plotGraph()
    planner = SinglePlanner(days=days, T=T)    
    plannernp = SinglePlannerNoPath(days=days, T=T)
    plannerc = SinglePlannerConst(days=days, T=T)    
    
    obj, vmax = planner.solvePost(i, g, data, effortx)
    obj1 = plannernp.solvePostFromSol(g, data, effortx, i)
    
    print obj
    print obj1
    objs = [obj, obj1]
    for elevel in range(1,10):
        g.graph.plotUniformPredictions(data, i, elevel, effortx)

        obj2 = plannerc.solvePostFromSol(g, data, elevel, effortx, i, vmax)

        objs.append(obj2)
        print effortx[0][elevel], obj2
    #for j in range(len(g.effort1)):
    #    obj2=0
    #    if g.effort1[j]==3.4:   
    #        obj2 = plannerc.solveFixedPost(g, g.alldata[:][j], 3.4, i, vmax)

    #    objs.append(obj2)
        
    for o in objs:
        print o

