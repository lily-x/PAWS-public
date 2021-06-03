'''
Created on Oct 25, 2017

@author: Sara
'''

import sys
sys.path.insert(0, '../library/piecewise-master')

from gurobipy import *
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

from matplotlib import colors
from matplotlib_scalebar.scalebar import ScaleBar

import csv
import pandas as pd
import time
import re
import random
from pwlf.pwlf import piecewise_lin_fit
import scipy
from piecewise.regressor import piecewise
from numpy import dtype
import csv

class PatrolProblem(object):

    def __init__(self, days=15, T=10, r=10, c=10, r_s=4, c_s=4, obj="max", resolution=None, method=None):
        self.T = T
        self.r = r
        self.c = c
        self.r_s = r_s
        self.c_s = c_s
        self.n = r*c
        self.resolution = resolution
        self.default = 0
        if not obj == "max":
            self.default = 1
        self.max = self.resolution*r / 2
        self.max_effort = T*days
        self.days = days
        self.method = method

    #gets the data in range self.max from patrol post
    def loadGrid(self, df, post_x, post_y):
        df = df[(df['x'] >= post_x-self.max)
                & (df['x'] <= post_x+self.max)
                & (df['y'] >= post_y-self.max)
                & (df['y'] <= post_y+self.max)]
        self.n = len(df.index)

        self.px = np.floor(post_x/self.resolution)*self.resolution
        self.py = np.floor(post_y/self.resolution)*self.resolution
        source = -1
        srcrow = None
        dist = 1000000
        i = 0
        for index, row in df.iterrows():
            new_dist = (np.abs(row['x']-self.px)+np.abs(row['y']-self.py))
            if new_dist < dist:
                dist = new_dist
                source = i
                srcrow = row
            i+=1
        self.row=srcrow
        return df, source

    #loads the data for a 2D model
    def loadFile2D(self, filename, postloc_file, post, points=0, datafolder="./kai_data/test/GP_"):

        df = pd.read_csv(filename)
        posts = pd.read_csv(postloc_file)
        df, src = self.loadGrid(df, posts['X'][post], posts['Y'][post])
        #df, row = self.loadGrid(df, 125000, 9920000)

        #row = df.index.get_loc(d.iloc[:].name)

        #df=df[df.columns[5:]]
        self.graph = PatrolGraph(df, self.resolution, src, datafolder, self.method)
        df = self.graph.data
        self.c = int(np.max(df['x']))
        self.r = int(np.max(df['y']))

        cols = [col for col in df.columns if 'Y_test' in col]
        df = df[cols]
        df.rename(columns=lambda x: tuple(re.findall(r"[-+]?\d*\.\d+|\d+",x)), inplace=True)
        #df.rename(columns = lambda x: x[0]*self.max_e+x[1], inplace=True)
        effortx1 = []
        effortx2 = []
        for col in df.columns:
            ix1 = float(col[0])
            ix2 = float(col[1])
            if ix1 not in effortx1: effortx1.append(ix1)
            if ix2 not in effortx2: effortx2.append(ix2)
        if 0.0 not in effortx1: effortx1.append(0.0)
        if 0.0 not in effortx2: effortx2.append(0.0)
        if self.max_effort not in effortx1: effortx1.append(self.max_effort)
        if self.max_effort not in effortx2: effortx2.append(self.max_effort)

        effortx1.sort()
        effortx2.sort()
        effortx1_mask = []
        effortx2_mask = []
        data = [[[0 for i in range(len(effortx2)) ]for j in range(len(effortx1)) ]for k in range(self.n)]
        for col in df.columns:
            for k in range(self.n):
                ix1 = float(col[0])
                ix2 = float(col[1])
                i=effortx1.index(ix1)
                j=effortx2.index(ix2)
                d = df[col]
                data[k][i][j] = d.iloc[k]
        for k in range(self.n):
            for i in range(len(effortx1)):
                data[k][i][-1] = data[k][i][-2]

            for j in range(len(effortx2)):
                data[k][-1][j] = data[k][-2][j]
        for index, row in self.graph.data.iterrows():
            en = []
            sp = self.distFromPostxy(row, self.graph.data.iloc[src])
            d = 10000000 #self.days*(self.T-2*sp)#self.distFromPost(k))
            if d < 0 : d=0
            for i in effortx1:
                if d >= i: en.append(i)
            if not effortx1[-1] in en:
                en.append(effortx1[-1])
            effortx1_mask.append(en)
            en = []
            for i in effortx2:
                if d >= i : en.append(i)
            if not effortx2[-1] in en:
                en.append(effortx2[-1])
            effortx2_mask.append(en)
        datanew = [[[0 for j in range(len(effortx2_mask[k])) ]for i in range(len(effortx1_mask[k])) ]for k in range(self.n)]
        for k in range(self.n):
            for i in range(len(effortx1_mask[k])):
                for j in range(len(effortx2_mask[k])):
                    datanew[k][i][j]= data[k][i][j]

        self.data = datanew
        self.effort1 = effortx1
        self.effort2 = effortx2
        self.effortx1_mask = effortx1_mask
        self.effortx2_mask = effortx2_mask


        if not points == 0:
            self.datanew, self.effortx1_mask, self.effortx2_mask = self.reducePoints2D(data, effortx1_mask, effortx2_mask, points)
        #self.datanew[:]
        b = np.reshape(np.asarray(self.datanew).transpose((0,2,1)), (self.n, -1))
        s = "5_step_data1_%s.csv" % post
        np.savetxt(s, b, delimiter=",")


        return self.datanew, self.effortx1_mask, self.effortx2_mask

    #loads the data for a 1D model
    def loadFile1D(self, filename, postloc_file, post, points=0, ceiling=False, datafolder='./kai_data/test/GP_'):
        df = pd.read_csv(filename)
        posts = pd.read_csv(postloc_file)
        df, srow = self.loadGrid(df, posts['X'][post], posts['Y'][post])

        self.graph = PatrolGraph(df, self.resolution, srow, datafolder, self.method)

        df = self.graph.data
        self.c = int(np.max(df['x']))
        self.r = int(np.max(df['y']))
        cols = [col for col in df.columns if 'Y_test' in col]
        df = df[cols]
        df.rename(columns=lambda x: tuple(re.findall(r"[-+]?\d*\.\d+|\d+",x)), inplace=True)

        effortx1 = []
        diff=0
        for col in df.columns:
            ix1 = float(col[0])
            if ix1 not in effortx1: effortx1.append(ix1)
        diff = effortx1[1]-effortx1[0]
        effortx1 = [effort+diff for effort in effortx1]
        print(effortx1)

        if 0.0 not in effortx1: effortx1.append(0.0)
        if self.max_effort not in effortx1: effortx1.append(self.max_effort)

        effortx1.sort()
        effortx1_mask = []
        data = [[0 for j in range(len(effortx1)) ]for k in range(self.n)]
        for col in df.columns:
            for k in range(self.n):
                ix1 = float(col[0])+diff
                i = effortx1.index(ix1)
                try:
                    d = df[col]
                except KeyError:
                    print(k, col)
                data[k][i] = d.iloc[k]
        for index, row in self.graph.data.iterrows():
            en = []
            #sp = self.distFromPostxy(row)
            d = 10000000#(self.T-2*sp)*self.days
            if d < 0 : d=0
            for i in effortx1:
                if d >= i: en.append(i)
            if not effortx1[-1] in en:
                en.append(effortx1)
            effortx1_mask.append(en)
        for k in range(self.n):
            data[k][-1]=data[k][-2]

        self.alldata = data
        self.data = data
        self.effort1 = effortx1
        self.effortx1_mask = effortx1_mask

        if ceiling:
            self.data = self.converToMax(data)
            self.alldata = self.data

        if not points == 0:
            self.data, self.effortx1_mask = self.reducePoints1D(self.data, effortx1_mask, points)

        return self.data, self.effortx1_mask

    def plotAllData(self,n):
        plt.plot(self.effort1,self.alldata[n][:])
        plt.plot(self.effortx1_mask[n][:], self.data[n][:])
        plt.show()

    def reduceOptPoints1D(self, data, effort1, points):
        reduced_data = [[[0 for i in range(points)] for j in range(points)]for k in range(self.n)]
        effortx1_mask = []
        effortx2_mask = []

        breaks1 = [[ 0 for i in range(points)] for j in range(self.n)]

        for k in range(self.n):
            x = effort1[k]
            y = data[k][:]

            breaks1[k], yhat, r = self.fitBins(x ,y , points)



            effortx1_mask.append(x)
            reduced_data[k] = yhat

        return reduced_data, effortx1_mask

    def reducePoints2D(self, data, effort1, effort2, points):
        reduced_data = [[[0 for i in range(points)] for j in range(points)]for k in range(self.n)]
        effortx1_mask = []
        effortx2_mask = []
        for k in range(self.n):
            c1 = np.linspace(0,len(effort1[k])-1,points, dtype=np.int16)
            c2 = np.linspace(0,len(effort2[k])-1,points, dtype=np.int16)
            #effortx1_mask.append(c1)
            #effortx2_mask.append(c2)
            i1 = 0
            j1 = 0
            e1 =[]
            e2 = []
            for i in c1:
                e1.append(effort1[k][i])
                e2.append(effort2[k][i])

                for j in c2:
                    reduced_data[k][i1][j1] = data[k][i][j]
                    j1 += 1
                j1 = 0
                i1 += 1
            effortx1_mask.append(e1)
            effortx2_mask.append(e2)

        return reduced_data, effortx1_mask, effortx2_mask
    def reducePoints1D(self, data, effort1, points):
        reduced_data = [[ 0 for j in range(points)]for k in range(self.n)]
        effortx1_mask = []
        for k in range(self.n):
            c1 = np.linspace(0,len(effort1[k])-1,points, dtype=np.int16)
            i1 = 0
            effort = []

            for i in c1:
                effort.append(effort1[k][i])

                reduced_data[k][i1] = data[k][i]
                i1 += 1
            effortx1_mask.append(effort)
        return reduced_data, effortx1_mask

    def reduceOptPoints(self, data, effort1, effort2, points):
        reduced_data = [[[0 for i in range(points)] for j in range(points)]for k in range(self.n)]
        effortx1_mask = []
        effortx2_mask = []

        for k in range(self.n):
            breaks1 = [[ 0 for i in range(points)] for j in range(len(effort1[k]))]
            breaks2 = [[ 0 for i in range(points)] for j in range(len(effort2[k]))]

            for e in range(len(effort1)):
                x = effort2[k]
                y = data[k][e][:]
                breaks1[e], r = self.fitBins(x ,y , points)
            for e in range(len(effort2)):
                x = effort1[k]
                y = data[k][:][e]
                breaks2[e], r = self.fitBins(x ,y , points)

            c1 = np.linspace(0,len(effort1[k])-1,points, dtype=np.int16)
            c2 = np.linspace(0,len(effort2[k])-1,points, dtype=np.int16)
            effortx1_mask.append(c1)
            effortx2_mask.append(c2)
            i1 = 0
            j1 = 0
            for i in c1:
                for j in c2:
                    reduced_data[k][i1][j1] = data[k][i][j]
                    j1 += 1
                j1 = 0
                i1 += 1
        return reduced_data, effortx1_mask, effortx2_mask

    def fitBins(self, x, y, segments):
        #   initialize piecwise linear fit with your x and y data
        x = np.asarray(x[:len(x)-1])
        y = np.asarray(y[:len(y)-1])
        myPWLF = piecewise_lin_fit(x,y)

        #   fit the data for four line segments
        res = myPWLF.fit(segments)
        breaks = myPWLF.fitBreaks

        #   predict for the determined points
        #xHat = np.linspace(min(x), max(x), num=10000)
        yHat = myPWLF.predict(breaks)

        #   plot the results
        #plt.figure()
        #plt.plot(x,y,'o')
        #plt.plot(xHat,yHat, '-')
        #plt.show()
        return breaks, yHat, res

    def plotHeatMap(self, sol):
        #img = np.transpose(np.asarray(sol).reshape(self.r,self.c))
        img = [[ sol[i*self.c+j] for j in range(self.c)] for i in range(self.r)]
        max_effort = np.max(img)
        print(max_effort)
        s = 10
        #plt.imshow(np.flipud(img), interpolation='none', cmap="Greens", extent=[0,self.c,0,self.r])
        plt.imshow(np.flipud(img), interpolation='none', cmap="Reds", extent=[0,self.c,0,self.r])
        plt.colorbar()
        plt.xticks(np.arange(0,self.n))
        plt.yticks(np.arange(0,self.n))
        plt.grid(ls='solid')
        plt.scatter([self.c_s+0.5],[self.r_s+0.5])
        plt.show()

    def plotTimeGraph(self, solf):
        for t in range(self.T):
            for i in range(self.n):
                for j in range(self.n):
                    if(solf[i][j][t]>0):
                        plt.plot([t,t+1],[i,j])
        plt.xlabel("Time")
        plt.ylabel("Grid-cell")
        step = 10
        #l = [ "(%d,%d)" %((k-k%c)/c,k%c) for k in range(n)]
        plt.yticks(np.arange(0,self.n,step),[ "(%d,%d)" %((k-k%self.c)/self.c,k%self.c) for k in range(self.n)][0::11])
        plt.show()

    def plotPurePatrols(self, solf):
        offset = -0.4
        sol = np.asarray(solf)
        #colors = ["blue", "red", "green"]
        flow = np.min(sol[np.nonzero(sol)])
        while flow > 0.0001:
            flow = np.min(sol[np.nonzero(sol)])
            #t=0
            i = self.r_s*self.c+self.c_s
            for t in range(self.T):
                min = 100
                jmin=-1
                for j in range(self.n):
                    s = sol[i][j][t]
                    if s>0 and s < min : jmin = j
                plt.plot([t,t+1],[i+offset,jmin+offset], color="red")
                sol[i][jmin][t] = sol[i][jmin][t] - flow
                i = jmin

            offset += 0.2
        plt.xlabel("Time")
        plt.ylabel("Grid-cell")
        step = 10
        #l = [ "(%d,%d)" %((k-k%c)/c,k%c) for k in range(n)]
        #plt.yticks(np.arange(0,self.n,step),[ "(%d,%d)" %((k-k%self.c)/self.c,k%self.c) for k in range(self.n)][0::11])
        plt.show()

    def plotPatrol(self, solf):
        for t in range(self.T):
            for i in range(self.n):
                for j in range(self.n):
                    if(solf[i][j][t]>0):
                        x1=(i-i%self.c)/self.c
                        y1=i%self.c
                        x2=(j-j%self.c)/self.c
                        y2=j%self.c
                        plt.plot([x1,y1],[x2,y2])
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.show()

    def distFromPost(self, i):
        x1 = np.abs((i - i % self.r) / self.r - self.c_s)
        y1 = np.abs(i % self.r - self.r_s)

        return x1+y1

    def distFromPostxy(self, row, srcrow):
        x = row['x']
        y = row['y']
        xs = srcrow['x']
        ys = srcrow['y']
        x1 = np.abs(x-xs)
        y1 = np.abs(y-ys)

        return x1+y1



    def plotData(self, n):
        if len(np.shape(self.data))==2:
            plt.plot(self.effort1,self.data[n][:])
            plt.show()
            return

        for i in range(len(self.effortx2_mask[n])):
            plt.plot(self.effortx1_mask[n],self.data[n][:][i])
        plt.show()

    def plotDataP(self, n, past):
        plt.plot(self.effort1,self.data[n][:][past])
        plt.show()

    def writeSolution(self, file, sol):
        with open(file, 'a') as csvfile:
            w = csv.writer(csvfile)

            #for i in range(len(sol)):
            strlst = [str(s) for s in sol]
            w.writerow(strlst)

    def converToMax(self, data):
        return [[ np.max(line[:i+1]) for i in range(len(line)) ] for line in data]


#datafolder = "./data/15/GP_"
plot_title = True

class PatrolGraph(object):
    nodes = []
    edges = []

    def __init__(self, data, step, row, datafolder, method=None):
        self.datafolder = datafolder

        self.epsilon = 0.001
        self.resolution = step
        self.min_xval = np.min(data['x'])
        self.min_yval = np.min(data['y'])
        data['x'] = (data['x'] - self.min_xval)/step
        data['y'] = (data['y']- self.min_yval)/step
        data['ID'] = pd.Series(range(len(data['x'])), index=data.index)

        self.data = data
        self.source = row
        self.method = method
        #self.plotGraph()

    def neighbors(self,n):
        x=self.data['x'].values[n]
        y=self.data['y'].values[n]

        df = self.data[((self.data['x'] == x+1)  & (self.data['y'] == y))
                       |((self.data['x'] == x-1) & (self.data['y'] == y))
                       |((self.data['x'] == x) & (self.data['y'] == y+1))
                       |((self.data['x'] == x) & (self.data['y'] == y-1))]
                       #|((self.data['x'] == x-1) & (self.data['y'] == y-1))
                       #|((self.data['x'] == x+1) & (self.data['y'] == y+1))]
        results = df['ID'].values#self.data.index.get_loc(df[0])
        return results

    def piecewisefn(self, x, effort, data, post):
        #cols = [col for col in data.columns if 'Y_test' in col]
        preds = []
        for n in range(len(x)):
            if x[n] <= effort[n][0]:
                preds.append(data[n][0])
            else:
                for i in range(0,len(effort[n])-1):
                    #print x[n] > effort[n][i]
                    #print x[n], effort[n][i], effort[n][i+1]
                    dt = np.abs(x[n]-effort[n][i+1])
                    #print dt
                    if ((x[n] > effort[n][i]) and (x[n] <= effort[n][i+1] or dt <= 0.001 )):
                        m = (x[n]-effort[n][i])/(effort[n][i+1]-effort[n][i])
                        m = m*(data[n][i+1]-data[n][i])+data[n][i]
                        preds.append(m)
                        break
                #clist.append((x[n] > effort[n][i] and x<effort[n][i+1]))
            #clist.append(x[n]>effort[n][-1])
            #preds.append(np.piecewise(x[n], clist, data[n][:len(effort[n])]))

        if False:
            with open("preds.csv", 'a') as csvfile:
                w = csv.writer(csvfile)
                header = ["row number", "post", "ID-Global", "x", "pred"]
                w.writerow(header)
                for n in range(len(x)):
                   row = [str(n), str(post), str(self.data.iloc[n]["ID_Global"]), str(x[n]), str(preds[n])]
                   w.writerow(row)
        return preds

    def plotPatrol(self, path, post=None, numpath=None, plot=None):
        mx=int(np.max(self.data['x']))
        my=int(np.max(self.data['y']))
        grid = [[0 for x in range(mx+1)] for y in range(my+1)]

        patrolx = []
        patroly = []
        for p in path:
            x = int(self.data.iloc[p]["x"])
            y = int(self.data.iloc[p]["y"])
            grid[y][x]=1
            patrolx.append(x)
            patroly.append(y)
        if plot == None:
            plt.imshow(np.flipud(grid), interpolation='none', cmap="Reds", extent=[0,mx,0,my])
        plt.plot(patrolx,patroly, linewidth=2, color="black")
        if not post == None:
            plt.title(self.datafolder + "Patrol Effort for Patrol Post %d" % (post))
            plt.xticks(np.arange(0,mx+1),[self.min_xval+self.resolution*i for i in range(mx+1)], rotation=60)
            plt.xlabel("x", fontsize=12)
            plt.yticks(np.arange(0,my+1),[self.min_yval+self.resolution*i for i in range(my+1)])
            plt.ylabel("y", fontsize=12)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            #plt.savefig("patrol_effort_map_post_%d_path_%d"%(post,numpath))
            #plt.close()
            #plt.show()
        else:
            plt.show()

    def plotGraph(self):
        mx=int(np.max(self.data['x']))
        my=int(np.max(self.data['y']))
        grid = [[0 for x in range(mx+1)] for y in range(my+1)]
        print(grid)

        color = ['black',(0,0,0,0)]
        cmapm = colors.ListedColormap(color)

        i=0
        for index, row in self.data.iterrows():
            grid[int(row['y'])][int(row['x'])]=1

            if i == self.source:
                sy=int(row['y'])
                sx=int(row['x'])

            i+=1
        #np.invert(gridmap)
        grid = np.ma.masked_where(grid ==1, grid)

        if np.min(grid) == 0:
            h = plt.imshow(np.flipud(grid), interpolation='none', cmap=cmapm, extent=[0,mx+1,0,my+1])
        #plt.set(h, 'AlphaData', gridmap)


        plt.xticks(np.arange(0,mx+1))
        plt.yticks(np.arange(0,my+1))
        plt.grid(ls='solid')
        if plot_title:
            plt.xticks(np.arange(0,mx+1),[self.min_xval+self.resolution*i for i in range(mx+1)], rotation=60)
            plt.xlabel("x", fontsize=12)
            plt.yticks(np.arange(0,my+1),[self.min_yval+self.resolution*i for i in range(my+1)])
            plt.ylabel("y", fontsize=12)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.plot([sx+0.5], [sy+0.5], marker='o', markersize=5, color="blue")

        plt.savefig(self.datafolder + "patrol_map_post.png")
            #plt.show()
        plt.plot([sx+0.5], [sy+0.5], marker='o', markersize=5, color="blue")

        plt.close()

    def plotDataHeatGraph(self, sol, data, post, name, vmax=None):
        self.plotHeatGraph(data, post, name+"objective", vmax=1.0, cmap="Blues")

    def plotObjHeatGraph(self, sol, effortx, data, post, name, vmax=None, objective=None):
        objvals = self.piecewisefn(sol, effortx, data, post=post)
        self.plotHeatGraph(objvals, post, name+"objective", vmax=1.0, cmap="Blues", objective=objective)

    def plotHeatGraph(self, sol, post=None, name="", vmax=None, cmap="Reds", objective=None):

        mx=int(np.max(self.data['x']))
        my=int(np.max(self.data['y']))

        color = ['black',(0,0,0,0)]
        cmapm = colors.ListedColormap(color)
        bounds=[-1,0]
        norm = colors.BoundaryNorm(bounds, cmapm.N)

        gridmap = [[0 for x in range(mx+1)] for y in range(my+1)]
        grid = [[0 for x in range(mx+1)] for y in range(my+1)]
        i=0
        for index, row in self.data.iterrows():
            if i == self.source:
                sy=int(row['y'])
                sx=int(row['x'])
            gridmap[int(row['y'])][int(row['x'])]=1

            grid[int(row['y'])][int(row['x'])]=sol[i]


            i+=1
        #np.invert(gridmap)
        gridmap = np.ma.masked_where(gridmap ==1, gridmap)
        if vmax == None:
            a = plt.imshow(np.flipud(grid), interpolation='none', cmap=cmap, extent=[0,mx+1,0,my+1])
        else:
            a = plt.imshow(np.flipud(grid), interpolation='none', cmap=cmap, extent=[0,mx+1,0,my+1], vmin=0, vmax=vmax)
        if np.min(gridmap) == 0:
            h = plt.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[0,mx+1,0,my+1])
        #plt.set(h, 'AlphaData', gridmap)

        plt.colorbar(a)
        plt.xticks(np.arange(0,mx+1))
        plt.yticks(np.arange(0,my+1))
        plt.grid(ls='solid')
        if not post == None:
            if plot_title:
                if objective:
                    plt.title("%s: Patrol Effort, Objective: %f" %(name, objective))
                else:
                    plt.title("Patrol Effort with Post %d %s" %(post, name))

                # plt.xticks(np.arange(0,mx+1),[self.min_xval+self.resolution*i for i in range(mx+1)], rotation=60)
                # plt.xlabel("x", fontsize=12)
                # plt.yticks(np.arange(0,my+1),[self.min_yval+self.resolution*i for i in range(my+1)])
                # plt.ylabel("y", fontsize=12)

            scalebar = ScaleBar(dx=self.resolution, units='m', fixed_value=1, fixed_units='km', location='lower left') # 1 km or 200 m or 500m
            plt.gca().add_artist(scalebar)
            # plt.xticks(fontsize=6)
            # plt.yticks(fontsize=6)
            plt.plot([sx+0.5], [sy+0.5], marker='o', markersize=5, color="blue")

            plt.savefig(self.datafolder + "patrol_effort_map_post_%d_%s.png" % (post, name))
            plt.close()
            #plt.show()
        else:
            plt.show()

    def writeSolution(self, file, sol, sol2=None, bp=None):
        with open(file, 'a') as csvfile:
            w = csv.writer(csvfile)
            header = ["post", "Runtime", "Obj-Val"]
            if not bp == None:
                header.append("levels")
            for index, row in self.data.iterrows():
                i = row["ID_Global"]
            #for i in range(len(sol)):
                header.append(str(i))
            w.writerow(header)
            strlst = [str(s) for s in sol]
            w.writerow(strlst)
            if not sol2 == None:
               strlst = [str(s) for s in sol2]
               w.writerow(strlst)
        #file2="GlobID_"+file
        #with open(file2, 'a') as csvfile:
        #    w = csv.writer(csvfile)
         #   header = ["post", "Runtime", "Obj-Val"]
         #   if not bp == None:
         #       header.append("levels")
         #   for index, row in self.data.iterrows():
         #       i = row["ID-Global"]
            #for i in range(len(sol)):
         #       header.append(str(i))
         #   w.writerow(header)

    def getPath(self, path, t, flow, pathflow):
        k=path[-1]
        edges = []
        if t == len(flow[0][0]):
            if len(pathflow)==0 or t==0:
                print("hi")
            return path, pathflow

        for n in self.neighbors(k):
            if flow[k][n][t] > self.epsilon:
                p2 = path+[n]
                pf2 = pathflow + [flow[k][n][t]]
                return self.getPath(p2,t+1,flow, pf2)
        return path, pathflow

    def samplePatrols(self, sol, days, post=None, name=""):
        paths = []
        pathprob = []
        flow = np.copy(sol)
        print(np.max(sol), np.max(flow))
        path, pf = self.getPath([self.source], 0, flow, [])
        paths.append(path)
        pathprob.append(np.min(pf))
        minpf = np.min(pf)
        for i in range(len(pf)):
                flow[path[i]][path[i+1]][i] -= minpf
        while np.max(flow) > 0.001:
            path, pf = self.getPath([self.source], 0, flow, [])
            print(path)
            paths.append(path)

            if len(pf) ==0:
                print(flow)
                print(np.max(flow))
                print(np.argmax(flow))
                print(pf)

            pathprob.append(np.min(pf))

            minpf = np.min(pf)
            for i in range(len(pf)):
                flow[path[i]][path[i+1]][i] -= minpf
        i=0
        #self.plotHeatGraph(sol, post)

        x = int(np.ceil(np.sqrt(len(paths))))
        i=0
        for path in paths:
            ix = int(i//x)
            iy = int(i%x)
            ax = plt.subplot2grid((x, x), (ix, iy))
            ax.tick_params( axis='both', which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
            self.subPlotPatrol(path, ax, post=post,numpath=i)
            i+=1
        plt.savefig(self.datafolder + "patrol_map_post_%d_%s" % (post, name))
        plt.close()


    def subPlotPatrol(self, path, plt, post=None, numpath=None):
        mx=int(np.max(self.data['x']))
        my=int(np.max(self.data['y']))
        grid = [[0 for x in range(mx+1)] for y in range(my+1)]

        patrolx = []
        patroly = []
        for p in path:
            x = int(self.data.iloc[p]["x"])
            y = int(self.data.iloc[p]["y"])
            grid[y][x]+=1
            patrolx.append(x)
            patroly.append(y)

        grid = np.array(grid) / float(np.max(grid) + 1)

        plt.imshow(np.flipud(grid), interpolation='none', cmap="Reds", extent=[0,mx,0,my])
        plt.plot(patrolx,patroly, linewidth=1.5, color="black")

    def plotUniformPredictions(self, data, post, effort, effortval):
        e = str(effortval[post][effort])
        print(e)
        name = "uniform_{0}".format(e)
        mx=int(np.max(self.data['x']))
        my=int(np.max(self.data['y']))

        color = ['black',(0,0,0,0)]
        cmapm = colors.ListedColormap(color)
        bounds=[-1,0]
        norm = colors.BoundaryNorm(bounds, cmapm.N)

        gridmap = [[0 for x in range(mx+1)] for y in range(my+1)]
        grid = [[0 for x in range(mx+1)] for y in range(my+1)]
        i=0
        for index, row in self.data.iterrows():
            if i == self.source:
                sy=int(row['y'])
                sx=int(row['x'])
            grid[int(row['y'])][int(row['x'])]= data[i][effort]
            gridmap[int(row['y'])][int(row['x'])]=1

            i+=1
        #np.invert(gridmap)
        gridmap = np.ma.masked_where(gridmap ==1, gridmap)
        a = plt.imshow(np.flipud(grid), interpolation='none', cmap="Greens", extent=[0,mx+1,0,my+1], vmin=0, vmax=1)
        if np.min(gridmap) == 0:
            h = plt.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[0,mx+1,0,my+1])
        #plt.set(h, 'AlphaData', gridmap)

        plt.colorbar(a)
        plt.xticks(np.arange(0,mx+1))
        plt.yticks(np.arange(0,my+1))
        plt.grid(ls='solid', lw=0.1)
        if not post == None:
            if plot_title:

                plt.title("Patrol Effort for Patrol Post {0} {1}".format(post,name))
                plt.xticks(np.arange(0,mx+1),[self.min_xval+self.resolution*i for i in range(mx+1)], rotation=60)
                plt.xlabel("x", fontsize=12)
                plt.yticks(np.arange(0,my+1),[self.min_yval+self.resolution*i for i in range(my+1)])
                plt.ylabel("y", fontsize=12)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            #plt.plot([sx+0.5], [sy+0.5], marker='o', markersize=5, color="blue")

            plt.savefig(self.datafolder + "patrol_effort_map_post_{0}_{1}.png".format(post, name))
            plt.close()
            #plt.show()
        else:
            plt.show()

    def plotUniformDataPredictions(self, data, post, effortval):
        e = str(effortval)
        print(e)
        name = "uniform_{0}".format(e)
        mx=int(np.max(self.data['x']))
        my=int(np.max(self.data['y']))

        color = ['black',(0,0,0,0)]
        cmapm = colors.ListedColormap(color)
        bounds=[-1,0]
        norm = colors.BoundaryNorm(bounds, cmapm.N)

        gridmap = [[0 for x in range(mx+1)] for y in range(my+1)]
        grid = [[0 for x in range(mx+1)] for y in range(my+1)]
        i=0
        for index, row in self.data.iterrows():
            if i == self.source:
                sy=int(row['y'])
                sx=int(row['x'])
            grid[int(row['y'])][int(row['x'])]= data[i]
            gridmap[int(row['y'])][int(row['x'])]=1

            i+=1
        #np.invert(gridmap)
        gridmap = np.ma.masked_where(gridmap ==1, gridmap)
        a = plt.imshow(np.flipud(grid), interpolation='none', cmap="Greens", extent=[0,mx+1,0,my+1], vmin=0, vmax=1)
        if np.min(gridmap) == 0:
            h = plt.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[0,mx+1,0,my+1])
        #plt.set(h, 'AlphaData', gridmap)

        plt.colorbar(a)
        plt.xticks(np.arange(0,mx+1))
        plt.yticks(np.arange(0,my+1))
        plt.grid(ls='solid', lw=0.1)
        if not post == None:
            if plot_title:

                plt.title("Patrol Effort for Patrol Post {0} {1}".format(post,name))
                plt.xticks(np.arange(0,mx+1),[self.min_xval+self.resolution*i for i in range(mx+1)], rotation=60)
                plt.xlabel("x", fontsize=12)
                plt.yticks(np.arange(0,my+1),[self.min_yval+self.resolution*i for i in range(my+1)])
                plt.ylabel("y", fontsize=12)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            #plt.plot([sx+0.5], [sy+0.5], marker='o', markersize=5, color="blue")

            plt.savefig(self.datafolder + "patrol_effort_map_post_{0}_{1}.png".format(post, name))
            plt.close()
            #plt.show()
        else:
            plt.show()



    def plotUniformRiskPredictions(self, data, post, effortval):
        effort_size = len(effortval[post])
        name = "uniform_risk"
        mx=int(np.max(self.data['x']))
        my=int(np.max(self.data['y']))

        color = ['black',(0,0,0,0)]
        cmapm = colors.ListedColormap(color)
        bounds=[-1,0]
        norm = colors.BoundaryNorm(bounds, cmapm.N)

        gridmap = [[0 for x in range(mx+1)] for y in range(my+1)]
        riskmap = np.zeros((my+1, mx+1))
        recommend_patrol_map = np.zeros((my+1, mx+1))
        i=0
        for index, row in self.data.iterrows():
            if i == self.source:
                sy=int(row['y'])
                sx=int(row['x'])
            gridmap[int(row['y'])][int(row['x'])]=1
            i+=1

        total_effort = 0
        for effort in range(effort_size-2):
            grid = [[0 for x in range(mx+1)] for y in range(my+1)]
            i=0
            for index, row in self.data.iterrows():
                grid[int(row['y'])][int(row['x'])]= data[i][effort]
                i+=1
            if effort != 0:
                riskmap += np.array(grid) / effort
            total_effort += 1

        riskmap /= total_effort

            #np.invert(gridmap)

        gridmap = np.ma.masked_where(gridmap == 1, gridmap)
        ratio = self.resolution / float(200) # 1000m -> 5, 500m -> 2.5, 200m -> 1

        # # control view of plot for a cropped section
        # small_min_x = int(10 / ratio)
        # small_max_x = int(40 / ratio)
        # small_min_y = int(40 / ratio)
        # small_max_y = int(60 / ratio)
        #
        # small_gridmap = gridmap[small_min_y:small_max_y, small_min_x:small_max_x]
        # small_riskmap = riskmap[small_min_y:small_max_y, small_min_x:small_max_x]

        # gridmap = small_gridmap
        # riskmap = small_riskmap

        # # --------------- recommend patrol effort ----------------
        # recommend_image = plt.imshow(np.flipud(recommend_patrol_map), interpolation='none', cmap="Greens", extent=[0,mx+1,0,my+1], vmin=0, vmax=np.max(recommend_patrol_map)*1.2)
        # if np.min(gridmap) == 0:
        #     h = plt.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[0,mx+1,0,my+1])
        # plt.colorbar(recommend_image)
        # plt.xticks(np.arange(0,mx+1))
        # plt.yticks(np.arange(0,my+1))
        # plt.grid(ls='solid')
        # plt.title("Post {0}: Maximal Illegal Probability per Patrol Effort".format(post))
        # plt.xticks(np.arange(0,mx+1),[self.min_xval+resolution*i for i in range(mx+1)], rotation=60)
        # plt.xlabel("x")
        # plt.yticks(np.arange(0,my+1),[self.min_yval+resolution*i for i in range(my+1)])
        # plt.ylabel("y")
        # plt.plot([sx+0.5], [sy+0.5], marker='o', markersize=5, color="blue")
        # plt.savefig(self.datafolder+"patrol_effort_map_post_{0}_recommend_effort.png".format(post))
        # plt.close()


        # --------------- risk map (average illegal prob per patrol effort) ----------------
        print gridmap
        print riskmap[gridmap]

        print('saving risk map')
        np.save(self.datafolder + 'riskmap_r_{}_method_{}.npy'.format(self.resolution, self.method), riskmap)

        max_value = np.max(np.ma.masked_where(gridmap==0, riskmap))
        # max_value = 0.16
        min_value = np.min(np.ma.masked_where(gridmap==0, riskmap))
        # min_value = 0.02
        a = plt.imshow(np.flipud(riskmap), interpolation='none', cmap='Reds', extent=[0,mx+1,0,my+1], vmin=min_value, vmax=max_value)
        # a = plt.imshow(np.flipud(riskmap), interpolation='none', cmap="Greens", extent=[small_min_x,small_max_x,small_min_y,small_max_y], vmin=min_value, vmax=max_value)

        if np.min(gridmap) == 0:
            h = plt.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[0,mx+1,0,my+1])
            # h = plt.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[small_min_x,small_max_x,small_min_y,small_max_y])
        #plt.set(h, 'AlphaData', gridmap)

        plt.colorbar(a)
        # plt.xticks(np.arange(0,mx+1))
        # plt.yticks(np.arange(0,my+1))
        # if self.resolution < 500 or len(data) > 1000:
        plt.grid(ls='solid', lw=0.1)
        #else:
        #    plt.grid(ls='solid')
        if not post == None:
            if plot_title:

                #plt.title("Post {0}: Average Illegal Probability per Patrol Effort".format(post))
                plt.title("Risk Map: resolution {}, method {}".format(self.resolution, self.method))
                scalebar = ScaleBar(dx=self.resolution, units='m', fixed_value=1, fixed_units='km', location='lower left') # 1 km or 200 m or 500m
                plt.gca().add_artist(scalebar)
                #plt.xticks(np.arange(0,mx+1),[self.min_xval+resolution*i for i in range(mx+1)], rotation=60)
                plt.xlabel("x", fontsize=6)
                #plt.yticks(np.arange(0,my+1),[self.min_yval+resolution*i for i in range(my+1)])
                plt.ylabel("y", fontsize=6)
            #plt.plot([sx+0.5], [sy+0.5], marker='o', markersize=5, color="blue")

            plt.savefig(self.datafolder + "patrol_effort_map_post_{0}_{1}.png".format(post, name))
            plt.close()
            #plt.show()
        else:
            plt.show()
