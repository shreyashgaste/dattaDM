import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import collections
from scipy.cluster import hierarchy
from sklearn import datasets
from random import randint
import random
import plotly.express as px
import altair as alt
import seaborn as sns
def app(data):
    st.title("Assignment 6")

    st.set_option('deprecation.showPyplotGlobalUse', False)

    def printf(url):
         st.markdown(f'<p style="color:#000;font:lucida;font-size:25px;">{url}</p>', unsafe_allow_html=True)

    operation = st.selectbox("Operation", ["AGNES",'DIANA','KMeans', 'KMedoid','DBSCAN'])


    if operation == "AGNES":
        arr = []
        n=0

        cols = []
        for i in data.columns[:-1]:
            cols.append(i)
        atr1, atr2 = st.columns(2)
        attribute1 = atr1.selectbox("Select Attribute 1", cols)
        attribute2 = atr2.selectbox("Select Attribute 2", cols, index=2)
        for i in range(len(data)):
                arr.append([data.loc[i, attribute1],data.loc[i, attribute2]])
        n = len(arr)
        k = int(st.number_input("Enter no of Clusters (k): ",min_value=1, step=1))
        
        minPoints = 0
        if len(arr)%k==0:
            minPoints=len(arr)//k
        else:
            minPoints = (len(arr)//k)+1
        # print(len(arr))
        # print(minPoints)

        def Euclid(a,b):
            # print(a,b)
            # finding sum of squares
            sum_sq = np.sum(np.square(a - b))
            return np.sqrt(sum_sq)

        points=[[0]]
        def findPoints(point):
            min = 100000.22
            pt=-1
            for i  in point:
                for j in range(len(arr)):
                    if j in point:
                        continue
                    else:
                        # print(arr[i], arr[j])
                        dis = Euclid(np.array(arr[i]),np.array(arr[j]))
                        if min > dis and dis < 1.900000:
                            min = dis
                        # print(min)
                        pt=j
            return pt

        travetsedPoints=[0]
        for i in range(0,k):
            if len(travetsedPoints) >= len(arr):
                break
        
        # if len(points)>=k:
        #   break

            while(len(points[i])<minPoints):
            # while(True):
                pt = findPoints(travetsedPoints)
                if pt in travetsedPoints:
                    break
                travetsedPoints.append(pt)
                points[i].append(pt)
            points.append([])
        points.remove([])
        # print(points)

        # colarr = ['blue','green','red','black']

        colarr = []

        for i in range(k):
            colarr.append('#%06X' % randint(0, 0xFFFFFF))

        # no_of_colors=k
        # colarr=["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        #        for j in range(no_of_colors)]
        # print(color)
        i=0
        cluster=[]
        for j in range(k):
            cluster.append(j)

        annotations=["Point-1","Point-2","Point-3","Point-4","Point-5"]
        fig, axes = plt.subplots(1, figsize=(15, 20))
        for atr in points:
            for j in range(minPoints):
                if atr[j]==-1:
                    continue
                pltY = atr[j]
                pltX = cluster[i%(k+1)]
                # pltX = arr[atr[j]][0]
                # pltY = arr[atr[j]][1]
                # pltY = data.loc[:,classatr]
                plt.scatter(pltX, pltY, color=colarr[i])
                label = str("(" + str(arr[atr[j]][0]) + "," + str(arr[atr[j]][1]) + ")")
                plt.text(pltX, pltY, label)

            i += 1

        plt.legend(loc=1, prop={'size':4})
        plt.show()
        # st.pyplot()

        # colarr = ['blue','green','red','black']
        j=0
        def findIndex(ptarr):
            # print("Ptarr: ", ptarr)
            for j in range(len(points)):
                if ptarr in points[j]:
                    return j
            
        fig, axes = plt.subplots(1, figsize=(10, 7))
        clusters=[]
        for i in range(k):
            clusters.append([[],[]])

        for i in range(len(arr)):
            j = findIndex(i)
            clusters[j%k][0].append(arr[i][0])
            clusters[j%k][1].append(arr[i][1])

        # print(i)
        # plt.scatter(arr[i][0],arr[i][1], color = colarr[j])
        # print(clusters)
        for i in range(len(clusters)):
            plt.scatter(clusters[i][0],clusters[i][1], color = colarr[i%k], label=cluster[i])
        plt.title("Sample plot")
        plt.xlabel("X Cordinate")
        plt.legend(loc=1, prop={'size':15})

        # plt.legend(["x*2" , "x*3"])
        plt.ylabel("Y Cordinate")
        plt.show()
        st.pyplot()

        st.subheader("Dendogram")
        # dismatrix =[]
        # # @st.cache()
        # # def creatematrix():
        # for i in range(len(arr)):
        #     for j in range(i+1, len(arr)):
        #         dismatrix.append([Euclid(np.array(arr[i]),np.array(arr[j]))])
        # # creatematrix()
        # ytdist = dismatrix
        # # # fig, axes = plt.subplots(figsize=(15, 8))
        # # Z = hierarchy.linkage(ytdist, method='single')
        # # dn = hierarchy.dendrogram(Z, above_threshold_color='#070dde',orientation='right')
        # # plt.figure()
        # st.write(dismatrix)
        # sns.clustermap(data=ydist)
        # plt.show()
        # # dn.show()
        # st.pyplot()
    if operation == "DIANA":
            arr = []
            n=0

            cols = []
            for i in data.columns[:-1]:
                cols.append(i)
            atr1, atr2 = st.columns(2)
            attribute1 = atr1.selectbox("Select Attribute 1", cols)
            attribute2 = atr2.selectbox("Select Attribute 2", cols, index=2)
            for i in range(len(data)):
                    arr.append([data.loc[i, attribute1],data.loc[i, attribute2]])
            n = len(arr)
            k = int(st.number_input("Enter no of Clusters (k): ",min_value=1, step=1))
        
            # iris = datasets.load_iris()
            # X = iris.data
            # arr = []
            # n=0
            # for i in X:
            #   arr.append([i[0],i[1]])
            #   n += 1

            # print(X)
            # print("------------")
            # print(arr[0], arr[1])
            # print("------------")
            # print(atr2)




            # arr = np.array([[1, 2],[3,2],[2,5],[1,3],[6,5],[7,5],[4,6],[3,5],[4,1],[5,6],[3,8],[8,5]])
            # k = 3
            minPoints = 0
            if len(arr)%k==0:
              minPoints=len(arr)//k
            else:
              minPoints = (len(arr)//k)+1
            # print(len(arr))
            print(minPoints)

            def Euclid(a,b):
              # print(a,b)
            # finding sum of squares
              sum_sq = np.sum(np.square(a - b))
              return np.sqrt(sum_sq)



            points=[[0]]
            def findPoints(point):
              max = 0
              pt=-1
              for i  in point:
                for j in range(len(arr)):
                  if j in point:
                    continue
                  else:
                    # print(arr[i], arr[j])
                    dis = Euclid(np.array(arr[i]),np.array(arr[j]))
                    if max < dis:
                      max = dis
                      # print(max)
                      pt=j
              return pt

            travetsedPoints=[0]
            for i in range(0,k):
              if len(travetsedPoints) >= len(arr):
                break
              
              # if len(points)>=k:
              #   break

              while(len(points[i])<minPoints):
              # while(True):
                pt = findPoints(travetsedPoints)
                if pt in travetsedPoints:
                  break
                travetsedPoints.append(pt)
                points[i].append(pt)
              points.append([])
            points.remove([])
            # st.write(points)



            # colarr = ['blue','green','red','black']

            colarr = []

            for i in range(k):
                colarr.append('#%06X' % randint(0, 0xFFFFFF))

            i=0
            cluster=[]
            for j in range(k):
              cluster.append(j)

            st.subheader("Cluster and Points")
            # annotations=["Point-1","Point-2","Point-3","Point-4","Point-5"]
            fig, axes = plt.subplots(1, figsize=(15, 20))
            for atr in points:
              for j in range(minPoints):
                if atr[j]==-1:
                  continue
                pltY = atr[j]
                pltX = cluster[i%(k+1)]
                # pltX = arr[atr[j]][0]
                # pltY = arr[atr[j]][1]
                # pltY = data.loc[:,classatr]
                plt.scatter(pltX, pltY, color=colarr[i])
                label = str("(" + str(arr[atr[j]][0]) + "," + str(arr[atr[j]][1]) + ")")
                plt.text(pltX, pltY, label)

              i += 1

            plt.legend(loc=1, prop={'size':4})
            # plt.show()
            st.pyplot()

            j=0
            def findIndex(ptarr):
                # print("Ptarr: ", ptarr)
                for j in range(len(points)):
                    if ptarr in points[j]:
                        return j
                
            fig, axes = plt.subplots(1, figsize=(10, 7))
            clusters=[]
            for i in range(k):
                clusters.append([[],[]])

            for i in range(len(arr)):
                j = findIndex(i)
                clusters[j%k][0].append(arr[i][0])
                clusters[j%k][1].append(arr[i][1])

                # print(i)
                # plt.scatter(arr[i][0],arr[i][1], color = colarr[j])
            for i in range(len(clusters)):
                plt.scatter(clusters[i][0],clusters[i][1], color = colarr[i%k], label=cluster[i])
            plt.title("Sample plot")
            plt.xlabel("X Cordinate")
            plt.legend(loc=1, prop={'size':15})

            # plt.legend(["x*2" , "x*3"])
            plt.ylabel("Y Cordinate")
            # plt.show()
            st.subheader("Clustering")
            st.pyplot()
        
    if operation == "KMeans":
        arr = []
        n=0

        cols = []
        for i in data.columns[:-1]:
            cols.append(i)
        atr1, atr2 = st.columns(2)
        attribute1 = atr1.selectbox("Select Attribute 1", cols)
        attribute2 = atr2.selectbox("Select Attribute 2", cols, index=2)
        for i in range(len(data)):
                arr.append([data.loc[i, attribute1],data.loc[i, attribute2]])
        n = len(arr)
        # for i in data[:-1]:
        #     arr.append([i[0],i[1]])
        #     n += 1
        # print(X)
        # print("------------")
        # print(arr[0], arr[1])
        # print("------------")
        # print(atr2)

        #clusters 
        # k=0
        
        k = int(st.number_input("Enter no of Clusters (k): ",min_value=1, step=1))

        # imeanArr = [2,4,10,12,3,20,30,11,25]
        meanMatrix = [[0] * k for i in range(n)]
        # print(meanMatrix)

        # kmeanList = [random.randint(1,3) for i in range(k)]
        # kmeanList = [4.2,4.5,5.2,5.5]
        
        kmeanList = [round(random.uniform(4.2,5.6),3) for i in range(k)]
        # for i in range(k):
        # print(kmeanList[i])

        # print(arr)

        imeanArr = []
        for i in arr:
            imeanArr.append(round(np.mean(i), 5))
        # print(imeanArr)

        clusterMean = [[] for i in range(k)]
        prevKmean=[]
        for i in range(10):
            # for j in range(k):
            #   print(f"kmeanList[{j}]",kmeanList[j])
            for i in range(len(imeanArr)):
                for j in range(k):
                # print("a:",imeanArr[i]," b:", kmeanList[j])
                    meanMatrix[i][j] = abs(kmeanList[j] - imeanArr[i])
                # print(f"meanMatrix[{i}][{j}]",meanMatrix[i][j])
        # print(meanMatrix)

        def findminIndex(lst):
            min, j = 1000000,0
            for i in range(len(lst)):
                if lst[i] < min:
                    min = lst[i]
                    j=i
            return min, j
        clusterMean = [[] for i in range(k)]
        for i in range(len(meanMatrix)): 
            min, j = findminIndex(meanMatrix[i])
            # for m in range(k):
            #   print(meanMatrix[i][m],", ", end='')
            # print(" = (",j,",", min,")")
            clusterMean[j].append(imeanArr[i])
        # print("Cluster:",clusterMean)
        
        prevKmean = list()
        prevKmean = kmeanList.copy()
        for i in range(len(clusterMean)):
            kmeanList[i] = np.mean(clusterMean[i])
            # print(f"kmeanList[{i}]",kmeanList[i])
        # print("kmeanList: ",kmeanList)
        # print("Prev kmean: ", prevKmean)
        
        flg=True
        for i in range(len(kmeanList)):
            if prevKmean[i] != kmeanList[i]:
                flg = False
                break
            if flg:
                break

        # print(" ---------------------------------------------- ")

        sizeofClusters=[]
        for i in clusterMean:
            sizeofClusters.append(len(i))

        # print(sizeofClusters)

        points=[]

        for i in clusterMean:
            for j in i:
                # print("Mean:",j)
                # print("Index:",imeanArr.index(j))
                # print("Point:",arr[imeanArr.index(j)])
                for v in range(len(imeanArr)):
                    if imeanArr[v] == j:
                        pt = arr[v]
                        if pt in points:
                            continue
                        points.append(pt)
                # pt = arr[imeanArr.index(j)]
                # imeanArr.remove(j)
                # print("Add: ",pt)
                # points.append(pt)
            # print(len(points))
            # print(points)

        colarr=['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

        # if(k<4):
        #     colarr = ["#FC0404","#0422FC","#5AFC04","#0x000", "#FCF804", "#FC04B4"]
        # else:
        #     for i in range(k):
        #         colarr.append('#%06X' % randint(0, 0xFFFF))

        

        # fig, axes = plt.subplots(1, figsize=(10, 7))



        # for i in range(len(sizeofClusters)):
        #   clno = sizeofClusters[i]
        i=0
        sum=0
        newCluster = [[[],[]]]
        for j in range(len(points)):
            if j >= sum+sizeofClusters[i%k]:
                sum=j
                i += 1
                newCluster.append([[],[]])
            newCluster[i][0].append(points[j][0])
            newCluster[i][1].append(points[j][1])
            # for i in newCluster:
            #   print(len(i[0]))

        pltX=[]
        pltY=[]
        # chart
        for i in range(len(newCluster)):
            # print(len(newCluster[i][0]), len(newCluster[i][1]))
            # print(newCluster[i][0])
            # print(newCluster[i][1])
            pltX.extend(newCluster[i][0])
            pltY.extend(newCluster[i][1])
            plt.scatter(newCluster[i][0], newCluster[i][1], color = colarr[i%k], label=i) 
            
        # plt.scatter(pltX,pltY)
        # print(len(pltX))
        # print(len(pltY))
        plt.title("Sample plot")
        plt.xlabel("X Cordinate")
        plt.ylabel("Y Cordinate")
        plt.legend(loc=1, prop={'size':15})
        plt.show()
        st.pyplot()





