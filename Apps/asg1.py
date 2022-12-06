from calendar import c
from json import load
import math
from xmlrpc.client import Boolean
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import time 
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import seaborn as sns


def app(data):
    st.title("Assignment 1")
    
    st.header("Dataset Table")
    st.dataframe(data, width=1000, height=500)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    def printf(url):
         st.markdown(f'<p style="color:#000;font-size:24px;">{url}</p>', unsafe_allow_html=True)

    operation = st.selectbox("Operation", ["Measure Central Tendency",'Dispersion','Analytical Plots'], index=0)
    cols = []
    for i in data.columns[:-1]:
        cols.append(i)

    if operation == "Measure Central Tendency":
        # selected1 = option_menu("Select Attribute 1", cols)
        # selected2 = option_menu("Select Attribute 2", cols)
        attribute1 = st.selectbox("Select Attribute", cols)
        st.subheader("Measures of Central Tendency:")
        
        def Mean():
            sum = 0
            arrmean=[]
            for i in range(len(data)):
                sum += data.loc[i, attribute1]
                arrmean.append(data.loc[i, attribute1])
            avg = sum/len(data)
            res = "Mean of attribute (" + attribute1 + ") is " + str(avg)
            # printf(res)
            res=np.mean(arrmean)
            st.write(f" (in built) mean of {attribute1} : {res}")

        def Mode():
            freq = {}
            for i in range(len(data)):
                freq[data.loc[i, attribute1]] = 0
            maxFreq = 0
            maxFreqElem = 0
            for i in range(len(data)):
                freq[data.loc[i, attribute1]
                    ] = freq[data.loc[i, attribute1]]+1
                if freq[data.loc[i, attribute1]] > maxFreq:
                    maxFreq = freq[data.loc[i, attribute1]]
                    maxFreqElem = data.loc[i, attribute1]
            res = "Mode of attribute ("+attribute1+") is "+str(maxFreqElem)
            printf(res)
        
        def Median():
            arr = []
            for i in range(len(data)):
                arr.append(data.loc[i, attribute1])
            arr.sort()
            n = len(data)
            i = int(n/2)
            j = int((n/2)-1)

            if n % 2 == 1:
                res = "Median of attribute("+attribute1+") is "+str(
                    arr[i])
            else:
                res = "Median of attribute ("+attribute1+") is "+str(
                    (arr[i]+arr[j])/2)
            printf(res)
            res=np.median(arr)
            st.write(f" (in built) median of {attribute1} : {res}")

        
        def Midrange():
            n = len(data)
            arr = []
            for i in range(len(data)):
                arr.append(data.loc[i, attribute1])
            arr.sort()
            res = "Midrange of attribute ("+attribute1+") is "+str((arr[n-1]+arr[0])/2)
            printf(res)
        
        def VSD():
            sum = 0
            arr=[]
            for i in range(len(data)):
                sum += data.loc[i, attribute1]
                arr.append(data.loc[i, attribute1])
            avg = sum/len(data)
            sum = 0
            for i in range(len(data)):
                sum += (data.loc[i, attribute1]-avg) * (data.loc[i, attribute1]-avg)
            var = sum/(len(data))
            res = "Variance of attribute ("+attribute1+") is   "+str(var)
            printf(res)
            res = np.var(arr)
            st.write(f"(in built fun) Variance of {attribute1} is {res}")
            st.write('==================================================')
            res = "Standard Deviation of attribute ("+attribute1+") is  "+str(math.sqrt(var))
            printf(res)
            res = np.std(arr)
            st.write(f"(in built fun) Standard Deviation of {attribute1} is {res}")
            
        
        Mean()
        st.write('====================================================')
        Mode()
        st.write('====================================================')
        Median()
        st.write('====================================================')
        Midrange()
        st.write('====================================================')
        VSD()
        
        

    if operation == "Dispersion":
        # "Range", "Quartiles", "Inetrquartile range", "Minimum", "Maximum"
        attribute = st.selectbox("Select Attribute", cols)
        st.subheader("Dispersion")
        def Range():
            arr = []
            for i in range(len(data)):
                arr.append(data.loc[i, attribute])
            arr.sort()
            res = "Range of attribute ("+attribute+") is "+str(arr[len(data)-1]-arr[0])
            printf(res)

        def Quartiles():
            arr = []
            for i in range(len(data)):
                arr.append(data.loc[i, attribute])
            arr.sort()
            res1 = "Lower quartile(Q1) of (" + attribute+")  is " + str((len(arr)+1)/4)
            res2 = "Middle quartile(Q2) of (" +attribute+") is " + str((len(arr)+1)/2)
            res3 = "Upper quartile(Q3) is (" +attribute+") is " + str(3*(len(arr)+1)/4)
            printf(res1)
            printf(res2)
            printf(res3)

            res = "Interquartile range(Q3-Q1) of given attribute ("+attribute+") is"+str((3*(len(arr)+1)/4)-((len(arr)+1)/4))
            printf(res)
                                    
        def MinMax():
            arr = []
            for i in range(len(data)):
                arr.append(data.loc[i, attribute])
            arr.sort()
            res = "Minimum value of attribute ("+attribute+") is "+str(arr[0])
            printf(res)
            res = "Maximum value of attribute ("+attribute+") is "+str(arr[len(data)-1])
            # printf(res)
            printf(res)

        Range()
        Quartiles()
        MinMax()


    if operation == "Analytical Plots":

       
        plots = ["Quantile-Quantile Plot","Histogram", "Scatter Plot", "Boxplot"]

        plotOpt = st.selectbox("Select Plot", plots)

        if plotOpt == "Quantile-Quantile Plot":
            atr1, atr2 = st.columns(2)
            attribute1 = atr1.selectbox("Select Attribute 1", cols)
            attribute2 = atr2.selectbox("Select Attribute 2", cols)
            # printf(attribute1) 
            # printf(attribute2) 
            classatr = data.columns[-1]
            arr = []
            sum = 0
            for i in range(len(data)):
                arr.append(data.loc[i, attribute1])
                sum += data.loc[i, attribute1]
            avg = sum/len(arr)
            sum = 0
            for i in range(len(data)):
                sum += (data.loc[i, attribute1]-avg) * \
                        (data.loc[i, attribute1]-avg)
            var = sum/(len(data))
            sd = math.sqrt(var)
            z = (arr-avg)/sd
            stats.probplot(z, dist="norm", plot=plt)
            plt.title("Normal Q-Q plot")
            st.pyplot()

        if plotOpt == "Histogram":
            atr1, atr2 = st.columns(2)
            attribute1 = atr1.selectbox("Select Attribute 1", cols)
            # attribute2 = atr2.selectbox("Select Attribute 2", cols)
            # printf(attribute1) 
            # printf(attribute2) 
            classatr = data.columns[-1]
            sns.set_style("whitegrid")
            sns.FacetGrid(data, hue=classatr, height=5).map(
                sns.histplot, attribute1).add_legend()
            plt.title("Histogram")
            plt.show(block=True)
            st.pyplot()
        
        if plotOpt == "Scatter Plot":
            atr1, atr2 = st.columns(2)
            attribute1 = atr1.selectbox("Select Attribute 1", cols)
            attribute2 = atr2.selectbox("Select Attribute 2", cols, index=1)
            classatr = data.columns[-1]
            sns.set_style("whitegrid")
            sns.FacetGrid(data, hue=classatr, height=4).map(
                plt.scatter, attribute1, attribute2).add_legend()
            plt.title("Scatter plot")
            plt.show(block=True)
            st.pyplot()

        if plotOpt == "Boxplot":
            # removing Id column
            atr1, atr2 = st.columns(2)
            attribute1 = atr1.selectbox("Select Attribute 1", cols)
            attribute2 = atr2.selectbox("Select Attribute 2", cols, index=1)
            classatr = data.columns[-1]
            sns.set_style("whitegrid")
            sns.boxplot(x=attribute1, y=attribute2, data=data)
            plt.title("Boxplot")
            plt.show(block=True)
            st.pyplot()
        
        
                                                              
                                

         
