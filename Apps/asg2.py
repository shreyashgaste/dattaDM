from json import load
from xmlrpc.client import Boolean
import streamlit as st
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
import math
import seaborn as sns
def app(data):
    st.title("Assignment 2")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.header("Dataset Table")
    st.dataframe(data, width=1000, height=500)
    def printf(url):
         st.markdown(f'<p style="color:#000;font:lucida;font-size:25px;">{url}</p>', unsafe_allow_html=True)

    operation = st.selectbox("Operation", ["Chi-Square Test",'Correlation(Pearson) Coefficient','Normalization Techniques'])
    cols = []
    for i in data.columns[:-1]:
        cols.append(i)

    #Operations
    if operation == "Chi-Square Test":
        atr1, atr2 = st.columns(2)
        attribute1 = atr1.selectbox("Select Attribute 1", cols)
        attribute2 = atr2.selectbox("Select Attribute 2", cols, index=1)
        # printf(attribute1) 
        # printf(attribute2) 
        classatr = data.columns[-1]
        arrClass = data[classatr].unique()
        g = data.groupby(classatr)
        f = {
        attribute1: 'sum',
        attribute2: 'sum'
        }
        v1 = g.agg(f)
        # st.write(v1)
        v = v1.transpose()
        st.table(v)
        # df_rows = v.to_numpy().tolist()
       
        total = v1[attribute1].sum()+v1[attribute2].sum()
        chiSquare = 0.0
        for i in arrClass:
            chiSquare += (v.loc[attribute1][i]-(((v[i].sum())*(v1[attribute1].sum()))/total))*(v.loc[attribute1][i]-(((v[i].sum())*(v1[attribute1].sum()))/total))/(((v[i].sum())*(v1[attribute1].sum()))/total)
            chiSquare += (v.loc[attribute2][i]-(((v[i].sum())*(v1[attribute2].sum()))/total))*(v.loc[attribute2][i]-(((v[i].sum())*(v1[attribute2].sum()))/total))/(((v[i].sum())*(v1[attribute2].sum()))/total)
        
        degreeOfFreedom = (len(v)-1)*(len(v1)-1)
        printf("Chi-square value is "+str(chiSquare))
        printf("Degree of Freedom is "+str(degreeOfFreedom))
        res = ""
        if chiSquare > degreeOfFreedom:
            res = "Attributes " + attribute1 + ' and ' + attribute2 + " are strongly correlated."
        else:
            res = "Attributes " + attribute1 + ' and ' + attribute2 + " are not correlated."

        printf(res)           
        
    if operation == "Correlation(Pearson) Coefficient":
        atr1, atr2 = st.columns(2)
        attribute1 = atr1.selectbox("Select Attribute 1", cols)
        attribute2 = atr2.selectbox("Select Attribute 2", cols)
        # printf(attribute1) 
        # printf(attribute2) 
        classatr = data.columns[-1]
    
        sum = 0
        for i in range(len(data)):
            sum += data.loc[i, attribute1]
        avg1 = sum/len(data)
        sum = 0
        for i in range(len(data)):
            sum += (data.loc[i, attribute1]-avg1)*(data.loc[i, attribute1]-avg1)
        var1 = sum/(len(data))
        sd1 = math.sqrt(var1)
        
        sum = 0
        for i in range(len(data)):
            sum += data.loc[i, attribute2]
        avg2 = sum/len(data)
        sum = 0
        for i in range(len(data)):
            sum += (data.loc[i, attribute2]-avg2)*(data.loc[i, attribute2]-avg2)
        var2 = sum/(len(data))
        sd2 = math.sqrt(var2)
        
        sum = 0
        for i in range(len(data)):
            sum += (data.loc[i, attribute1]-avg1)*(data.loc[i, attribute2]-avg2)
        covariance = sum/len(data)
        pearsonCoeff = covariance/(sd1*sd2) 
        printf("Covariance value is "+str(covariance))
        printf("Correlation coefficient(Pearson coefficient) is "+str(pearsonCoeff))
        res = ""
        if pearsonCoeff > 0:
            res = "Attributes " + attribute1 + ' and ' + attribute2 + " are positively correlated."
        elif pearsonCoeff < 0:
            res = "Attributes " + attribute1 + ' and ' + attribute2 + " are negatively correlated."
        elif pearsonCoeff == 0:
            res = "Attributes " + attribute1 + ' and ' + attribute2 + " are independant."
        printf(res)
        
    if operation == "Normalization Techniques":
        normalizationOperations = ["Min-Max normalization","Z-Score normalization","Normalization by decimal scaling"]
        function = st.selectbox("Normalization Methods", normalizationOperations)
        atr1, atr2 = st.columns(2)
        attribute1 = atr1.selectbox("Select Attribute 1", cols)
        attribute2 = atr2.selectbox("Select Attribute 2", cols, index=1)
        # printf(attribute1) 
        # printf(attribute2) 
        classatr = data.columns[-1]

        d = data
        if function == "Min-Max normalization":
            n = len(data)
            arr1 = []
            for i in range(len(data)):
                arr1.append(data.loc[i, attribute1])
            arr1.sort()
            min1 = arr1[0]
            max1 = arr1[n-1]
            
            arr2 = []
            for i in range(len(data)):
                arr2.append(data.loc[i, attribute2])
            arr2.sort()
            min2 = arr2[0]
            max2 = arr2[n-1]
            
            for i in range(len(data)):
                d.loc[i, attribute1] = ((data.loc[i, attribute1]-min1)/(max1-min1))
            
            for i in range(len(data)):
                d.loc[i, attribute2] = ((data.loc[i, attribute2]-min2)/(max2-min2))
        elif function == "Z-Score normalization":
            sum = 0
            for i in range(len(data)):
                sum += data.loc[i, attribute1]
            avg1 = sum/len(data)
            sum = 0
            for i in range(len(data)):
                sum += (data.loc[i, attribute1]-avg1)*(data.loc[i, attribute1]-avg1)
            var1 = sum/(len(data))
            sd1 = math.sqrt(var1)
            
            sum = 0
            for i in range(len(data)):
                sum += data.loc[i, attribute2]
            avg2 = sum/len(data)
            sum = 0
            for i in range(len(data)):
                sum += (data.loc[i, attribute2]-avg2)*(data.loc[i, attribute2]-avg2)
            var2 = sum/(len(data))
            sd2 = math.sqrt(var2)
            
            for i in range(len(data)):
                d.loc[i, attribute1] = ((data.loc[i, attribute1]-avg1)/sd1)
            
            for i in range(len(data)):
                d.loc[i, attribute2] = ((data.loc[i, attribute2]-avg2)/sd2)
        elif function == "Normalization by decimal scaling":        
            j1 = 0
            j2 = 0
            n = len(data)
            arr1 = []
            for i in range(len(data)):
                arr1.append(data.loc[i, attribute1])
            arr1.sort()
            max1 = arr1[n-1]
            
            arr2 = []
            for i in range(len(data)):
                arr2.append(data.loc[i, attribute2])
            arr2.sort()
            max2 = arr2[n-1]
            
            while max1 > 1:
                max1 /= 10
                j1 += 1
            while max2 > 1:
                max2 /= 10
                j2 += 1
            
            for i in range(len(data)):
                d.loc[i, attribute1] = ((data.loc[i, attribute1])/(pow(10,j1)))
            
            for i in range(len(data)):
                d.loc[i, attribute2] = ((data.loc[i, attribute2])/(pow(10,j2)))
        st.subheader("Normalized Attributes")
        st.dataframe([d[attribute1], d[attribute2]])
        sns.set_style("whitegrid")
        sns.FacetGrid(d, hue=classatr, height=4).map(plt.scatter, attribute1, attribute2).add_legend()
        plt.title(f"{function}")
        plt.show(block=True)
        st.pyplot()
    
