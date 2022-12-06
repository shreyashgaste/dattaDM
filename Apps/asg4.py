from json import load
from turtle import color
from xmlrpc.client import Boolean
import streamlit as st
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB  
from matplotlib.colors import ListedColormap  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode
import math

from sklearn.model_selection import train_test_split
import random

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.tree import _tree

def app(data):
    st.title("Assignment 4")
    
    st.header("Dataset Table")
    st.dataframe(data, width=1000, height=500)
    def printf(url):
         st.markdown(f'<p style="color:#000;font-size:24px;">{url}</p>', unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Prepare the data data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    cols = []
    for i in data.columns[:-1]:
        cols.append(i)
    
    # Fit the classifier with max_depth=3
    classatr = data.columns[-1]
       
    clf = DecisionTreeClassifier(max_depth=3, random_state=1234)
    model = clf.fit(X, y)
    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]

    # y = target values, last column of the data frame
    y = data.iloc[:, -1]

    # plt.xlabel("Feature")
    plt.ylabel(classatr)

    colarr = ['blue','green','red','black']
    i=0
    for attribute in cols:
        pltX = data.loc[:, attribute]
        pltY = data.loc[:,classatr]
        plt.scatter(pltX, pltY, color=colarr[i], label=attribute)
        i += 1

    
    plt.legend(loc=4, prop={'size':8})
    plt.show()
    st.pyplot()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #Train the model
    model = LogisticRegression()
    model.fit(x_train, y_train) #Training the model
    
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(model, x_test, y_test)
    st.pyplot()
    # cm = confusion_matrix(X_test, Y_test)   
    # # print(cm)
    # sns.heatmap(cm, cmap="icefire", annot=True)
    # plt.show()
    # # st.write(confusion_matrix(Y_test, ))
    # st.pyplot()
                
    # st.write(model)
    
    # get the text representation
    # text_representation = tree.export_text(clf)
    # st.write(text_representation)
    
    # text_representation = tree.export_text(clf, feature_names=iris.feature_names)
    # st.write(text_representation)
    def get_rules(tree, feature_names, class_names):
        tree_ = tree.tree_
        
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        path = []
        paths = []

        def recurse(node, path, paths):

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            rule = "if "

            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                rule += "response: "+str(np.round(path[-1][0][0][0],3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {class_names[l]} (probability: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]

        return rules

    rules = get_rules(clf, iris.feature_names, iris.target_names)
    for r in rules:
        printf(r)
    
    
    