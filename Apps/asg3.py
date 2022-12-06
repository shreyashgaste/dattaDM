import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import math
import io
import requests
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn import preprocessing

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.tree import _tree


def app(data):
    st.title("Assignment 3")
    classatr = data.columns[-1]
    
    st.header("Dataset Table")
    st.dataframe(data, width=1000, height=500)
    def compute_impurity(feature, impurity_criterion):
        
        probs = feature.value_counts(normalize=True)
        
        if impurity_criterion == 'entropy':
            impurity = -1 * np.sum(np.log2(probs) * probs)
        elif impurity_criterion == 'gini':
            impurity = 1 - np.sum(np.square(probs))
        else:
            raise ValueError('Unknown impurity criterion')
            
        return(round(impurity, 3))


    
    target_entropy = compute_impurity(data[classatr], 'entropy')
    target_entropy

    # data['elevation'].value_counts()

    # for level in data['elevation'].unique():
    #     st.write('level name:', level)
    #     data_feature_level = data[data['elevation'] == level]
    #     st.write('corresponding data partition:')
    #     st.write(data_feature_level)
    #     st.write('partition target feature impurity:', compute_impurity(data_feature_level[classatr], 'entropy'))
    #     st.write('partition weight:', str(len(data_feature_level)) + '/' + str(len(data)))
    #     st.write('====================')

    def comp_feature_information_gain(data, target, descriptive_feature, split_criterion):
        
        
        st.write('target feature:', target)
        st.write('descriptive_feature:', descriptive_feature)
        st.write('split criterion:', split_criterion)
                
        target_entropy = compute_impurity(data[target], split_criterion)

        entropy_list = list()
        weight_list = list()
        
        for level in data[descriptive_feature].unique():
            data_feature_level = data[data[descriptive_feature] == level]
            entropy_level = compute_impurity(data_feature_level[target], split_criterion)
            entropy_list.append(round(entropy_level, 3))
            weight_level = len(data_feature_level) / len(data)
            weight_list.append(round(weight_level, 3))

        # st.write('impurity of partitions:', entropy_list)
        # st.write('weights of partitions:', weight_list)

        feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))
        st.write('remaining impurity:', feature_remaining_impurity)
        
        information_gain = target_entropy - feature_remaining_impurity
        st.write('information gain:', information_gain)
        
        

        return(information_gain)

    if st.button("Information gain"):
        split_criterion = 'entropy'
        st.subheader("Information Gain")
        mx = 0
        ftr = ""
        for feature in data.drop(columns=classatr).columns:
            feature_info_gain = comp_feature_information_gain(data, classatr, feature, split_criterion)
            if feature_info_gain > mx:
                mx=feature_info_gain
                ftr = feature
            st.subheader(f"{feature} information gain: {feature_info_gain} ")
            st.write('====================')
        st.header(f"Maximum information gain is : {mx} for feature {ftr}")

    if st.button("Gini Index"):
        split_criteria = 'gini'
        st.subheader("Gini index")
        mx = 0
        ftr = ""
        for feature in data.drop(columns=classatr).columns:
            feature_info_gain = comp_feature_information_gain(data, classatr, feature, split_criteria)
            if feature_info_gain > mx:
                mx=feature_info_gain
                ftr = feature
            st.subheader(f"{feature} Gini index: {feature_info_gain} ")
            st.write('====================')
        st.header(f"Maximum Gini index is : {mx} for feature {ftr}")

    def DeciTree():
        df =  data
        colums = df.columns
        targetAttr = df.columns[-1]
        features = list(colums)
        features.remove(targetAttr)
        x = df[features]
        y = df[targetAttr] # Target variable

        dataEncoder = preprocessing.LabelEncoder()
        encoded_x_data = x.apply(dataEncoder.fit_transform)

        st.header("Criteria:Information Gain")
        # "leaves" (aka decision nodes) are where we get final output
        # root node is where the decision tree starts
        # Create Decision Tree classifer object
        decision_tree = DecisionTreeClassifier(criterion="entropy")
        # Train Decision Tree Classifer
        decision_tree = decision_tree.fit(encoded_x_data, y)
        
        #plot decision tree
        fig, ax = plt.subplots(figsize=(6, 6)) 
        #figsize value changes the size of plot
        tree.plot_tree(decision_tree,ax=ax,feature_names=features)
        plt.show()
        st.pyplot(plt)

        st.header("Criteria:Gini Index")
        decision_tree = DecisionTreeClassifier(criterion="gini")
        # Train Decision Tree Classifer
        decision_tree = decision_tree.fit(encoded_x_data, y)
        
        fig, ax = plt.subplots(figsize=(6, 6)) 
        tree.plot_tree(decision_tree,ax=ax,feature_names=features)
        plt.show()
        st.pyplot(plt)

        X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1)

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(max_depth=2, random_state=1)

        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        c_matrix = confusion_matrix(y_test, y_pred)

        tp = c_matrix[1][1]
        tn = c_matrix[2][2]
        fp = c_matrix[1][2]
        fn = c_matrix[2][1]


        st.subheader("Confusion Matrix:")
        # st.write(c_matrix)
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.set(font_scale=2.4)
        sns.heatmap(c_matrix, cmap="icefire", annot=True, linewidths=1, ax=ax)
        plt.show()
        st.pyplot(fig)
        # Tabulate the results in confusion matrix and evaluate the performance of above classifier using following metrics :
        st.subheader('Performance :')
        # st.write("Model Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))
        val = metrics.accuracy_score(y_test, y_pred)
        st.write('Accuracy score : ',metrics.accuracy_score(y_test, y_pred))
        st.write('Misclassification Rate : ', 1-metrics.accuracy_score(y_test, y_pred))
        # precision score
        val = metrics.precision_score(y_test, y_pred, average='macro')
        # print('Precision score : ' + str(val))
        st.write('Precision score : ', precision_score(y_test, y_pred, average='macro'))
        st.write("Recall(Sensitivity): ", recall_score(y_test, y_pred, average="macro"))
        st.write("Specificity: ", recall_score(y_test, y_pred, pos_label=0, average="macro"))



        # Accuracy score

        #Assignment 4
        st.header("Rule Base Classifier")
        # get the text representation
        text_representation = tree.export_text(clf,feature_names=features)
        st.text(text_representation)

        #Extract Code Rules
        st.subheader("Extract Code Rules")



        def tree_to_code(tree, feature_names):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
            st.text("def predict({}):".format(", ".join(feature_names)))

            def recurse(node, depth):
                indent = "    " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    st.text("{}     if {} <= {}:".format(indent, name, np.round(threshold,2)))
                    recurse(tree_.children_left[node], depth + 1)
                    st.text("{}     else:  # if {} > {}".format(indent, name, np.round(threshold,2)))
                    recurse(tree_.children_right[node], depth + 1)
                else:
                    st.text("\t{}return {}".format(indent, tree_.value[node]))

            recurse(0, 1)
        
        tree_to_code(decision_tree,features)
        # if st.button("Extract Rules"):
        
    if st.button("Decision Tree"):
        DeciTree()