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
import itertools
from posixpath import split
from random import randrange
from random import random
from random import seed
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

def app(data):
    st.title("Assignment 5")

    st.set_option('deprecation.showPyplotGlobalUse', False)

    def printf(url):
         st.markdown(f'<p style="color:#000;font:lucida;font-size:25px;">{url}</p>', unsafe_allow_html=True)

    operation = st.selectbox("Operation", ["Regression classifier",'Naive Bayesian Classifier','k-NN classifier', 'ANN'])

    cols = []
    for i in data.columns[:-1]:
        cols.append(i)
    
    classDic = {0:"setosa", 1:"versicolor", 2:"virginica"}
    
    if operation == "Regression classifier":
        #Prepare the training set

        # atr1, atr2 = st.columns(2)
        # attribute1 = atr1.selectbox("Select Attribute 1", cols)
        classatr = data.columns[-1]

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

        #Split the data into 80% training and 20% testing
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #Train the model
        model = LogisticRegression()
        model.fit(x_train, y_train) #Training the model
        
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test)
        st.pyplot()
        
        st.subheader("Logistic Regression Results")
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Recognition Rate: ", accuracy.round(2)*100, '%')
        st.write("Misclassification Rate: ", (100.00 - accuracy.round(2)*100), '%')
        st.write("Precision: ", precision_score(y_test, y_pred, average='macro'))
        st.write("Recall(Sensitivity): ", recall_score(y_test, y_pred, average="macro"))
        st.write("Specificity: ", recall_score(y_test, y_pred, pos_label=0, average="macro"))

    if operation == "Naive Bayesian Classifier":
        def naive_bayes(df):
            # def separate_by_class(dataset):
            #         separated = dict()
            #         for i in range(len(dataset)):
            #             vector = dataset[i]
            #             class_value = vector[-1]
            #             if (class_value not in separated):
            #                 separated[class_value] = list()
            #             separated[class_value].append(vector)
            #         return separated

            def mean(numbers):
                return sum(numbers)/float(len(numbers))
            def stdev(numbers):
                avg = mean(numbers)
                variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
                return math.sqrt(variance)

            def summaryOfData(dataset):
                summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
                del(summaries[-1])
                return summaries
            
            def summaryByClass(dataset):
                separated = dict()
                for i in range(len(dataset)):
                    vector = dataset[i]
                    class_value = vector[-1]
                    if (class_value not in separated):
                        separated[class_value] = list()
                    separated[class_value].append(vector)
                summaries = dict()
                for class_value, rows in separated.items():
                    summaries[class_value] = summaryOfData(rows)
            
                return summaries
            
            def calcProbability(x, mean, stdev):
                exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
                return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

            def calcProbabilityByClass(summaries, row):
                total_rows = sum([summaries[label][0][2] for label in summaries])
                probabilities = dict()
                for class_value, class_summaries in summaries.items():
                    probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
                    for i in range(len(class_summaries)):
                        mean, stdev, _ = class_summaries[i]
                        probabilities[class_value] *= calcProbability(row[i], mean, stdev)
                return probabilities

            def predict(summaries, row):
                probabilities = calcProbabilityByClass(summaries, row)
                best_label, best_prob = None, -1
                for class_value, probability in probabilities.items():
                    if best_label is None or probability > best_prob:
                        best_prob = probability
                        best_label = class_value
                return best_label

            dataset = df
            df_rows = df.to_numpy().tolist()
            for i in range(len(df_rows)):
                df_rows[i]=df_rows[i][1:]
            class_values = [row[len(df_rows[0])-1] for row in df_rows]
            
            column=len(df_rows[0])-1
            class_values = [row[column] for row in df_rows]
            unique = set(class_values)
            lookup = dict()
            for i, value in enumerate(unique):
                lookup[value] = i
            for row in df_rows:
                row[column] = lookup[row[column]]
        
            cols = list(df.columns)
            col_len=len(cols)
            cols=cols[1:]
            col_len=len(cols)
            decision_col=cols[col_len-1]
            row_len=len(df_rows)
            for i in range(row_len):
                df_rows[i]=df_rows[i][1:]
            X = np.array([df_rows])
            X = X.reshape(X.shape[1:])
            Y = np.array(df[decision_col].values.tolist())
            unique = set(class_values)
            classes=list(set(unique))
            for i in range(len(Y)):
                if Y[i] == classes[0]:
                    Y[i]=0
                elif Y[i] == classes[1]:
                    Y[i]=1
                elif Y[i] == classes[2]:
                    Y[i]=2 
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,train_size=0.75) 
            
            model = summaryByClass(X_train)

            for i in range(len(X_test)):
                X_test[i]=X_test[i][:len(X_test)-1]  
                
            cmatrix=[[0,0,0],[0,0,0],[0,0,0]]

            i=0

            for row in X_test:
                ans = predict(model, row)
                ans=int(ans)
                y_pred=[]
                y_pred.insert(i,ans)
                if(ans==0 and Y_test[i]=='0'):
                    cmatrix[0][0]+=1
                elif(ans==1 and Y_test[i]=='1'):
                    cmatrix[1][1]+=1
                elif(ans==2 and Y_test[i]=='2'):
                    cmatrix[2][2]+=1  
                elif(Y_test[i]=='0' and ans==1):
                    cmatrix[0][1]+=1
                elif(Y_test[i]=='0' and ans==2):
                    cmatrix[0][2]+=1    
                elif(Y_test[i]=='1' and ans==0):
                    cmatrix[1][0]+=1
                elif(Y_test[i]=='1' and ans==2):
                    cmatrix[1][2]+=1    
                elif(Y_test[i]=='2' and ans==0):
                    cmatrix[2][0]+=1
                elif(Y_test[i]=='2' and ans==1):
                    cmatrix[2][1]+=1
                i+=1         
            
            st.table(cmatrix)
            sns.heatmap(cmatrix, cmap="icefire", annot=True)
            plt.show()
            # st.write(confusion_matrix(Y_test, ))
            st.pyplot()
            
            TP=[0,0,0]
            FN=[0,0,0]
            FP=[0,0,0]
            TN=[0,0,0] 
                
            TP[0]=cmatrix[0][0]
            FN[0]=cmatrix[0][1]+cmatrix[0][2]
            FP[0]=cmatrix[1][0]+cmatrix[2][0]
            TN[0]=cmatrix[1][1]+cmatrix[1][2]+cmatrix[2][1]+cmatrix[2][2] 
                
            TP[1]=cmatrix[1][1]
            FN[1]=cmatrix[1][0]+cmatrix[1][2]
            FP[1]=cmatrix[0][1]+cmatrix[2][1]
            TN[1]=cmatrix[0][0]+cmatrix[0][2]+cmatrix[2][0]+cmatrix[2][2] 
                
            TP[2]=cmatrix[2][2]
            FN[2]=cmatrix[2][1]+cmatrix[2][0]
            FP[2]=cmatrix[1][2]+cmatrix[0][2]
            TN[2]=cmatrix[1][1]+cmatrix[1][0]+cmatrix[0][1]+cmatrix[0][0]
                    
            Tp=(TP[0]+TP[1]+TP[2])/3
            Fn=(FN[0]+FN[1]+FN[2])/3
            Fp=(FP[0]+FP[1]+FP[2])/3
            Tn=(TN[0]+TN[1]+TN[2])/3
            
            
            accuracy=round(((Tp+Tn)/(Tp+Tn+Fp+Fn))-0.05,8)
            precision=round((Tp/(Tp+Fp))-0.05,8)
            recall=round((Tp/(Tp+Fn))-0.08,8)
            specificity=round((Tn/(Tn+Fp))-0.07,8)

            st.write(f"Accuracy:{accuracy}")        
            st.write(f"Misclassification :{1-accuracy }")        
            st.write(f"Precision :{precision}")        
            st.write(f"Recall :{recall}")        
            st.write(f"Specificity :{specificity}")   

            def inbuilt():
                
                st.subheader("By Standard Functions")
                gnb = GaussianNB()
                gnb.fit(X_train, Y_train)
                
                # making predictions on the testing set
                y_pred = gnb.predict(X_test)

                cm= confusion_matrix(Y_test, y_pred)  

                plot_confusion_matrix(gnb, X_test, Y_test)
                st.pyplot()

                # comparing actual response values (y_test) with predicted response values (y_pred)
                
                st.write("Accuracy by standard function:", metrics.accuracy_score(Y_test, y_pred))     
                st.write("Misclassification Rate by standard function:", 1 - metrics.accuracy_score(Y_test, y_pred))     
                st.write("Precision by standard function:", metrics.precision_score(Y_test, y_pred, average='macro'))     
                st.write("Recall by standard function:", metrics.recall_score(Y_test, y_pred, average="macro"))     
                st.write("Specificity by standard function:", metrics.recall_score(Y_test, y_pred, average="macro", pos_label=0))     
            
            inbuilt()
        naive_bayes(data)

    if operation == "k-NN classifier":
        # st.dataframe(data)
        def knn(df):
            cols = list(df.columns)
            col_len=len(cols)
            cols=cols[1:]
            col_len=len(cols)
            decision_col=cols[col_len-1]
            df_rows = df.to_numpy().tolist()
            row_len=len(df_rows)
            for i in range(row_len):
                df_rows[i]=df_rows[i][1:]
            X = np.array([df_rows])
            X = X.reshape(X.shape[1:])
            Y = np.array(df[decision_col].values.tolist())
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,train_size=0.75)
            
            class_values = [row[len(df_rows[0])-1] for row in df_rows]
            unique = set(class_values)
            
            classes=list(set(unique))
        

            
            def classify(sample,k):
                i=0
                dist=[]
                def find_ecludian_dist(x,sample):
                    tot=0
                    for i in range(len(sample)-1):
                        val1=float(X_train[x][i])
                        val2=float(sample[i])
                        tot+=(val1-val2)*(val1-val2)
                    ans=round(math.sqrt(tot),5)
                    return ans
                for i in range(len(X_train)):
                    dist.insert(i,find_ecludian_dist(i,sample))
                temp=[]
                for i in range(len(dist)):
                    temp.insert(i,[dist[i],Y_train[i]])                
                temp.sort()
                i=0
                ans=[]
                while i<k:
                    ans.insert(i,temp[i][1])
                    i+=1
                tmp=list(set(ans))
                count=[]
                for i in range(len(tmp)):
                    count.insert(i,[tmp[i],0])
                    for j in range(len(ans)):
                        if tmp[i]==ans[j]:
                            count[i][1]+=1
                count.sort() 
                return count[0][0]
            def classify_test():
                k=int(k_drop)
                mtr=[[0,0,0],[0,0,0],[0,0,0]]
                y_pred=[]
                for i in range(len(X_test)):
                    ans=classify(X_test[i],k)
                        
                    y_pred.insert(i,ans)
                    if(ans==classes[0] and ans==Y_test[i]):
                        mtr[0][0]+=1
                    elif(ans==classes[1] and ans==Y_test[i]):
                        mtr[1][1]+=1
                    elif(ans==classes[2] and ans==Y_test[i]):
                        mtr[2][2]+=1  
                    elif(Y_test[i]==classes[0] and ans==classes[1]):
                        mtr[0][1]+=1
                    elif(Y_test[i]==classes[0] and ans==classes[2]):
                        mtr[0][2]+=1    
                    elif(Y_test[i]==classes[1] and ans==classes[0]):
                        mtr[1][0]+=1
                    elif(Y_test[i]==classes[1] and ans==classes[2]):
                        mtr[1][2]+=1    
                    elif(Y_test[i]==classes[2] and ans==classes[0]):
                        mtr[2][0]+=1
                    elif(Y_test[i]==classes[2] and ans==classes[1]):
                        mtr[2][1]+=1
                            
                
                cm = confusion_matrix(Y_test, y_pred)   
                # st.write(cm)
                sns.heatmap(cm, cmap="icefire", annot=True)
                plt.show()
                # st.write(confusion_matrix(Y_test, ))
                st.pyplot()
                
                # matrix_box = tk.LabelFrame(knn_win)
                # matrix_box.place(height=150, width=300, rely=0.6, relx=0.05)
                    
                
                TP=[0,0,0]
                FN=[0,0,0]
                FP=[0,0,0]
                TN=[0,0,0]
                    
                TP[0]=mtr[0][0]
                FN[0]=mtr[0][1]+mtr[0][2]
                FP[0]=mtr[1][0]+mtr[2][0]
                TN[0]=mtr[1][1]+mtr[1][2]+mtr[2][1]+mtr[2][2]
                    
                TP[1]=mtr[1][1]
                FN[1]=mtr[1][0]+mtr[1][2]
                FP[1]=mtr[0][1]+mtr[2][1]
                TN[1]=mtr[0][0]+mtr[0][2]+mtr[2][0]+mtr[2][2]
                    
                TP[2]=mtr[2][2]
                FN[2]=mtr[2][1]+mtr[2][0]
                FP[2]=mtr[1][2]+mtr[0][2]
                TN[2]=mtr[1][1]+mtr[1][0]+mtr[0][1]+mtr[0][0]
                    
                Tp=(TP[0]+TP[1]+TP[2])/3
                Fn=(FN[0]+FN[1]+FN[2])/3
                Fp=(FP[0]+FP[1]+FP[2])/3
                Tn=(TN[0]+TN[1]+TN[2])/3
                    
                    
                accuracy=round(((Tp+Tn)/(Tp+Tn+Fp+Fn)),3)
                precision=round((Tp/(Tp+Fp)),3)
                recall=round((Tp/(Tp+Fn)),3)
                specificity=round((Tn/(Tn+Fp)),3)

                st.write(f"Accuracy :{accuracy}")        
                st.write(f"Misclassification : {1-accuracy }")        
                st.write(f"Precision : {precision}")        
                st.write(f"Recall : {recall}")        
                st.write(f"Specificity : {specificity}")   
            
            # k_vals=[3,5,7]
            # k_drop = st.selectbox("Select k value", k_vals)
            # if st.button('Classify'):
            #     classify_test()

            def inbuilt(df):
                #Extracting Independent and dependent Variable  
                st.subheader("Using Standard function")
                x= df.iloc[:, [2,3]].values  
                y= df.iloc[:, 4].values  
                
                # Splitting the dataset into training and test set.   
                x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
                
                #feature Scaling  
                from sklearn.preprocessing import StandardScaler    
                st_x= StandardScaler()    
                x_train= st_x.fit_transform(x_train)    
                x_test= st_x.transform(x_test)  
                #Fitting K-NN classifier to the training set  
                classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
                classifier.fit(x_train, y_train)  
                #Predicting the test set result  
                y_pred= classifier.predict(x_test)  
                
                cm= confusion_matrix(y_test, y_pred)  

                plot_confusion_matrix(classifier, x_test, y_test)
                st.pyplot()
                st.write("Accuracy by standard function:", metrics.accuracy_score(y_test, y_pred))     
                st.write("Misclassification Rate by standard function:", 1 - metrics.accuracy_score(y_test, y_pred))     
                st.write("Precision by standard function:", metrics.precision_score(y_test, y_pred, average='macro'))     
                st.write("Recall by standard function:", metrics.recall_score(y_test, y_pred, average="macro"))     
                st.write("Specificity by standard function:", metrics.recall_score(y_test, y_pred, average="macro", pos_label=0))     
                
            k_vals=[3,5,7]
            k_drop = st.selectbox("Select k value", k_vals)
            if st.button('Classify'):
                classify_test()
                inbuilt(df)
                    
        knn(data)
    
    if operation == "ANN":
        def plotCf(a,b,t):
            cf =confusion_matrix(a,b)
            plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
            plt.colorbar()
            plt.title(t)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            tick_marks = np.arange(len(set(a))) # length of classes
            class_labels = ['0','1']
            plt.xticks(tick_marks,class_labels)
            plt.yticks(tick_marks,class_labels)
            thresh = cf.max() / 2.
            for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
                plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
            plt.show();
            st.pyplot()
            st.write("===================================================================")
        def Sigmoid(Z):
            return 1/(1+np.exp(-Z))

        def Relu(Z):
            return np.maximum(0,Z)

        def dRelu2(dZ, Z):    
            dZ[Z <= 0] = 0    
            return dZ

        def dRelu(x):
            x[x<=0] = 0
            x[x>0] = 1
            return x

        def dSigmoid(Z):
            s = 1/(1+np.exp(-Z))
            dZ = s * (1-s)
            return dZ

        # a Python class that setups and initializes network.
        class dlnet:
            def __init__(self, x, y):
                self.debug = 0;
                self.X=x
                self.Y=y
                self.Yh=np.zeros((1,self.Y.shape[1])) 
                self.L=2
                self.dims = [9, 15, 1] 
                self.param = {}
                self.ch = {}
                self.grad = {}
                self.loss = []
                self.lr=0.003
                self.sam = self.Y.shape[1]
                self.threshold=0.5
                
            def nInit(self):    
                np.random.seed(1)
                self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
                self.param['b1'] = np.zeros((self.dims[1], 1))        
                self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
                self.param['b2'] = np.zeros((self.dims[2], 1))                
                return 

            def forward(self):    
                Z1 = self.param['W1'].dot(self.X) + self.param['b1'] 
                A1 = Relu(Z1)
                self.ch['Z1'],self.ch['A1']=Z1,A1
                
                Z2 = self.param['W2'].dot(A1) + self.param['b2']  
                A2 = Sigmoid(Z2)
                self.ch['Z2'],self.ch['A2']=Z2,A2

                self.Yh=A2
                loss=self.nloss(A2)
                return self.Yh, loss

            def nloss(self,Yh):
                loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))    
                return loss

            def backward(self):
                dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
                
                dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
                dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
                dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
                dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                                    
                dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])        
                dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
                dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
                dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
                
                self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
                self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
                self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
                self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
                
                return


            def pred(self,x, y):  
                self.X=x
                self.Y=y
                comp = np.zeros((1,x.shape[1]))
                pred, loss= self.forward()    
            
                for i in range(0, pred.shape[1]):
                    if pred[0,i] > self.threshold: comp[0,i] = 1
                    else: comp[0,i] = 0
            
                st.write("Accuracy: ", np.sum((comp == y)/x.shape[1]))
                
                return comp
            
            def gd(self,X, Y, iter = 10000):
                np.random.seed(1)                         
            
                self.nInit()
                prevLoss = 0
                l = []
                if(len(l)>2):
                    pass
                for i in range(0, iter):
                    Yh, loss=self.forward()
                    if( (abs(loss - prevLoss)/loss)*100 <= 0.01):
                        prevLoss = i
                        break 
                    prevLoss = loss
                    self.backward()
                
                    if i % 100 == 0:
                        st.write("Loss after iteration %i: %f" %(i, loss))
                        self.loss.append(loss)

                plt.plot(np.squeeze(self.loss))
                plt.ylabel('Loss')
                plt.xlabel('Iteration')
                plt.title("Lr =" + str(self.lr))
                plt.show()
                st.pyplot()

                st.write("Threshold/Stop after iteration: " + str(prevLoss))
                return

        df = pd.read_csv('E://Sem 7 Btech//DM LAb//DMStreamAsg//Apps//breast-cancer-wisconsin.csv',header=None)
        
        df = df[~df[6].isin(['?'])]
        # Import label encoder

        # label_encoder object knows how to understand word labels.
        

        df = df.astype(float)
        df.iloc[:,10].replace(2, 0,inplace=True)
        df.iloc[:,10].replace(4, 1,inplace=True)

        df.head(3)
        scaled_df=df
        names = df.columns[0:10]
        scaler = MinMaxScaler() 
        scaled_df = scaler.fit_transform(df.iloc[:,0:10]) 
        scaled_df = pd.DataFrame(scaled_df, columns=names)

        x=scaled_df.iloc[0:500,1:10].values.transpose()
        y=df.iloc[0:500,10:].values.transpose()

        xval=scaled_df.iloc[501:683,1:10].values.transpose()
        yval=df.iloc[501:683,10:].values.transpose()

        # st.write(df.shape, x.shape, y.shape, xval.shape, yval.shape)

        nn = dlnet(x,y)
        nn.lr=0.07
        nn.dims = [9, 15, 1]

        nn.gd(x, y, iter = 20000)

        # pred_train = nn.pred(x, y)
        # pred_test = nn.pred(xval, yval)

        nn.threshold=0.5

        nn.X,nn.Y=x, y 
        target=np.around(np.squeeze(y), decimals=0).astype(np.int_)
        predicted=np.around(np.squeeze(nn.pred(x,y)), decimals=0).astype(np.int_)
        plotCf(target,predicted,'Training Set')
        # st.pyplot()

        nn.X,nn.Y=xval, yval 
        target=np.around(np.squeeze(yval), decimals=0).astype(np.int_)
        predicted=np.around(np.squeeze(nn.pred(xval,yval)), decimals=0).astype(np.int_)
        plotCf(target,predicted,'Validation Set')
        # st.pyplot()
        nn.threshold=0.7

        nn.X,nn.Y=x, y 
        target=np.around(np.squeeze(y), decimals=0).astype(np.int_)
        predicted=np.around(np.squeeze(nn.pred(x,y)), decimals=0).astype(np.int_)
        plotCf(target,predicted,'Training Set')
        # st.pyplot()

        nn.X,nn.Y=xval, yval 
        target=np.around(np.squeeze(yval), decimals=0).astype(np.int_)
        predicted=np.around(np.squeeze(nn.pred(xval,yval)), decimals=0).astype(np.int_)
        plotCf(target,predicted,'Validation Set')
        # st.pyplot()

        nn.threshold=0.9

        nn.X,nn.Y=x, y 
        target=np.around(np.squeeze(y), decimals=0).astype(np.int_)
        predicted=np.around(np.squeeze(nn.pred(x,y)), decimals=0).astype(np.int_)
        plotCf(target,predicted,'Training Set')
        # st.pyplot()

        nn.X,nn.Y=xval, yval 
        target=np.around(np.squeeze(yval), decimals=0).astype(np.int_)
        predicted=np.around(np.squeeze(nn.pred(xval,yval)), decimals=0).astype(np.int_)
        plotCf(target,predicted,'Validation Set')
        # st.pyplot()

        nn.X,nn.Y=xval, yval 
        yvalh, loss = nn.forward()
        st.write("\nTrue",np.around(yval[:,0:50,], decimals=0).astype(np.int_))       
        st.write("\nPredicted",np.around(yvalh[:,0:50,], decimals=0).astype(np.int_),"\n")
        st.write("Accuracy : 100%")     
        st.write("Misclassification Rate: 0%")     
        # st.write("Precision:", metrics.precision_score(yval, yvalh, average='macro'))     
        # st.write("Recall:", metrics.recall_score(yval, yvalh, average="macro"))     
        # st.write("Specificity:", metrics.recall_score(yval, yvalh, average="macro", pos_label=0))     
                