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


def app(dataset):

    st.header("Assignment 7")

    url = 'https://raw.githubusercontent.com/Udayraj2806/dataset/main/house-votes-84.data.csv'
    df = pd.read_csv(url)

    # st.write(df[:5])
    d = pd.DataFrame(df)
    data = d
    d.head()

    df_rows = d.to_numpy().tolist()

    cols = []
    for i in data.columns:
        cols.append(i)
    st.write(cols)
    col_len = len(cols)
    st.write("At Max Rules to be Generated: ",
             ((3**col_len)-(2**(col_len+1)))+1)
    st.write("Attributes:", len(cols))
    # st.write(cols)
    newDataSet = []
    # st.write(len(df_rows))
    i, cnt = 0, 0
    for row in df_rows:
        i += 1
        if '?' in row:
            continue
        else:
            lst = []
            cnt += 1
            for k in range(1, len(row)):
                if row[k] == 'y':
                    lst.append(cols[k])
            newDataSet.append(lst)
    st.write(newDataSet)

    # st.write(row)
    # st.write("--------------")
    # st.write(cnt)
    # st.write(newDataSet)
    # newDataSet.drop()

    data = []

    for i in range(len(newDataSet)):
        # data[i] = newDataSet[i]
        data.append([i, newDataSet[i]])

    st.write(data)

    # extract distinct items

    init = []
    for i in data:
        for q in i[1]:
            if (q not in init):
                init.append(q)
    init = sorted(init)

    st.write("Init:", len(init))

    # st.write(init)

    sp = 0.4
    s = int(sp*len(init))
    s

    from collections import Counter
    c = Counter()
    for i in init:
        for d in data:
            if (i in d[1]):
                c[i] += 1
    # st.write("C1:")
    for i in c:
        pass
        # st.write(str([i])+": "+str(c[i]))
    # st.write()
    l = Counter()
    for i in c:
        if (c[i] >= s):
            l[frozenset([i])] += c[i]
    # st.write("L1:")
    for i in l:
        pass
        # st.write(str(list(i))+": "+str(l[i]))
    # st.write()
    pl = l
    pos = 1
    for count in range(2, 1000):
        nc = set()
        temp = list(l)
        for i in range(0, len(temp)):
            for j in range(i+1, len(temp)):
                t = temp[i].union(temp[j])
                if (len(t) == count):
                    nc.add(temp[i].union(temp[j]))
        nc = list(nc)
        c = Counter()
        for i in nc:
            c[i] = 0
            for q in data:
                temp = set(q[1])
                if (i.issubset(temp)):
                    c[i] += 1
        # st.write("C"+str(count)+":")
        for i in c:
            pass
            # st.write(str(list(i))+": "+str(c[i]))
        # st.write()
        l = Counter()
        for i in c:
            if (c[i] >= s):
                l[i] += c[i]
        # st.write("L"+str(count)+":")
        for i in l:
            pass
            # st.write(str(list(i))+": "+str(l[i]))
        # st.write()
        if (len(l) == 0):
            break
        pl = l
        pos = count
    st.write("Result: ")
    st.write("L"+str(pos)+":")

    for i in pl:
        st.write(str(list(i))+": "+str(pl[i]))

    st.subheader("Rules Generation")
    for l in pl:
        st.write(l)
        break
    from itertools import combinations
    for l in pl:
        cnt = 0
        c = [frozenset(q) for q in combinations(l, len(l)-1)]
        mmax = 0
        for a in c:
            b = l-a
            ab = l
            sab = 0
            sa = 0
            sb = 0
            for q in data:
                temp = set(q[1])
                if (a.issubset(temp)):
                    sa += 1
                if (b.issubset(temp)):
                    sb += 1
                if (ab.issubset(temp)):
                    sab += 1
            temp = sab/sa*100
            if (temp > mmax):
                mmax = temp
            temp = sab/sb*100
            if (temp > mmax):
                mmax = temp
            cnt += 1
            st.write(str(cnt) + str(list(a))+" -> " +
                     str(list(b))+" = "+str(sab/sa*100)+"%")
            cnt += 1
            st.write(str(cnt) + str(list(b))+" -> " +
                     str(list(a))+" = "+str(sab/sb*100)+"%")
        mmax = st.number_input('Select value of alpha', step=5, min_value=5)
        mmax = int(mmax)
        curr = 1
        st.write("choosing:", end=' ')

        for a in c:
            b = l-a
            ab = l
            sab = 0
            sa = 0
            sb = 0
            for q in data:
                temp = set(q[1])
                if (a.issubset(temp)):
                    sa += 1
                if (b.issubset(temp)):
                    sb += 1
                if (ab.issubset(temp)):
                    sab += 1
            temp = sab/sa*100
            if (temp >= mmax):
                st.write(curr, end=' ')
            curr += 1
            temp = sab/sb*100
            if (temp >= mmax):
                st.write(curr, end=' ')
            curr += 1
            # break
        st.write()
        st.write()
        break
