'''
@func:linear regression
@create:2021.10.24
'''

from textwrap import wrap


def linearregx():
    import streamlit as st
    # 线性回归器。为了实现线性输出与实际输出的残差平方和最小(sum of squares of differences),普通最小二乘法 - (OLS:Oridinary Least Squares)
    import sys
    import numpy as np

    X = [1,2,3,4,6,7,8,9,11,13,14,15,15,16,17,18,19,20]
    y = [12,23,34,45,56,67,78,80,100,30,114,124,131,142,151,167,50,40]

    col1,col2 = st.columns(2)
    col1.write('X value')
    col1.write(np.array(X))
    col2.write('y value')
    col2.write(np.array(y))


    num_training = int(0.8*len(X))
    num_test = len(X) - num_training

    X_train = np.array(X[:num_training]).reshape((num_training,1))
    y_train = np.array(y[:num_training])

    X_test = np.array(X[num_training:]).reshape((num_test,1))
    y_test = np.array(y[num_training:])


    col1,col2,col3,col4 = st.columns(4)
    col1.write('X_train')
    col1.write(X_train)
    col2.write('y_train')
    col2.write(y_train)
    col3.write('X_test')
    col3.write(X_test)
    col4.write('y_test')
    col4.write(y_test)


    from sklearn import linear_model
    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(X_train,y_train)

    import matplotlib.pyplot as plt
    y_train_pred = linear_regressor.predict(X_train)
    fig1,ax1 = plt.subplots()
    ax1.grid(alpha=0.4)
    ax1.scatter(X_train,y_train,color='green')
    ax1.plot(X_train,y_train_pred,color='black',linewidth = 1)
    plt.title('Training data')
    st.pyplot(fig1)

    y_test_pred = linear_regressor.predict(X_test)
    fig2,ax2 = plt.subplots()
    ax2.grid(alpha=0.4)
    ax2.scatter(X_test,y_test,color='red')
    ax2.plot(X_test,y_test_pred,color='black',linewidth = 1)
    plt.title('Test data')
    st.pyplot(fig2)

    # 评价回归拟合器误差，几个重要的指标(metrics)
    # 平均绝对误差-MAE:mean absolute error
    # 均方误差-MES:mean squared error,误差平方均值
    # 中位数绝对误差-median absolute error,用中位数可以消除异常值outlier的干扰
    # 解释方差分-EVS:explained variance score,对数据集波动的解释能力
    # R^2得分-拟合优度
    # [通常做法]:保证均方误差最低，解释方差分最高
    import sklearn.metrics as sm
    st.write("Mean absolute error=",round(sm.mean_absolute_error(y_test,y_test_pred),2))
    st.write("Mean squared error=",round(sm.mean_squared_error(y_test,y_test_pred),2))
    st.write("Median absolute error=",round(sm.median_absolute_error(y_test,y_test_pred),2))
    st.write("Explained variance score=",round(sm.explained_variance_score(y_test,y_test_pred),2))
    st.write("R^2 score=",round(sm.r2_score(y_test,y_test_pred),2))



    # # 以下内容测试没有通过。一种保存模型数据的方法
    # # 保存模型数据:
    # import cPickle as pickle
    # output_model_file = 'saved_model.pkl'
    # with open(output_model_file,'w') as f:
    #     pickle.dump(linear_regressor,f)

    # # 加载并使用数据：
    # with open(output_model_file,'r') as f:
    #     model_linregr = pickle.load(f)

    # y_test_pred_new = model_linregr.predict(X_test)
    # print('\nNew mean absolute error=',round(sm.mean_absolute_error(y_test,y_test_pred_new),2))




# test
# linearregx()