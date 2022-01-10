'''
@func:polynomial regression
@create:2021.10
'''
def PolyRegx():
    import streamlit as st
    # 多项式回归
    import sys
    import numpy as np
    from sklearn import linear_model
    

    # filename = sys.argv[1]
    X = []
    y = []
    with open('data/data_multivar.txt', 'r') as f:
        for line in f.readlines():
            data = [float(i) for i in line.split(',')]
            xt, yt = data[:-1], data[-1]
            X.append(xt)
            y.append(yt)
    
    # Train/test split
    num_training = int(0.8 * len(X))
    num_test = len(X) - num_training

    # Training data
    #X_train = np.array(X[:num_training]).reshape((num_training,1))
    X_train = np.array(X[:num_training])
    y_train = np.array(y[:num_training])

    st.write(X_train)
    st.write(y_train)

    # Test data
    #X_test = np.array(X[num_training:]).reshape((num_test,1))
    X_test = np.array(X[num_training:])
    y_test = np.array(y[num_training:])

    from sklearn.preprocessing import PolynomialFeatures
    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(X_train, y_train)

    polynomial = PolynomialFeatures(degree=10)
    X_train_transformed = polynomial.fit_transform(X_train)
    print(X_train)
    datapoint = [[0.39,2.78,7.11],[0.39,2.78,7.11]]
    poly_datapoint = polynomial.fit_transform(datapoint)

    poly_linear_model = linear_model.LinearRegression()
    poly_linear_model.fit(X_train_transformed, y_train)
    st.write("\nLinear regression:\n", linear_regressor.predict(datapoint))
    st.write("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))


    # (SGD=随机梯度下降)Stochastic Gradient Descent regressor:随机梯度下降.是不将所有样本都进行计算后再更新参数，
    # 而是选取一个样本，计算后就更新参数)
    # (BGD=批量梯度下降)将所有样本数据都计算完成后再更新参数的方法叫做批量梯度下降
    sgd_regressor = linear_model.SGDRegressor(loss='huber', n_iter_no_change=50)
    sgd_regressor.fit(X_train, y_train)
    st.write("\nSGD regressor:\n", sgd_regressor.predict(datapoint))





# test
# PolyRegx()