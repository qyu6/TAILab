'''
@func:apply ridge regression
@create:2021.10.25
'''

def RidgeRegx():
    import streamlit as st
    
    # 岭回归|通过引入正则化项的系数作为阈值来消除异常值的影响(普通最小二乘法会将每个点都考虑进去，进而带来误差)
    import sys
    import numpy as np

    # filename = sys.argv[1]
    X = []
    y = []
    with open('data/data_multivar.txt', 'r') as f:
        for line in f.readlines():
            data = [float(i) for i in line.split(',')]
            xt, yt = data[:-1], data[-1]
            X.append(xt)
            y.append(yt)

    # alpha控制岭回归器的复杂程度。Alpha=0，相当于普通最小二乘法，如果希望对异常值不那么敏感，则增加alpha的值
    # Train/test split
    num_training = int(0.8 * len(X))
    num_test = len(X) - num_training

    # Training data
    #X_train = np.array(X[:num_training]).reshape((num_training,1))
    X_train = np.array(X[:num_training])
    y_train = np.array(y[:num_training])

    # Test data
    #X_test = np.array(X[num_training:]).reshape((num_test,1))
    X_test = np.array(X[num_training:])
    y_test = np.array(y[num_training:])

    # Create linear regression object
    from sklearn import linear_model

    linear_regressor = linear_model.LinearRegression()
    ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)

    # Train the model using the training sets
    linear_regressor.fit(X_train, y_train)
    ridge_regressor.fit(X_train, y_train)

    # Predict the output
    y_test_pred = linear_regressor.predict(X_test)
    y_test_pred_ridge = ridge_regressor.predict(X_test)

    # Measure performance
    import sklearn.metrics as sm

    st.write("LINEAR:")
    st.write("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
    st.write("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
    st.write("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
    st.write("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
    st.write("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

    st.write("\nRIDGE:")
    st.write("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2)) 
    st.write("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2)) 
    st.write("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2)) 
    st.write("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2)) 
    st.write("R2 score =", round(sm.r2_score(y_test, y_test_pred_ridge), 2))


# test
# RidgeRegx()