'''
@func: Random Forest Regressor
@create: 2021-11-1
'''

from matplotlib.pyplot import subplot
import streamlit as st


def RandomForestRegressorx1():

    # 随机森林是一个决策树集合，它基本上就是用一组由数据集的若干子集构建的决策树构成。
    # 再用决策树平均值改善整体学习效果
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor 
    from sklearn import preprocessing
    from sklearn.metrics import mean_squared_error, explained_variance_score
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt

    def plot_feature_importances(feature_importances, title, feature_names):
        # Normalize the importance values 
        feature_importances = 100.0 * (feature_importances / max(feature_importances))

        # Sort the values and flip them
        index_sorted = np.flipud(np.argsort(feature_importances))

        # Arrange the X ticks
        pos = np.arange(index_sorted.shape[0]) + 0.5

        # Plot the bar graph
        fig,ax = plt.subplots()
        ax.bar(pos, feature_importances[index_sorted], align='center',color='blue')
        plt.xticks(pos, feature_names[index_sorted],rotation=30)
        plt.ylabel('Relative Importance')
        plt.grid(alpha=0.3)
        plt.title(title)
        st.pyplot(fig)



    # Load the dataset from the input file
    # X, y, feature_names = load_dataset('work/bike_day.csv')
    df = pd.read_csv('data/bike_day.csv')
    X = np.array(df.iloc[0:len(df),2:13])
    y = np.array(df.iloc[0:len(df),-1])
    feature_names = np.array(list(df.iloc[:,2:13]))
    st.dataframe(df)

    X, y = shuffle(X, y, random_state=7) 

    # Split the data 80/20 (80% for training, 20% for testing)
    num_training = int(0.9 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # Fit Random Forest regression model
    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
    rf_regressor.fit(X_train, y_train)

    # Evaluate performance of Random Forest regressor
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred) 
    st.write("\n#### Random Forest regressor performance ####")
    st.write("Mean squared error =", round(mse, 2))
    st.write("Explained variance score =", round(evs, 2))

    # Plot relative feature importances 
    plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest Regressor(bike_day.csv)', feature_names)





def RandomForestRegressorx2():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor 
    from sklearn import preprocessing
    from sklearn.metrics import mean_squared_error, explained_variance_score
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt

    def plot_feature_importances(feature_importances, title, feature_names):
        # Normalize the importance values 
        feature_importances = 100.0 * (feature_importances / max(feature_importances))

        # Sort the values and flip them
        index_sorted = np.flipud(np.argsort(feature_importances))

        # Arrange the X ticks
        pos = np.arange(index_sorted.shape[0]) + 0.5

        # Plot the bar graph
        fig,ax = plt.subplots()
        ax.bar(pos, feature_importances[index_sorted], align='center',color='blue')
        plt.xticks(pos, feature_names[index_sorted],rotation=30)
        plt.ylabel('Relative Importance')
        plt.grid(alpha=0.3)
        plt.title(title)
        st.pyplot(fig)


    # Load the dataset from the input file
    # X, y, feature_names = load_dataset('work/bike_day.csv')
    df = pd.read_csv('data/bike_hour.csv')
    X = np.array(df.iloc[0:len(df),2:14])
    y = np.array(df.iloc[0:len(df),-1])
    feature_names = np.array(list(df.iloc[:,2:14]))
    st.dataframe(df)


    X, y = shuffle(X, y, random_state=7) 

    # Split the data 80/20 (80% for training, 20% for testing)
    num_training = int(0.9 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # Fit Random Forest regression model
    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
    rf_regressor.fit(X_train, y_train)

    # Evaluate performance of Random Forest regressor
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred) 
    st.write("\n#### Random Forest regressor performance ####")
    st.write("Mean squared error =", round(mse, 2))
    st.write("Explained variance score =", round(evs, 2))

    # Plot relative feature importances 
    plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest Regressor(bike_hour.csv)', feature_names)

# RandomForestRegressorx1()
# RandomForestRegressorx2()