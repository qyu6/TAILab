'''
@func: AdaBoost Decision Tree Regressor
@create: 2021.10.26
'''

def AdaboostDTRegressorx():
    import streamlit as st
    
    # 基于AdaBoost的决策树回归器. 
    # Adaboost = Adaptive Boosting是指自适应增强算法，是一种利用其它系统增强模型准确性的技术。
    # 这种技术基于是将不同的算法进行组合，用加权汇总的方式获得最终结果，被称为弱学习器(weak learners)
    # AdaBoost算法在每个阶段获取的信息都会反馈到模型中，这样学习器就可以在后一阶段训练难以分类的样本
    # 首先使用AdaBoost算法对数据集进行回归拟合，再计算误差，然后根据误差评估结果，用同样数据集重新拟合。可看做是回归器的调优过程，直到达到预期的准确性
    import numpy as np

    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import datasets
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
        plt.grid(alpha=0.2)
        plt.title(title)
        st.pyplot(fig)

    # if __name__=='__main__':
    # Load housing data
    housing_data = datasets.load_boston() 

    # Shuffle the data
    st.write(housing_data)
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    # Split the data 80/20 (80% for training, 20% for testing)
    num_training = int(0.8 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # Fit decision tree regression model
    dt_regressor = DecisionTreeRegressor(max_depth=4)
    dt_regressor.fit(X_train, y_train)

    # Fit decision tree regression model with AdaBoost
    ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
    ab_regressor.fit(X_train, y_train)

    # Evaluate performance of Decision Tree regressor
    y_pred_dt = dt_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_dt)
    evs = explained_variance_score(y_test, y_pred_dt) 
    st.write("\n#### Decision Tree Performance ####")
    st.write("Mean squared error =", round(mse, 2))
    st.write("Explained variance score =", round(evs, 2))

    # Evaluate performance of AdaBoost
    y_pred_ab = ab_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_ab)
    evs = explained_variance_score(y_test, y_pred_ab) 
    st.write("\n#### AdaBoost Performance ####")
    st.write("Mean squared error =", round(mse, 2))
    st.write("Explained variance score =", round(evs, 2))

    # Plot relative feature importances 
    plot_feature_importances(dt_regressor.feature_importances_, 
            'Decision Tree Regressor', housing_data.feature_names)
    plot_feature_importances(ab_regressor.feature_importances_, 
            'AdaBoost Regressor', housing_data.feature_names)


# test
# AdaboostDTRegressorx()