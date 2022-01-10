'''
@func:naive bayes classifier
@create:2021.11.6
'''

from numpy import int16
import warnings
warnings.filterwarnings('ignore')


def NaiveBayesClassifierx():
    import streamlit as st


    # 用贝叶斯定理来进行建模的监督学习分类器
    # cross_validation被替换为model_selection
    # from sklearn.cross_validation import train_test_split 替换为：from sklearn.model_selection import train_test_split
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.naive_bayes import GaussianNB


    # visualize classification results:
    def plot_classifier(classifier, X, y):
        # define ranges to plot the figure 
        x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
        y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0
        

        # denotes the step size that will be used in the mesh grid
        step_size = 0.01

        # define the mesh grid
        x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

        # compute the classifier output
        mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

        # reshape the array
        mesh_output = mesh_output.reshape(x_values.shape)

        # Plot the output using a colored plot 
        fig,ax = plt.subplots()

        # choose a color scheme you can find all the options based on mesh_output
        # here: http://matplotlib.org/examples/color/colormaps_reference.html
        ax.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.cividis)

        # Overlay the training points on the plot 
        ax.scatter(X[:, 0], X[:, 1], c=y, s=60, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

        # specify the boundaries of the figure
        plt.xlim(x_values.min(), x_values.max())
        plt.ylim(y_values.min(), y_values.max())

        # specify the ticks on the X and Y axes
        plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
        plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

        # plt.show()
        st.pyplot(fig)



    input_file = 'data/data_multivar(2).txt'

    X = []
    y = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            X.append(data[:-1])
            y.append(data[-1]) 

    X = np.array(X)
    y = np.array(y)

    # plt.scatter(X[:,0],y,c='r',marker='o')
    # plt.show()
 

    classifier_gaussiannb = GaussianNB()
    classifier_gaussiannb.fit(X, y)
    y_pred = classifier_gaussiannb.predict(X)


    # compute accuracy of the classifier
    accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
    st.write("(Before train/test split) Accuracy of the classifier =", round(accuracy, 2), "%")

    # plot_classifier(classifier_gaussiannb, X, y)

    ###############################################
    # Train test split
    from sklearn import model_selection

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
    classifier_gaussiannb_new = GaussianNB()
    classifier_gaussiannb_new.fit(X_train, y_train)
    y_test_pred = classifier_gaussiannb_new.predict(X_test)

    # compute accuracy of the classifier
    accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
    st.write("(After train/test split) Accuracy of the classifier =", round(accuracy, 2), "%")

    # plot_classifier(classifier_gaussiannb_new, X_test, y_test)

    ###############################################
    # Cross validation and scoring functions

    st.write('(After apply cross-validation ↓)')

    num_validations = 5
    accuracy = model_selection.cross_val_score(classifier_gaussiannb, 
            X, y, scoring='accuracy', cv=num_validations)
    st.write("Accuracy: ",round(100*accuracy.mean(), 2), "%")

    f1 = model_selection.cross_val_score(classifier_gaussiannb, 
            X, y, scoring='f1_weighted', cv=num_validations)
    st.write("F1: ",round(100*f1.mean(), 2), "%")

    precision = model_selection.cross_val_score(classifier_gaussiannb, 
            X, y, scoring='precision_weighted', cv=num_validations)
    st.write("Precision: ",round(100*precision.mean(), 2), "%")

    recall = model_selection.cross_val_score(classifier_gaussiannb, 
            X, y, scoring='recall_weighted', cv=num_validations)
    st.write("Recall: ",round(100*recall.mean(), 2),"%")


    plot_classifier(classifier_gaussiannb_new, X_test, y_test)


# NaiveBayesClassifierx()