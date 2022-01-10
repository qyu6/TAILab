'''
@func:show confusion matrix visualizes
@create:2021-11-7
'''

def ConfusionMatrixx():
    import streamlit as st
    from contextlib import contextmanager,redirect_stdout
    from io import StringIO
    from time import sleep


    # 显示终端命令行的原始格式信息到前端
    @contextmanager
    def st_capture(output_func):
        with StringIO() as stdout,redirect_stdout(stdout):
            old_write = stdout.write

            def new_write(string):
                ret = old_write(string)
                output_func(stdout.getvalue())
                return ret

            stdout.write = new_write
            yield

    # 混淆矩阵
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Show confusion matrix
    def plot_confusion_matrix(confusion_mat):
        fig,ax = plt.subplots()
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(4)
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.show()
        st.pyplot(fig)


    y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
    y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
    confusion_mat = confusion_matrix(y_true, y_pred)
    
    plot_confusion_matrix(confusion_mat)

    # Print classification report
    from sklearn.metrics import classification_report
    target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']

    outputx = st.empty()
    with st_capture(outputx.code):
        print(classification_report(y_true, y_pred, target_names=target_names))







# test:
# ConfusionMatrixx()
