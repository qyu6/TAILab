'''
@func:show terminal content
@create:2021.11.8
'''


def ShowTerminalContent(content):
    from contextlib import contextmanager,redirect_stdout
    from io import StringIO
    from time import sleep
    import streamlit as st

    # 使用时需要将如下函数放入到1级自定义函数中，嵌套函数将失效，比如：(def(def(xx)))
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
    

    # 以下为输出终端内容调用格式
    output = st.empty()
    with st_capture(output.code):
        print(content)