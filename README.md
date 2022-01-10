# TAILab
T.的人工智能实验室 (Welcome issue or PR)

![Forks](https://img.shields.io/github/forks/qyu6/TAILab?color=red) ![Stars](https://img.shields.io/github/stars/qyu6/TAILab) ![code size](https://img.shields.io/github/commit-activity/m/qyu6/TAILab?color=yellow) ![code size](https://img.shields.io/github/last-commit/qyu6/TAILab?color=purple) ![code size](https://img.shields.io/github/repo-size/qyu6/TAILab?&color=gree) ![code size](https://img.shields.io/github/license/qyu6/TAILab?color=lightgrey)

------

### 1.TAILab平台地址|Web Address

[Streamlit](https://share.streamlit.io/tqthooo2021/tai-lab/main/labmain.py) (手机端/电脑端均可使用, Access Code: tailab123456)

### 2.TAILab平台主模块框架|Lab Framework

```
<TAILab Framework> *(key words...)
├── [1]-在线Python编程环境 (在线python编译环境，scipts/API/etc)
├── [2]-统计学模型API[...]
├── [3]-机器学习模型API
│   ├── 通用模块
│   │   ├── 数据预处理 (标准化/归一化/缩放/二值化/One-Hot/labeling..)
│   └── 回归模型
│   │   ├── 线性回归器 (线性回归/模型评价标准:MAE,MSE,EVS,R^2)
│   │   └── 岭回归器 (岭回归/L2正则项)
│   │   └── 多项式回归器 (多项式回归/SGD随机梯度下降)
│   │   └── AdaBoost决策树回归器 (自适应提升树回归方法)
│   │   └── 随机森林回归器 (随机森林回归/特征重要性可视化/ensemble learning)
│   └── 分类模型
│   │   ├── 逻辑回归分类器 (逻辑回归分类/二维分类可视化)
│   │   └── 朴素贝叶斯分类器 (贝叶斯分类/训练测试数据集split/交叉验证/分类性能指标:accuracy,f1-score,precision,recall)
│   │   └── 高斯朴素贝叶斯分类器 (高斯朴素贝叶斯，手写数字数据集)
│   └── 聚类模型[...]
│   └── [...]
├── [4]-深度学习模型API[...]
├── [5]-训练开发工具☆
│   ├── OCR-光学字符识别
│   └── Tree-文件结构可视化
├── [6]-其他
│   ├── [1]-主流库 (不同主流库API一站式搜索查询功能)
│   └── [2]-论文期刊
│   └── [3]-有用链接
│   └── [4]-代码技巧 (tips:python/Git/SQL/Vim...)
└────── [...]
```

### 3.开发日志|Dev Log

* https://github.com/tqthooo2021/TAI-Lab/blob/main/DevLog.md

### 4.致谢|Ackonwledgement

* [Streamlit - Streamlit is the fastest way to build data apps.](https://discuss.streamlit.io/)
* [Streamlit documentation](https://docs.streamlit.io/)
* [Sign into Streamlit · Streamlit](https://share.streamlit.io/) (Deploy: Use Github accoutn log in, select public repository and main.py file which used for streamlit run xx/xx/...)
* [GitHub - PacktPublishing/Python-Machine-Learning-Cookbook: Code files for Python-Machine-Learning-Cookbook](https://github.com/PacktPublishing/Python-Machine-Learning-Cookbook)
* [Shields.io: Quality metadata badges for open source projects](https://shields.io/)
* [A possible design for doing per-session persistent state in Streamlit · GitHub](https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92)
