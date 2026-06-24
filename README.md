1. 项目简介

本项目基于 arXiv 计算机领域论文数据，完成论文研究方向自动分类任务。项目通过 arXiv API 获取论文标题、摘要、类别、作者和发布时间等信息，并利用 TF-IDF 方法将论文文本转换为高维稀疏特征。实验比较了 MultinomialNB、LogisticRegression、LinearSVC 和 LogisticRegression_GridSearchCV 等机器学习模型，并使用 Accuracy、Precision macro、Recall macro 和 F1 macro 对模型性能进行评估。

本项目最终选择 LogisticRegression_GridSearchCV 作为最优模型，对 cs.AI、cs.LG、cs.CV、cs.CL 和 cs.SE 五类 arXiv 论文进行研究方向分类。

2. 项目结构
<img width="491" height="467" alt="image" src="https://github.com/user-attachments/assets/7a5c8e21-0c89-4529-9170-68643e961f96" />

3. 环境配置

建议使用 Python 3.9 及以上版本运行本项目。

方式一：使用 pip 安装依赖

pip install requests lxml pandas numpy matplotlib scikit-learn

方式二：使用 Anaconda 创建虚拟环境

conda create -n arxiv_cls python=3.10
conda activate arxiv_cls
pip install requests lxml pandas numpy matplotlib scikit-learn

4. 主要依赖库

requests       # 请求 arXiv API
lxml           # 解析 arXiv 返回的 XML 数据
pandas         # 数据读取、清洗和表格处理
numpy          # 数值计算
matplotlib     # 绘制实验图表
scikit-learn   # TF-IDF 特征提取、机器学习建模、参数调优与模型评估

5. 数据准备

本项目数据来源于 arXiv 官方 API，主要采集以下五类计算机领域论文：

cs.AI  人工智能
cs.LG  机器学习
cs.CV  计算机视觉
cs.CL  自然语言处理
cs.SE  软件工程

由于完整数据文件体积较大，仓库中未直接上传完整 clean_arxiv.csv。用户可以通过运行代码重新采集并生成数据。

运行数据爬取脚本：

python 01_crawl_arxiv.py

运行数据清洗脚本：

python 02_clean_data.py

数据清洗后会生成用于模型训练的数据文件。清洗过程包括去重、缺失值处理、字段整理、标题与摘要拼接、数值辅助特征构造等步骤。

6. 运行步骤

第一步：安装依赖

pip install requests lxml pandas numpy matplotlib scikit-learn

第二步：爬取 arXiv 论文数据

python 01_crawl_arxiv.py

该脚本会从 arXiv API 获取论文标题、摘要、类别、作者、发布时间等原始信息。

第三步：清洗数据并构造特征

python 02_clean_data.py

该脚本会对原始数据进行清洗，并构造以下字段：

text              # 标题与摘要拼接文本
title_length      # 标题长度
summary_length    # 摘要长度
author_count      # 作者数量
category_count    # 论文类别标签数量
published_year    # 发布年份
published_month   # 发布月份

第四步：训练模型并生成实验结果

python 03_modeling.py

该脚本会完成以下任务：

1. 读取清洗后的论文数据
2. 过滤得到五分类实验数据
3. 使用 TF-IDF 提取文本特征
4. 训练 MultinomialNB、LogisticRegression、LinearSVC 等模型
5. 使用 GridSearchCV 对 LogisticRegression 进行参数调优
6. 输出 Accuracy、Precision macro、Recall macro 和 F1 macro 等指标
7. 生成模型对比图、混淆矩阵、学习曲线、参数调优热力图和特征消融图

7. 输出结果

运行 03_modeling.py 后，将在 figures/ 和 results/ 文件夹中生成结果文件。

figures 文件夹

category_distribution.png       # 五类论文数量分布图
model_metrics_bar.png           # 不同模型性能对比图
gridsearch_heatmap.png          # GridSearchCV 参数调优热力图
gridsearch_c_curve.png          # 正则化参数 C 影响曲线
learning_curve.png              # 学习曲线
confusion_matrix.png            # 最优模型混淆矩阵
feature_ablation_impact.png     # 特征消融分析图

results 文件夹

classification_report.txt       # 最优模型分类报告
model_metrics.csv               # 各模型 Accuracy、Precision、Recall、F1 指标
best_model_params.txt           # GridSearchCV 最优参数
gridsearch_results.csv          # 网格搜索完整结果
feature_ablation_results.csv    # 特征消融实验结果
model_analysis_summary.txt      # 模型综合分析摘要

8. 模型说明

本项目主要比较以下模型：

MultinomialNB：
多项式朴素贝叶斯模型，适合作为文本分类基线模型。
LogisticRegression：
逻辑回归模型，适合处理 TF-IDF 生成的高维稀疏文本特征。
LinearSVC：
线性支持向量机模型，适用于高维文本分类任务。
LogisticRegression_GridSearchCV：
在 LogisticRegression 基础上进行参数搜索，调优 TF-IDF 特征数量、ngram 范围和正则化参数 C。

9. 复现实验结果

完整复现实验可按以下顺序执行：

pip install requests lxml pandas numpy matplotlib scikit-learn
python 01_crawl_arxiv.py
python 02_clean_data.py
python 03_modeling.py

若不重新爬取完整数据，也可直接查看仓库中的 figures/ 和 results/ 文件夹，其中保存了本次实验生成的主要图表和模型结果。

10. 项目说明

本项目为人工智能课程论文配套代码。项目代码已上传至 GitHub 仓库，便于查看源码、运行步骤和实验结果。实验结果和运行说明均通过本 README 文件进行说明。
