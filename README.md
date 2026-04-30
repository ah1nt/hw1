# EuroSAT MLP Classifier

这是一个基于 NumPy 手工搭建的三层神经网络 (MLP) 图像分类器，用于在 EuroSAT 数据集上实现土地覆盖分类。本作业未使用 PyTorch/TensorFlow 等自动微分框架，自主实现了前向传播、自动微分与反向传播。

## 环境依赖
本项目的实现仅依赖标准科学计算和图像处理库：
- `numpy` (用于矩阵运算和实现神经网络)
- `Pillow` (用于加载和处理图像)
- `scikit-learn` (用于划分训练集、验证集和测试集，以及计算混淆矩阵)
- `matplotlib` (用于可视化 Loss/Accuracy 曲线和隐藏层权重)

安装依赖：
```bash
pip install numpy Pillow scikit-learn matplotlib
```

## 文件结构
- `dataset.py`：数据加载与预处理模块（图像展平、归一化、划分数据集）。
- `model.py`：模型定义模块（定义了 `Layer`, `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `CrossEntropyLoss` 及 `SGDOptimizer` 和 `MLP` 类）。
- `train.py`：训练循环模块（包含学习率衰减、验证集准确率评估及保存最优权重）。
- `search.py`：超参数查找模块（实现网格搜索以寻找最优的超参数组合）。
- `eval.py`：测试评估模块（加载最优权重并在测试集上计算 Accuracy，绘制混淆矩阵和错例分析，以及隐藏层权重的可视化）。
- `main.py`：主入口脚本，将上述流程串联起来，一键运行完整实验。

## 如何运行
1. **准备数据集**：
   确保 `EuroSAT_RGB` 文件夹与代码处于同一目录，或在 `main.py` 中修改 `data_dir` 为数据集路径。

2. **一键运行完整实验 (训练 + 搜索 + 测试评估)**：
   运行以下命令：
   ```bash
   python main.py
   ```
   该脚本将依次执行：
   - 数据加载和拆分（Train: 70%, Val: 15%, Test: 15%）
   - 网格搜索寻找最优超参数
   - 用最优超参数训练最终模型
   - 在测试集上评估模型并输出指标
   - 所有的输出（图片、权重文件）将保存在 `output/` 文件夹中。

3. **输出产物**：
   运行结束后，`output/` 文件夹将包含：
   - `grid_search_results.txt`: 超参数搜索结果记录
   - `best_weights.pkl`: 最优模型权重
   - `loss_acc_curves.png`: 训练/验证 Loss 与 Accuracy 曲线
   - `confusion_matrix.png`: 测试集混淆矩阵可视化
   - `weights_visualization.png`: 第一层隐藏层权重可视化
   - `error_analysis.png`: 测试集错例分析图片
