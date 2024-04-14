> 本仓库存放节点分类方法CLNode和CLNodeH的核心代码

### 1. 文件结构
- CLNode文件夹存放着同质图节点分类方法CLNode的代码

```shell
文件结构树
CLNode
|____checkpoints #此文件夹存放着训练过程中产生的模型参数
|____data #存放数据集
|____clnode.py #CLNode方法的入口
|____early_stop.py #early stopping的代码
|____GCN.py #存放GCN模型的架构代码
|____util.py #工具函数，包括增加数据噪声、多视角节点质量评估等核心代码
```

- CLNodeH文件夹存放着异质图节点分类方法CLNodeH的代码
```shell
文件结构树
CLNode
|____checkpoints #此文件夹存放着训练过程中产生的模型参数
|____clnode.py #CLNodeH方法的入口
|____early_stop.py #early stopping的代码
|____HAN.py #存放HAN模型的架构代码
|____hgb.py #异质图数据集的预处理代码
|____util.py #工具函数，包括增加数据噪声、多语义质量评估等
```

### 2. 运行方法
1. 配置python环境
```shell
$ conda create -n pyg python==3.10
$ conda activate pyg
$ pip install -r requirements.txt
```

2. 同质图节点分类方法
```shell
$ cd CLNode
$ python clnode.py --noise_percent=20 --scheduler=geom --seed=1
```
`clnode.py`文件使用CLNode方法优化了GCN模型在含噪Cora数据集上的性能，上面命令行三个参数分别为：
- noise_percent: 要向Cora数据集中增加的标签噪声的比例
- scheduler：训练调度器中的调度函数
- seed：随机种子，用于复现实验结果

3. 异质图节点分类方法
```shell
$ cd CLNodeH
$ python clnodeh.py --noise_percent=30 --scheduler=geom --seed=1
```
`clnodeh.py`文件使用CLNodeH方法优化了HAN模型在含噪DBLP数据集上的性能，参数含义与`clnode.py`中保持一致。
