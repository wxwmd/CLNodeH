> 本仓库存放节点分类方法CLNode和CLNodeH的核心代码

### 1. 文件结构
- CLNode文件夹存放着同质图节点分类方法CLNode的代码
- CLNodeH文件夹存放着异质图节点分类方法CLNodeH的代码


### 2. 运行方法
1. 安装依赖
```shell
pip install -r requirements.txt
```

2. 同质图节点分类方法
```shell
cd CLNode
python clnode.py #数据集：cora 基线模型：GCN
```

3. 异质图节点分类方法
```shell
cd CLNodeH
python clnodeh.py #数据集：DBLP 基线模型：HAN
```