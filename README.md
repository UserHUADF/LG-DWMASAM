# LG-DWMASAM 边重要性辨识实验代码

这是根据 `代码-边重要性辨识.docx` 中的 LG-DWMASAM 流程整理的可运行实验项目。论文里的实证数据部分为空，所以项目同时提供：

- 通用 CSV 输入接口：`time,source,target,weight`
- 一份可复现实验样例：`data/sample_temporal_edges.csv`
- 三种线图权重映射：`product`、`harmonic`、`geometric`
- 边重要性排序、单调性、删边连通效用下降率、链路级联传播检验

## 运行

```bash
python3 run_experiment.py --input data/sample_temporal_edges.csv --operator product
```

输出文件默认写入 `outputs/`：

- `summary.json`：整体实验摘要、指标、Top 边
- `edge_scores.csv`：每条边的聚合得分、分时间层得分和排名

## 输入格式

```csv
time,source,target,weight
1,A,B,0.70
1,B,C,0.80
2,A,B,0.60
```

同一条有向边可以跨时间层出现；未出现时视为该层不活跃。重复记录会自动累加权重。

## 参数说明

- `--operator`：线图连边权重映射算子
- `--alpha`：公式 (2-8) 中入流和出流的调节参数，范围 `(0, 1)`
- `--kshell-q`：公式 (2-5) 的 `Q^L(t)`。原文未给出估计规则，默认按 1.0 执行
- `--top-ratio`：删边检验中移除排名前多少比例的边
- `--seed-ratio`：级联传播检验中选取排名前多少比例的边作为种子
- `--threshold`：线性阈值级联模型的激活阈值
- `--no-normalize`：关闭 TCM、SK、C 融合前的最大值归一化

## 自检

```bash
python3 -m unittest discover -s tests
```
