"""用合成数据把拟合跑通。换成实测时读测量表，不要改预测式。

实测步骤：
1. 用 ModelSpec 写下这个模型的块、单卡容量和元素字节数。
2. write_template 生成表头。每一行是一次前向：负载、方案、端到端时间；
   对照运行另填各块分段时间。
3. load_samples 读入后，holdout_by_load 按 batch 和序列长度整组留出测试集。
4. fit 只在训练集上估计参数。evaluate 只比较测试集的预测时间和实测时间。
"""

from timepred.demo import main

if __name__ == "__main__":
    main()
