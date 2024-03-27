# 简单记录一下学习CUDA过程中的一些探索性的问题

## 1.cuda vec type

测试比较了load float, float2, float4这三种类型的性能，测试结果如下：

[实验记录](./Tests/VectorType/readme.md)

## 2. sgemm 单精度矩阵通用乘法

参考[深入浅出](https://zhuanlan.zhihu.com/p/435908830) 和 [链接](https://linn-ylz.com/tags/Computer-Science/) 进行了学习。

[实验记录](./SGEMM/readme.md)