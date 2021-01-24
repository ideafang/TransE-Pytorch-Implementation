# TransE-Pytorch-Implementation
Simple implementation of transe framework based on pytorch

This repository draws on [LYuhang/Trans-Implementation](https://github.com/LYuhang/Trans-Implementation)

build data preprocessing, dataset construction, negative samples generation, model training, MR verification, model preservation functions.

基于pytorch的TransE模型轻量实现方法

本仓库的方法借鉴了[LYuhang/Trans-Implementation](https://github.com/LYuhang/Trans-Implementation) 中的TransE模型构建，训练过程。对整个流程进行大量精简和重构，优化了数据输入和处理过程，为自建数据集的构建提供基类和基本方法。对入门更为友好。

## 环境配置

- pytorch
- numpy
- pickle

## 运行

代码默认使用`FB15k-237`数据集，其他数据集可以直接放到dataset文件夹中，然后在train.py文件中

