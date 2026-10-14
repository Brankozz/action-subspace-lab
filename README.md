# 动作子空间分析台

一个在浏览器里运行的轨迹分析工具：检验一组轨迹是否等于“共用部分 + 低秩偏差”，查看它们在子空间里的分布，并从子空间里采样新轨迹。

- 支持上传 parquet、csv、json、npy 文件，所有计算都在浏览器本地完成，数据不会上传。
- `index.html` 是页面，`analysis-core.js` 是数值计算部分（PCA、留出重建、相位打乱对照、动态时间规整）。

## 示例数据

`samples/` 下的文件由 LIBERO 演示数据的动作和末端位置量化压缩而来，来源是
[yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)（CC BY 4.0），
原始数据来自 [LIBERO](https://libero-project.github.io/)。
