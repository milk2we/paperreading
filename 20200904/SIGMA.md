## SIGMA: A Sparse and Irregular GEMM Accelerator with Flexible Interconnects for DNN Training
    SIGMA：具有用于DNN训练的灵活互连的稀疏和不规则GEMM加速器

> * 摘要
> 
> Abstract—The advent of Deep Learning (DL) has radically transformed the computing industry across the entire spectrum from algorithms to circuits. As myriad application domains embrace DL, it has become synonymous with a genre of workloads across vision, speech, language, recommendations, robotics, and games. The key compute kernel within most DL workloads is general matrix-matrix multiplications (GEMMs), which appears frequently during both the forward pass (inference and training) and backward pass (training). GEMMs are a natural choice for hardware acceleration to speed up training, and have led to 2D systolic architectures like NVIDIA tensor cores and Google Tensor Processing Unit (TPU).
> 
>    Unfortunately, emerging GEMMs in DL are highly irregular and sparse, which lead to poor data mappings on systolic architectures. This paper proposes SIGMA, a flexible and scalable architecture that offers high utilization of all its processing elements (PEs) regardless of kernel shape and sparsity. Within SIGMA includes a novel reduction tree microarchitecture named Forwarding Adder Network (FAN). SIGMA performs 5.7× better than systolic array architectures for irregular sparse matrices, and roughly 3× better than state-of-the-art sparse accelerators. We demonstrate an instance of SIGMA operating at 10.8 TFLOPS efficiency across arbitrary levels of sparsity, with a 65.10 mm2 and 22.33 W footprint on a 28 nm process.

    深度学习的出现已经从算法到电路彻底的改变了整个计算业。无数的应用开始涌入深度学习，深度学习已经成为了视觉等的代名词。在DL中的关键计算核心就就是通用矩阵-矩阵计算（GEMM）,无论是前向计算还是反向计算GEMM出现的次数最为频繁。对于利用硬件加速器去加速训练，GEMM是一个很自然的选择。并且产生了2D 收缩架构例如：NVIDIA Tensor Core 和Google Tensor Processing(TPU).
        Tips: 2D systolic architectures 2维收缩架构是一个新名词，需要查阅

    令人遗憾的是，DL中的GEMM非常不规则且稀疏，这导致收缩体系结构上的数据映射不佳。这篇文章提出了SIGMA，一种灵活可拓展的架构，这种架构无论kernel形状和稀疏度其处理单元（PEs）均具有极高的利用率，。SIGMA包含了一种还原树未处理架构，命名为转发加法器网络（FANS）。SIGMA对比收缩阵列架构由5.7x加速针对于无规则的稀疏矩阵，并且可以达到3x加速效果对比SOTA稀疏加速器。作者演示了一个SIGMA实例，该实例在任意级别的稀疏度下均以10.8 TFLOPS的效率工作，在28 nm工艺上具有65.10 mm2和22.33 W的占位面积。

>* Introducation

> Deep learning (DL) has emerged as the premier algorithmic technique for analyzing data across multiple domains, especially in visual understanding, speech perception, and automated reasoning. The application of DL consists of two steps; namely training and inference. During training, a Deep Neural Network (DNN) model uses a loss function and optimization process to minimize model error on the training dataset. During inference, the trained DNN model is used to
classify data points.

    DL已经成为数据处理领域的的第一算法技术，尤其在虚拟理解等领域。DL的应用通常分为两部分，分别是训练和推理。在训练过程中，DNN使用损失函数和优化措施来最小化模型的误差在训练数据集上面。在推理阶段，训练好的DNN模型用来分类数据点。
    Tips:常见的损失函数：交叉损失熵等 优化措施：SGD\ADAM等 

>Given latency sensitivity and energy-efficiency demands for DNN inference, a suite of custom accelerators have been proposed [3], [10], [21], [27], [33] to run trained models efficiently by capturing various forms of data reuse [10], [28]. However, the right acceleration platform for training current and future models, is still an open question, which is the focus of this work.

    给定DNN推理的等待时间敏感性和能效需求，已提出了一套定制加速器[3]，[10]，[21]，[27]，[33]，以通过捕获各种形式的数据重用来有效地运行经过训练的模型[10]，[28]。 但是，用于训练当前和将来模型的正确加速平台仍然是一个悬而未决的问题，这是这项工作的重点。
