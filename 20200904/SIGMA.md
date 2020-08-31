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

    [3]“Nvdla deep learning accelerator,” in http://nvdla.org, 2018.
    [10] Y.-H. Chen et al., “Eyeriss: An energy-efficient reconfigurable accelerator for deep convolutional neural networks,” in ISSCC,2016.
    [21]K. Hedge et al., “Ucnn: Exploiting computational reuse in deep neural networks via weight repetition,” in ISCA, 2018.
    [27] H. Kwon, et al., “Maeri: Enabling flexible dataflow mapping over dnn accelerators via reconfigurable interconnects,” in ASPLOS, 2018.
    [28] H. Kwon et al., “Understanding reuse, performance, and hardware cost of dnn dataflow: A data-centric approach,” in MICRO, 2019.
    [33] A. Parashar et al., “Scnn: An accelerator for compressed-sparse convolutional neural networks,” in ISCA, 2017.

>The DL training process is extremely compute intensive. This is elucidated in an OpenAI study [6], which shows that the compute requirements for training has grown 300,000 times from AlexNet (2012) to AlphaGo Zero (2018). GPUs are currently the most popular acceleration platform in use for training; and recent research focus on parallelizing large DNN models over multiple GPU nodes. Some companies like Google and Microsoft have built their own specialized training platforms such as the cloud TPU [4] and Brainwave FPGAs [14] respectively.

DL训练过程非常耗费计算资源。 OpenAI研究[6]中阐明了这一点，该研究表明，培训的计算需求已从AlexNet（2012）到AlphaGo Zero（2018）增长了300,000倍。 GPU是目前最流行的用于训练的加速平台。 最近的研究重点是在多个GPU节点上并行化大型DNN模型。 像Google和Microsoft这样的公司已经建立了自己的专业培训平台，例如Cloud TPU [4]和Brainwave FPGA [14]。

    [4] “Cloud tpu,” in https://cloud.google.com/tpu, 2019.
    [6] D. Amodei and D. Hernandez, “https://blog.openai.com/aiand-compute/,” 2018.
    [14] J. Fowers et al., “A configurable cloud-scale dnn processor for real-time ai,” in ISCA, 2018.
>is extremely compute-intensive 极度的消耗计算资源

>The core compute component of DL training (and inference) is the GEMM operation [1]. Fig. 1a shows the GEMM dimensions (M, N, K) and operation; while Fig. 1b shows example dimensions found in modern DL workloads. During forward pass, DNNs with fully-connected (FC) layers and multilayer perceptron (MLPs) naturally map to GEMM operations, with MK representing inputs and KN representing weights. For Convolutional Neural Networks (CNNs), GPUs remap the conv operation into a GEMM via the Im2Col operation [18] or other efficient ordering operations. During the backward pass, two GEMM operations arise: MN x (KN)T and (MK)T x MN for computing the error gradient w.r.t inputs and weights respectively.
![20200830173146](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200830173146.png)

DL训练（和推理）的核心计算组件是GEMM操作[1]。 图1a显示了GEMM尺寸（M，N，K）和操作。 图1b显示了在现代DL工作负载中发现的示例维度。 在前向传递过程中，具有全连接层（FC）和多层感知器（MLP）的DNN自然映射到GEMM操作，其中MK代表输入，KN代表权重。 对于卷积神经网络（CNN），GPU通过Im2Col操作[18]或其他有效的排序操作将conv操作重新映射为GEMM。 在反向传播期间，出现两个GEMM操作：MN x（KN）T和（MK）T x MN，分别用于计算误差梯度w.r.t输入和权重。

    [1] “https://petewarden.com/2015/04/20/why-gemm-is-at-theheart-of-deep-learning/,” 2015.
    [18] S. Hadjis et al., “Caffe con troll: Shallow ideas to speed up deep learning,” in DanaC, 2015.

>GEMMs comprises around 70% of the total compute cycles during training (as we show in Sec. II); and therefore is a primary target for hardware acceleration. State-of-the-art training accelerators use systolic arrays as the compute fabric for accelerating GEMMs - with sizes ranging from tiny 4x4 engines within each Streaming Multiprocessor in the NVIDIA Volta GPU [31] to a large monolithic 128x128 engine in the Google Cloud TPU [4]. Systolic arrays are built by interconnecting MAC units to a tightly-coupled 2D grid. They are efficient for running GEMMs by enabling reuse of the (M,K) or (K,N) matrix over the rows and columns of the array, reducing data accesses and instruction overheads.

GEMM约占训练期间总计算周期的70％（如我们在第二节中所示）； 因此是硬件加速的主要目标。 最先进的训练加速器使用脉动收缩阵列作为加速GEMM的计算结构-大小从NVIDIA Volta GPU中每个流式多处理器中的微型4x4算子[31]到Google Cloud TPU中的大型单块128x128引擎不等 [4]。 通过将MAC单元互连到紧密耦合的2D网格来构建脉动阵列。 通过在阵列的行和列上重用（M，K）或（K，N）矩阵，它们对于运行GEMM非常有效，从而减少了数据访问和指令开销。

    [31] Nvidia, “Nvidia tesla v100 gpu architecture,” in Volta Architecture Whitepaper, 2017.
    [4] “Cloud tpu,” in https://cloud.google.com/tpu, 2019.

>As DNN models evolve at a rapid rate, it is imperative to design the underlying acceleration substrate to remain efficient for future models and training techniques. While GEMMs continue to remain the favorite abstraction to unroll tensors to during the forward and backward passes, architects need to be cognizant of three trends:

>• Many GEMMs have irregular (or non-square) dimensions arising from minibatches and weight factorization [34].

>• GEMMs exhibit varying degrees of weight sparsity from pruning, and activation sparsity from non-linearities (like ReLU, pooling, and dropout). The number of nonzeros varies throughout training iterations (from 10% to 90%) [48].

>• DNN models are being developed at a rapid rate as AI becomes evermore pervasive, making it impractical to pick a specific set of matrix dimensions or sparsity ratios to design accelerators for

随着DNN模型的快速发展，必须设计潜在的加速度基础以保持对未来模型和训练技术的有效性。 尽管在向前和向后传递期间，GEMM仍然是张量展开最喜欢的抽象表示，但是架构师需要意识到以下三种趋势：

•许多GEMM由于小批处理和权重分解而具有不规则（或非正方形）尺寸[34]。

•GEMM由于修剪而表现出不同程度的重量稀疏性，而由于非线性（如ReLU，池化和dropout）而引起的活化稀疏性不同。 在训练迭代过程中，非零的数量会有所不同（从10％到90％）[48]。

•随着AI变得越来越普遍，DNN模型正在快速发展，这使得选择一组特定的矩阵尺寸或稀疏率来设计加速器以实现不现实。

    [34] J. Park et al., “Deep learning inference in facebook data centers: Characterization, performance optimizations and hardware implications,” CoRR, vol. abs/1811.09886, 2018.
    [48] M. H. Zhu and S. Gupta, “To prune, or not to prune:exploring the efficacy of pruning for model compression,”arXiv:1710.01878v2 [stat.ML], 2017.


>Based on our observations, we recommend three key requirements for future GEMM engines.

>• Flexibility: GEMM engines should be able to efficiently run matrices of arbitrary dimensions.

>• Sparsity Support: GEMM engines need support for unstructured sparsity in both weights and activations in order to fully utilize hardware resources.

>• Scalability: GEMM engines need to scale efficiently for integration into different kinds of accelerators. For example, tiny tensor cores in CPUs/ GPUs to large cores in a future TPU.


>根据我们的观察，我们建议未来GEMM算子的三个关键要求。

>•灵活性：GEMM算子应该能够有效地运行任意尺寸的矩阵。

>•稀疏性支持：GEMM算子需要支持权重和激活方面的非结构性稀疏性，以便充分利用硬件资源。

>•可拓展性：GEMM算子需要有效地进行拓展以集成到各种加速器中。 例如，在将来的TPU中，CPU / GPU中的微小张量内核到大型内核。


>Unfortunately, state-of-the-art GPUs and TPUs fare poorly on some of the requirements, as we discuss later in Sec. III. GPUs [31] provide some flexibility in terms of tiling irregular GEMMs into 4x4 chunks and mapping over tensor cores, but add complexity for scheduling and accumulation across SMs. TPU, being a large inflexible 128x128 array, can lead to compute under-utilization if the GEMM dimensions do not align with the dimensions of the physical array. GPUs cannot exploit both input and weight sparsity. Even when only one type of sparsity is present, current CUDA libraries require the datatype to be FP32 and the sparse data to be structured. TPU do not natively support sparsity since its rigid internal connectivity and per-cycle systolic dataflow prevent skipping multiplications with at least one operand that is zero. And finally, while systolic arrays scale well due to a regular 2D structure (as is evident from 4x4 versions in GPUs to a 128x128 version in the TPU), larger arrays take proportionally longer to load data and collect final outputs.

不幸的是，最先进的GPU和TPU不能满足某些要求，正如我们稍后将在Sec3中讨论的那样. GPU [31]在将​​不规则的GEMM分成4x4块并在张量核心上进行映射方面提供了一定的灵活性，但是增加了跨SM的调度和累积的复杂性。如果GEMM尺寸与物理阵列的尺寸不一致，则TPU是一个较大的非柔性128x128阵列，可能会导致计算利用率不足。 GPU无法同时利用输入和权重稀疏性。即使仅存在一种稀疏类型，当前的CUDA库也要求数据类型为FP32并构造稀疏数据。 TPU本身不支持稀疏性，因为其严格的内部连接性和每个周期的收缩数据流可防止与至少一个为零的操作数跳过乘法。最后，尽管脉动式阵列由于采用规则的2D结构而可以很好地扩展（从GPU中的4x4版本到TPU中的128x128版本可见），但是较大的阵列按比例需要更长的时间来加载数据和收集最终输出。

    [31] Nvidia, “Nvidia tesla v100 gpu architecture,” in Volta Architecture Whitepaper, 2017.

>In this work, we demonstrate the microarchitecture of a flexible and scalable GEMM accelerator named SIGMA that can handle (a) arbitrary amounts of sparsity, (b) arbitrary irregularity in GEMM dimensions, while guaranteeing close to full compute utilization. SIGMA’s key novelty is a highly Flexible Dot Product Engine (Flex-DPE), that can map GEMMs of arbitrary shapes and sparsity distributions via a rich interconnect fabric. Further, Flex-DPE uses tree-based topologies - enabling data loading and collection times of O(1) and O(log2N) respectively, instead of O(√N) for an equivalent sized square systolic array. The full SIGMA engine connects multiple Flex-DPEs together via a simple global network-on-chip (NoC). The NoC allocate a cluster of FlexDPEs for one GEMM. Each cluster is called a Flexible Dot Product Unit (Flex-DPU). SIGMA can thus morph into a large Flex-DPU running one GEMM or into multiple small variable-sized Flex-DPUs running different GEMMs.

在这项工作中，我们演示了一个名为SIGMA的灵活，可扩展的GEMM加速器的微体系结构，该加速器可以处理（a）任意数量的稀疏性，（b）GEMM尺寸中的任意不规则性，同时保证接近完全的计算利用率。 SIGMA的主要创新之处是高度灵活的点积算子（Flex-DPE），它可以通过丰富的互连结构映射任意形状和稀疏分布的GEMM。 此外，Flex-DPE使用基于树的拓扑-分别启用O（1）和O（log2N）的数据加载和收集时间，而不是等效大小的方形脉动阵列的O（√N）。 完整的SIGMA算子通过一个简单的全局片上网络（NoC）将多个Flex-DPE连接在一起。 NoC为一个GEMM分配了FlexDPE集群。 每个群集称为柔性点产品单元（Flex-DPU）。 因此，SIGMA可以演变为运行一个GEMM的大型Flex-DPU或运行不同GEMM的多个小型可变大小的Flex-DPU。

>Our key contributions are the following:

>>1) Analysis of modern DL training workloads to make the case for accelerating sparse, irregular GEMMs.

>2) A novel accelerator named SIGMA for handling irregular and unstructured sparse GEMM operations.

>3) A novel reduction network, named Forwarding Adder Network (FAN), for efficient partial sum accumulation.

>4) Layout implementation of SIGMA for scalability and backend analysis.


本文的主要贡献如下：

1）分析现代DL训练工作量，以为加速稀疏，不规则的GEMM提供依据。

2）一种名为SIGMA的新型加速器，用于处理不规则和非结构化的稀疏GEMM操作。

3）一种新颖的归约网络，称为转发加法器网络（FAN），用于有效地进行部分和累加。

4）SIGMA的布局实现，以实现可拓展性和后端分析。

>The rest of the paper is organized as follows: Sec. II discusses modern training workloads and their GEMM characteristics. Sec. III dissects state-of-the-art deep learning accelerators and design considerations. Sec. IV proposes the SIGMA microarchitecture, and Sec. V describes the physical implementation and hardware costs. Sec. VI evaluates the performance of SIGMA against the state-of-the-art. Sec. VII discusses the related works, and Sec. VIII concludes.

本文的其余部分安排如下： 第二章讨论了现代培训工作量及其GEMM特性。 第三部分剖析了最先进的深度学习加速器和设计注意事项。 第四部分提出了SIGMA微体系结构。第五部分描述了物理实现和硬件成本。 第六部分根据最新技术评估SIGMA的性能。 第七节讨论了相关作品。 第八章总结。

> * DL训练特性

>In this section, we analyze GEMM kernel shapes and sparsity levels from modern deep learning applications. 

>Target Workloads. For the kernel characterization exercise, we consider three workloads: Transformer [42], Google Neural Machine Translation (GNMT) [45], and Neural Collaborative Filtering (NCF) [20]. We also leverage ”Baidu DeepBench” [2], which identifies key GEMM kernels encountered across various CNNs/ RNNs/ LSTMs. For Transformer, we use a 324 Million parameter model [43] with the LM1B (billion word corpus) dataset. For GNMT, we evaluate the state of art 8-layer GNMT model with WMTGerman-English dataset.

在本部分中，我们将从现代深度学习应用程序分析GEMM内核形状和稀疏性特性。

目标工作量。 针对于内核的表征方法，我们考虑了三种工作负载：Transformer[42]，谷歌神经机器翻译（GNMT）[45]和神经协作过滤（NCF）[20]。 我们还利用“百度DeepBench” [2]，它确定了跨各种CNN / RNN / LSTM遇到的关键GEMM内核。 对于Transformer，我们将324百万参数模型[43]与LM1B（十亿字语料库）数据集一起使用。 对于GNMT，我们使用WMTGerman-English数据集评估了最新的8层GNMT模型。

    [42] A. Vaswani, et al., “Attention is all you need,” CoRR, vol. abs/1706.03762, 2017.
    [45] Y. Wu et al., “Google’s neural machine translation system: Bridging the gap between human and machine translation,” 2016.
    [2] “Baidu-deep bench,” 2016.
    [43] A. Vaswani et al., “Tensor2tensor for neural machine translation,”CoRR, vol. abs/1803.07416, 2018.

>Time-Breakdown of Compute Primitives. Figure 2 shows the time break-up of different operations when training GNMT and Transformer on a NVIDIA V100 GPU [31]. We observe that approximately 70% of time is spent on matrix multiplications (MatMul) operations or operations that can cast as MatMuls. Thus, MatMul is a key compute primitive to accelerate in hardware to speed-up training.

计算基元的时间分解。 图2显示了在NVIDIA V100 GPU上训练GNMT和Transformer时，不同操作的时间分解[31]。 我们观察到大约70％的时间花费在矩阵乘法（MatMul）运算或可以转换为MatMuls的运算上。 因此，MatMul是关键的计算原语，可以在硬件上加速以加快训练速度。

![20200831095925](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200831095925.png)

    [31] Nvidia, “Nvidia tesla v100 gpu architecture,” in Volta Architecture Whitepaper, 2017.

>GEMM shapes. Transformer, GNMT, NCF and DeepBench [2] have matrices of different sizes and shapes as shown in Fig. 1b. Training is performed in different batch sizes, which lead to different input matrix dimensions. The observed shapes of the operand matrices vary from tall-skinny (rows dominate over columns) to fat-short (columns dominate over rows) - this is due to low minibatch sizes. Thus, GEMM accelerators need scalability and flexibility to handle large and irregular GEMM sizes efficiently.

GEMM形状。 Transformer，GNMT，NCF和DeepBench [2]具有不同大小和形状的矩阵，如图1b所示。 训练的时候batch_size的大小不同，这导致了不同的输入矩阵尺寸。 观察到的操作数矩阵的形状从高瘦（行占列为主）到胖短（列占行为主）-这是由于小批量的大小所致。 因此，GEMM加速器需要可拓展性和灵活性，以有效处理较大和不规则的GEMM尺寸。

    [2] “Baidu-deep bench,” 2016.

>Sparsity within GEMMs. As the objective of this work is not focused on algorithm techniques to generate sparse models, we leverage a pruning approach similar to Zhu et al. [48] via a slow sparsification technique that increases the sparsity level of weights from zero to a final sparsity level in a fixed set of pruning steps.

GEMM中的稀疏性。 由于这项工作的目标不是集中在生成稀疏模型的算法技术上，因此我们采用了与Zhu等类似的修剪方法[48]通过慢速稀疏化技术，在固定的一组修剪步骤中将权重的稀疏性级别从零增加到最终稀疏性级别。

    [48] M. H. Zhu and S. Gupta, “To prune, or not to prune: exploring the efficacy of pruning for model compression,” arXiv:1710.01878v2 [stat.ML], 2017.

>For GNMT [45] with ∼210M parameters, we achieve close to state-of-the-art accuracy with 90% weight sparsity (resulting in ∼22M parameters), similar to results outlined in [48]. The pruning is applied to embedding, decoder projection layer and all LSTM layers in both the encoder and decoder. Workloads like transformer and ResNet-50 also exhibits good accuracy with around 80% and 70% weight sparsity respectively [15]. Activation sparsity in DNN models comes from ReLU and dropout layers.

对于具有〜210M参数的GNMT [45]，我们获得了90％的重量稀疏度（达到〜22M参数），接近了最先进的精度，类似于[48]中概述的结果。 修剪应用于编码器和解码器中的嵌入，解码器投影层和所有LSTM层。 诸如变压器和ResNet-50之类的工作负载也具有良好的精度，其稀疏度分别约为80％和70％[15]。 DNN模型中的激活稀疏性来自ReLU和dropout层。

    [48] M. H. Zhu and S. Gupta, “To prune, or not to prune: exploring the efficacy of pruning for model compression,” arXiv:1710.01878v2 [stat.ML], 2017.
    [45] Y. Wu et al., “Google’s neural machine translation system: Bridging the gap between human and machine translation,” 2016.
    [15] T. Gale et al., “The state of sparsity in deep neural networks,” arXiv:1902.09574v1 [cs.LG], 2019.


>Improper handling of sparse matrices wastes compute resources and causes unnecessary but expensive movement of zeros across the memory hierarchy. As matrices are getting bigger and sparser, the need for sparsity support becomes more important. Thus, GEMM accelerators need support to handle both weight and activation sparsity efficiently.

稀疏矩阵的处理不当会浪费计算资源，并在内存层次结构中导致零的不必要但昂贵的移动。 随着矩阵越来越大和越来越稀疏，稀疏支持的需求变得越来越重要。 因此，GEMM加速器需要支持以有效处理权重和激活稀疏性。


>* 分析GPUs和TPUs效率低下的原因

>In this section, we demonstrate the inefficiencies with current GEMM accelerators, and discuss the design choices Figure 3: GPU performance evaluation on different GEMMs. that eventually lead to our proposed design.

在本节中，我们演示了当前GEMM加速器的低效率，并讨论了设计选择。图3：不同GEMM上的GPU性能评估。 最终我们提出了设计方案。



![20200831103210](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200831103210.png)

>A. Irregular and Sparse GEMMs on GPU
>We measured the compute efficiency on V100 GPUs with and without sparsity for various GEMM dimensions. In Fig. 3a, we run some of the deep learning MatMul kernels (dense irregular without any sparsity) for workloads described in Sec. II on a single card V100 GPU and measure the efficiency with FP32 and FP16 data type. FP16 data type can take advantage of the systolic arrays (“tensor cores”) in V100 for GEMM computation. While FP16 uses the tensor cores to boost the efficiency compared to the FP32 version, they still operate at a fraction of the peak efficiency due to irregularity in kernel dimensions; whereas a dense regular GEMM (2k, 2k, 2k) with FP16 tensor cores provide up to 76% efficiency.

我们测量了具有和不具有稀疏性的各种GEMM尺寸的V100 GPU的计算效率。 在图3a中，运行了一些第二章描述的工作负载深度学习模型MatMul内核（密集的不规则而没有任何稀疏性）。 在单卡V100 GPU上，并使用FP32和FP16数据类型测量效率。 FP16数据类型可以利用V100中的脉动阵列（“张量核心”）进行GEMM计算。 与FP32版本相比，虽然FP16使用张量内核来提高效率，但由于内核尺寸的不规则性，它们仍只能以峰值效率的一小部分工作。 而带有FP16张量芯的密集常规GEMM（2k，2k，2k）可提供高达76％的效率。

>We then introduce sparsity to the above MatMul kernels and use NVIDIA cuSPARSE [5] libraries, which support sparse matrices computation. cuSPARSE libraries API currently support only one of the matrices to be sparse with only FP32 data type. In this experiment, we induce random sparsity of 50% and 80% to one of the matrices while keeping the other matrix dense. From Fig. 3b, we observe on average 4x reduction in efficiency compared to the equivalent dense FP32 matrix computation by adding sparsity. We expect the efficiency to decrease further when both matrices are sparse. Current GPU systems cannot efficiently map sparse GEMM computation onto their compute engine when there is no structure in the sparsity, and thus we need to fundamentally re-architect how we design a system that can take advantage of sparse computation to achieve high efficiency for deep learning workloads.

然后，我们为上述MatMul内核引入稀疏性，并使用NVIDIA cuSPARSE [5]库，该库支持稀疏矩阵计算。 cuSPARSE库API当前仅支持使用FP32数据类型稀疏的一种矩阵。 在此实验中，我们对其中一个矩阵诱导了50％和80％的随机稀疏性，同时使另一个矩阵保持密集。 从图3b中，通过添加稀疏度，与等效的密集FP32矩阵计算相比，我们观察到效率平均降低4倍。 当两个矩阵都稀疏时，我们期望效率会进一步降低。 当稀疏性中没有任何结构时，当前的GPU系统无法将稀疏的GEMM计算有效地映射到其计算引擎上，因此，我们需要从根本上重新设计我们如何设计一种系统，该系统可以利用稀疏计算来实现深度学习的高效率工作量。

    [5] “https://docs.nvidia.com/cuda/cusparse/index.html,” 2019.

>B. Irregular and Sparse GEMMs on TPU 

>Google’s TPUs are a poster-child for large GEMMs due to their 128×128 systolic array. However, across a suite of GEMMs from modern DL workloads, we observe that it is common to have less than 50% of the array utilized when running irregular matrices, as we show later in Sec. VI. In addition, systolic arrays cannot inherently address sparsity. The reasons for these inefficiencies are discussed next.

Google的TPU由于其128×128的收缩阵列而成为大型GEMM的衍生品。 但是，在来自现代DL工作负载的GEMM套件中，我们观察到运行不规则矩阵时阵列的利用率不足50％是很常见的，正如我们稍后在Sec VI中所示。 另外，脉动阵列不能固有地解决稀疏性。 接下来讨论这些效率低下的原因。

