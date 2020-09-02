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

C. GEMMs on Systolic Arrays vs. SIGMA 

>Systolic arrays face under-utilization in many different scenarios due to two inherent features: a rigid shape, and a simple but rigid interconnect. In Fig. 4, we contrast a systolic array against an abstract view of SIGMA’s Flex DPE (which will be presented in detail later in Sec. IV-A). Fig. 4a shows three example GEMM operations: (i) dense regular, (ii) dense irregular and (iii) sparse irregular matrices. The shaded boxes in the figure highlight quantitative metrics such as utilization, runtime, multicast behavior, and SRAM reads/writes for each example. For the purposes of this example, it is sufficient to assume that SIGMA’s Flex-DPE has two specialized networks between the PEs and the SRAMs on the sides of the array (not shown in the figure) - a distribution network and a reduction network. The specific mplementation of these networks is discussed later in Sec. IV.

由于两个固有特征：刚性形状和简单但刚性的互连，收缩压阵列在许多不同情况下都面临利用率不足的问题。 在图4中，我们将脉动阵列与SIGMA Flex DPE的抽象视图进行了对比（稍后将在第四节-A节中详细介绍）。 图4a显示了三个示例GEMM操作：（i）密集的规则，（ii）密集的不规则和（iii）稀疏的不规则矩阵。 图中的阴影框突出显示了每个指标的定量指标，例如利用率，运行时间，多播行为以及SRAM读/写。 就本示例而言，假设SIGMA的Flex-DPE在PE和阵列侧面的SRAM之间（图中未显示）有两个专用网络-分配网络和简化网络。 这些网络的具体实现将在第四章中讨论。

![20200831173110](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200831173110.png)

>Dense Regular Matrices. Fig. 4b shows how the dense regular matrices are mapped. In the example, we use a KN matrix stationary, and MK matrix streaming dataflow (Nsta, M-str). An alternate term for this dataflow is weightstationary, and is used by the Google TPU [23], [37]. Partial sums are generated at each cross-point and accumulated over the columns. The systolic array and Flex-DPE designs are able to fully utilize its PEs by mapping all of KN matrix onto its PEs. They key differences between the two are that (i) a systolic array sends the streaming matrix in a store and forward manner, while SIGMA multicasts the data to the corresponding PEs in one cycle, and (ii) the systolic array uses a linear reduction while SIGMA uses a tree-based reduction, as we describe later in Sec. IV.

密集的常规矩阵。 图4b显示了如何映射密集的规则矩阵。 在示例中，我们使用固定的KN矩阵和MK矩阵流数据流（Nsta，M-str）。 此数据流的另一个术语是weightstationary，由Google TPU [23]，[37]使用。 在每个交叉点生成部分和，并将其累加到各列上。 通过将所有KN矩阵映射到其PE上，脉动阵列和Flex-DPE设计能够充分利用其PE。 它们之间的主要区别在于（i）脉动阵列以存储和转发方式发送流矩阵，而SIGMA在一个周期内将数据多播到相应的PE，并且（ii）脉动阵列使用线性缩减，而 SIGMA使用基于树的约简，正如我们稍后将在SecIV中描述的那样。

![20200831173923](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200831173923.png)

>Dense Irregular Matrices. Fig. 4c shows how dense irregular matrices are mapped. A systolic array design suffers from under-utilization due to its rigid structure. Despite the fact that there are 16 elements in the dense irregular KN matrix and 16 PEs in the systolic array, only half of the matrix can be mapped at a time. This is due to the rigid shape of the systolic array. All partial sums are accumulated down a column via forwarding; mapping the other half of the dense irregular N-matrix onto the systolic array at the same time will lead to incorrect functionality, since the accumulated output (a.A+b.I) should not get added to (a.E + b.M). The second half of the N-matrix will therefore have to be loaded once the first half of the matrix is computed, more than doubling the computation time. In contrast, the Flex-DPE is able to map all of the dense irregular stationary elementsat one go, utilizing all PEs completely. This is enabled by having a flexible reduction network that can accumulate both sets of outputs (a.A+b.I) and (a.E + b.M) separately and concurrently, as we describe later in Sec. IV. This not only provides a runtime advantage, but also an energy advantage since the M-matrix only needs to be read and streamed through the array once. 

密集的不规则矩阵。图4c示出了如何映射稠密的不规则矩阵。脉动阵列设计由于其刚性结构而无法充分利用。尽管在密集的不规则KN矩阵中有16个元素，在收缩阵列中有16个PE，但一次只能映射一半的矩阵。这是由于脉动阵列的刚性形状。所有部分金额都通过转发向下累积到一列；同时将另一半稠密的不规则N矩阵映射到收缩阵列会导致功能不正确，因为不应将累加的输出（a.A + b.I）添加到（a.E + b.M）。因此，一旦矩阵的前半部分被计算出来，就必须加载N矩阵的后半部分，这将使计算时间增加一倍以上。相比之下，Flex-DPE可以一次绘制所有密集的不规则固定元素，从而完全利用所有PE。正如我们稍后将在本节中介绍的那样，它具有灵活的归约网络，可以分别和同时积累两组输出（a.A + b.I）和（a.E + b.M）。 IV。这不仅提供了运行时优势，而且还提供了能源优势，因为仅需读取M矩阵并将其流式传输一次即可。

![20200902091035](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902091035.png)


>Sparse Irregular Matrices. Fig. 4d shows how sparse irregular matrices are mapped. Not only does a systolic array suffer under-utilization from irregular matrices, but also from sparsity. To maintain correctness of the final output, a systolic array must map the non-zero values onto the compute unit. This limitation comes due to the rigid forwarding network between PEs. The Flex-DPE design is able to map only non-zero elements because of the flexible distribution and reduction networks. There are two different dataflows enabling sparse compute in a Flex-DPE. The N-sta, M-str dataflow for Flex-DPE in Fig. 4d maps only non-zero elements onto the compute, giving it 100% stationary utilization, making it more efficient than the systolic array. However, the streaming matrix may send zerovalued elements. This is because all non-zero stationary elements are mapped if there is at least one streaming value that needs to be multiplied with it.

稀疏的不规则矩阵。 图4d显示了稀疏不规则矩阵的映射方式。 收缩阵列不仅会因矩阵不规则而利用不足，而且会因稀疏性而遭受损失。 为了保持最终输出的正确性，脉动阵列必须将非零值映射到计算单元上。 该限制归因于PE之间的刚性转发网络。 由于灵活的分配和归约网络，Flex-DPE设计仅能够映射非零元素。 在Flex-DPE中，有两种不同的数据流可实现稀疏计算。 图4d中Flex-DPE的N-sta，M-str数据流仅将非零元素映射到计算上，使其具有100％的固定利用率，使其比脉动阵列更高效。 但是，流矩阵可以发送零值元素。 这是因为如果至少有一个流值需要与其相乘，则将映射所有非零固定元素。

![20200902091437](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902091437.png)

>Fig. 4e shows the N-str, M-str dataflow (i.e., No Local Reuse [10]) for the Flex-DPE that fully utilizes the compute. This is done by streaming only necessary multiplication pairs and not keeping any values stationary. We provide more details about the distribution and reduction network architecture that can enable this feature in Section IV. The equivalent dataflow is not possible for the systolic array as it does not allow arbitrary pairings of vectors from the M and N matrices due to its rigid cycle-by-cycle forwarding network.


>Distribution and Reduction Latency. Another point to notice from the quantitative data in Fig. 4b-e is that the data loading and accumulation time in systolic arrays is always proportional to the array dimensions, while SIGMA’s networks allow O(1) distribution and O(log2N) reduction.

分配和减少延迟。 从图4b-e中的定量数据中要注意的另一点是，收缩压阵列中的数据加载和累积时间始终与阵列尺寸成比例，而SIGMA的网络允许O（1）分布和O（log2N）减少。

![20200902091721](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902091721.png)

>Summary. Table I summarizes the sources of inefficiency in systolic arrays and how SIGMA addresses each. Architectural details are provided next.

摘要。 表I总结了脉动阵列效率低下的原因以及SIGMA如何解决每个问题。 接下来提供建筑细节。

![20200902091903](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902091903.png)

>* IV. SIGMA ARCHITECTURE

>The fundamental building block within SIGMA’s compute fabric is a processor named Flexible Dot Product Engine (Flex-DPE), described in Sec. IV-A. Several Flex-DPEs are connected together via a simple NoC to create the full SIGMA compute fabric. Each GEMM operation reserves a contiguous group of Flex-DPEs, creating a Flexible Dot Product Unit (Flex-DPU), described in Sec. IV-B. The memory-system is similar to the TPU [4], [23]. Fig. 8 depicts the high level schematic of SIGMA.

SIGMA计算结构的基本组成部分是一个名为“ Flexible Dot Product Engine”（Flex-DPE）的处理器，如第 IV-A二节所述。 几个Flex-DPE通过简单的NoC连接在一起，以创建完整的SIGMA计算结构。 每个GEMM操作都会保留一组连续的Flex-DPE，以创建一个灵活的点产品单元（Flex-DPU），如第IV-B节所述。 该存储系统类似于TPU [4]，[23]。 图8描绘了高层次SIGMA的示意图。
![20200902092724](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902092724.png)



>A. Microarchitecture of Flex-DPE
>A k-sized Flex-DPE houses k multipliers, k − 1 adders, local buffers, control unit, and flexible interconnects. The multipliers are laid out in a logical 1-D structure. Fig. 5 shows an overview. A 1D substrate enables us to run matrixmatrix (M*M) multiplications as multiple vector matrix multiplications (V*M), similar to Brainwave [14]. Recall from Fig. 4 that a k-sized square systolic array has √k columns and √ k rows, with each column computing an independent dot-product when running a weight [23] or inputstationary dataflow [37]. In contrast, a k-sized Flex-DPE can be configured to create myriad combinations of dot-products: one dot-product of size k, two dot-products of size k/2, √k dot-products of size √ k (like the systolic array), and so on. In fact, the flexible distribution and reduction networks also enable creation of multiple variable-sized dot-products, which is crucial for sparsity. In Sec. V, we study how the Flex-DPE scales with area and power.

一个k大小的Flex-DPE包含k个乘法器，k-1个加法器，本地缓冲区，控制单元和灵活的互连。 乘法器以逻辑一维结构布置。 图5显示了概述。 一维底物使我们能够将矩阵矩阵（M * M）乘法作为多个矢量矩阵乘法（V * M）进行，类似于Brainwave [14]。 从图4回忆起，一个k大小的方形脉动阵列具有√k列和√k行，当运行权重[23]或输入平稳数据流[37]时，每列计算一个独立的点积。 相比之下，可以将k大小的Flex-DPE配置为创建多种点积组合：一个大小为k的点积，两个大小为k / 2的点积，√k大小为√k的点积（ 例如脉动阵列），等等。 实际上，灵活的分配和减少网络还可以创建多个可变大小的点产品，这对于稀疏至关重要。 在秒 V，我们研究Flex-DPE如何随面积和功率缩放。

![20200902093202](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902093202.png)

>1) Distribution Network: Benes Topology: The role of a distribution network within any GEMM engine is to load the stationary matrix (MN or KN), and stream the other matrix, as shown in Fig. 4. In a systolic array, the distribution network behavior is implemented via the horizontal and vertical forwarding links between PEs. This leads to an O(k) data loading time for a k ×k systolic array.

1）分发网络：Benes拓扑：任何GEMM引擎中的分发网络的作用是加载固定矩阵（MN或KN），并传输另一个矩阵，如图4所示。 分布网络行为是通过PE之间的水平和垂直转发链路实现的。 这导致k×k脉动阵列的O（k）数据加载时间。

>In SIGMA, we adopt a Benes network [7] to support the flexibility demonstrated in Fig. 4. Benes is a non-blocking N-input N-output multistage network with 2log(N)+1 levels, each with N tiny 2x2 switches. The switches, as shown in Fig. 5-Step(iv), require two control bits; one for selecting the vertical output and one for diagonal output. Numerous Benes routing algorithms have been proposed [7], [8], [29]. The non-blocking property of Benes allows any source to communicate with any destination without any intermediate contention. We use latch-free switches (except for timing closure) at each-stage, allowing a O(1) data communication across the Benes network. We also support multicasts within the Benes network to avoid having to read the same data element multiple times from SRAM.

在SIGMA中，我们采用Benes网络[7]来支持图4所示的灵活性。Benes是具有2log（N）+1级的无阻塞N输入N输出多级网络，每个级都有N个2x2小型开关。 。 如图5-步骤（iv）所示，这些开关需要两个控制位。 一种用于选择垂直输出，另一种用于对角线输出。 已经提出了许多Benes路由算法[7]，[8]，[29]。 Benes的非阻塞属性允许任何源与任何目标进行通信，而无需任何中间争用。 我们在每个阶段都使用无闩锁开关（时序收敛除外），从而允许在Benes网络上进行O（1）数据通信。 我们还支持Benes网络中的多播，以避免必须多次从SRAM读取同一数据元素。

![20200902100103](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902100103.png)

>Fig. 5-Step(iv), require two control bits; one for selecting the vertical output and one for diagonal output. Numerous Benes routing algorithms have been proposed [7], [8], [29]. The non-blocking property of Benes allows any source to communicate with any destination without any intermediate contention. We use latch-free switches (except for timing closure) at each-stage, allowing a O(1) data communication across the Benes network. We also support multicasts within the Benes network to avoid having to read the same data element multiple times from SRAM.

图5-步骤（iv），需要两个控制位； 一种用于选择垂直输出，另一种用于对角线输出。 已经提出了许多Benes路由算法[7]，[8]，[29]。 Benes的非阻塞属性允许任何源与任何目标进行通信，而无需任何中间争用。 我们在每个阶段都使用无闩锁开关（时序收敛除外），从而允许在Benes网络上进行O（1）数据通信。 我们还支持Benes网络中的多播，以避免必须多次从SRAM读取同一数据元素。

>Other design-choices are also feasible for the distribution network. A crossbar gives the same non-blocking behavior as Benes and has much simpler routing, but it does not scale well (N2). Blocking interconnects such as buses [10], trees [11], [27], butterfly and mesh networks [9], are still valid design choices due to their low wire costs, but will cause performance degradation due to increased distribution delays.

对于分配网，其他设计选择也是可行的。 交叉开关具有与Benes相同的非阻塞行为，并且布线简单得多，但伸缩性不佳（N2）。 诸如总线[10]，树[11]，[27]，蝶形和网状网络[9]之类的阻塞互连由于其较低的电线成本仍然是有效的设计选择，但由于分配延迟的增加会导致性能下降。

> 2)Reduction Network: FAN Topology: Dot-product reduction can be implemented in three ways.
>Spatio-Temporal Reduction: The TPU weight-stationary systolic array implementation performs reduction via forwarding along the column, requiring O(k)-cycles for a k×k array. The time taken is independent of the actual size m of the dot-product which may be smaller.
还原网络：FAN拓扑：可以通过三种方式实现点产品还原。
时空缩减：TPU重量平稳脉动阵列实现通过沿列转发进行缩减，对于k×k阵列需要O（k）个周期。 所花费的时间与点积的实际大小m无关，后者可能较小。

>Temporal Reduction: Designs like EIE [19] perform inplace reduction within PEs. The time taken is still linear like spatio-temporal, but equal to O(m) - i.e., the dot-product size.
时间减少：EIE等设计[19]在PE中执行原位减少。 所花费的时间仍然像时空一样是线性的，但是等于O（m）-即点积大小。

>Spatial Reduction: In SIGMA, we implement a spatial tree-based reduction, as it requires O(log2m) cycles, for a m-sized dot-product. The challenge with realizing this log2mcycle reduction, however, is that non-powers of two sized reductions are hard to map over traditional binary adder trees. Suppose we are trying to accumulate three separate dot-products for (a0, a1, a2, b0, b1, c0, c1, c2) on an eightwide adder tree. Following the natural binary-tree topology, a2-b0 and b1-c0 will reach the same adder as they go up the tree, which is incorrect functionally.

空间缩减：在SIGMA中，我们实现了基于空间树的缩减，因为它需要O（log2m）周期才能生成m尺寸的点积。 但是，实现这种log2mcycle减少的挑战在于，很难将传统的二进制加法器树映射成两次大小的非幂。 假设我们试图在一个八度加法器树上累积三个单独的点积（a0，a1，a2，b0，b1，c0，c1，c2）。 按照自然的二叉树拓扑结构，a2-b0和b1-c0到达树时将到达相同的加法器，这在功能上是不正确的。

>FAN Topology. To address this issue, we propose a novel adder-tree topology named Forwarding Adder Network (FAN) that places forwarding links between different levels of adders over a traditional binary adder tree. The topology and variable labels of a 32-wide FAN are shown in Fig. 6a. VecIDs and adderIDs are numbered in increasing order from left to right, and each adderID has a corresponding adderLvl value. Below is a pseudocode describing the link connections between adders to create FAN of any power of 2 size.
风扇拓扑。 为了解决这个问题，我们提出了一种新颖的加法树拓扑结构，称为转发加法器网络（FAN），该拓扑将转发链接放置在传统二进制加法器树上不同级别的加法器之间。 图6a中显示了32宽FAN的拓扑和变量标签。 VecID和adderID从左到右按递增顺序编号，并且每个adderID都有一个对应的adderLvl值。 下面是一个伪代码，描述加法器之间的链接连接，以创建任意大小为2的幂的FAN。
![20200902100746](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902100746.png)

>Routing Algorithm. The routing algorithm for FAN is shown in Fig. 6c. For every adder, if vecID[adderID] equals to vecID[adderID+1], accumulation is enabled. If the vecIDs are not equal and the adder is in the zeroth level, the bypass link is enabled. For example, in Fig. 6a, Adder 12 needs to bypass ‘c’ and ‘d’ to the next adder levels. From the second adder level onward, there is a N-to-2 mux before every FP32 Adder. To determine which inputs get selected, comparators are used to identify cluster regions.
路由算法。 FAN的路由算法如图6c所示。 对于每个加法器，如果vecID [adderID]等于vecID [adderID + 1]，则启用累加。 如果vecID不相等且加法器处于零级，则启用旁路链接。 例如，在图6a中，加法器12需要绕过“ c”和“ d”到下一个加法器级别。 从第二个加法器级别开始，每个FP32加法器之前都有一个N至2多路复用器。 为了确定选择哪些输入，使用比较器来识别群集区域。
![20200902101027](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902101027.png)

>Benefits and Overhead. FAN offers similar benefits as the ART topology proposed in MAERI [27] in terms of creating dot-products of variable sizes. However, FAN is much more lightweight. This is because MAERI’s ART is built using three input adders (two from parent nodes, one from a sibling node), which makes it extremely prohibitive, especially when working with FP32 data type (commonly used during DNN training). Fig. 6b shows the performance evaluation between linear reduction (i.e., temporal or spatiotemporal), ART, and FAN. For performance calculations, we use 100 stationary folds (when stationary elements need to be replaced) with stream dimension of 1000 each. As shown in Fig. 6b-iii, taking logN cycles rather than N cycles before starting the next fold significantly improves performance as the number of PEs increases. Our findings show that 512PE FAN only has a 10% and 31% area power overhead over linear, compared to ART which has a 92% and 86% overhead respectively. FAN also provides EDP benefits over linear starting from 128-PE. At 512-PE, FAN’s EDP is 45% and 34% lower than linear and ART respectively. From our results, we conclude that FAN is both high performance and scalable.

好处和开销。在创建可变大小的点积方面，FAN提供了与MAERI [27]中提出的ART拓扑相似的优势。但是，FAN更轻巧。这是因为MAERI的ART是使用三个输入加法器（两个来自父节点，一个来自同级节点）建立的，这使其具有极大的禁止性，尤其是在处理FP32数据类型（通常在DNN训练期间使用）时。图6b示出了线性减少（即，时间或时空），ART和FAN之间的性能评估。对于性能计算，我们使用100个固定折痕（需要更换固定元件时），每个折痕尺寸为1000。如图6b-iii所示，随着PE数量的增加，在开始下一个折叠之前采取logN个周期而不是N个周期可以显着提高性能。我们的研究结果表明，相比于ART，512PE FAN的线性开销分别只有10％和31％，而ART的开销分别为92％和86％。与从128-PE开始的线性相比，FAN还具有EDP的优势。在512-PE的情况下，FAN的EDP分别比线性和ART低45％和34％。根据我们的结果，我们得出结论FAN具有高性能和可扩展性。
![20200902101214](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902101214.png)

>3) Execution within Flex-DPE: The Flex-DPE design allows mapping for dense or sparse, and regular or irregular GEMMs. In Fig. 4, we have seen how different combinations of matrices are mapped onto Flex-DPE. Fig. 5 depicts the steps involved in generating the mapping for sparse matrices which we will describe later.

在Flex-DPE中执行：Flex-DPE设计允许映射密集或稀疏，规则或不规则的GEMM。 在图4中，我们看到了矩阵的不同组合是如何映射到Flex-DPE上的。 图5描述了生成稀疏矩阵映射所涉及的步骤，我们将在后面描述。

>B. Composing Flex-DPEs into a Flex-DPU using a NoC

>To extract maximum possible parallelism from the available multipliers, SIGMA can dynamically compose a number of Flex-DPE units together to form a logical ensemble which we call Flex-DPU. A Flex-DPU is responsible for running one GEMM. Multiple Flex-DPUs can be scheduled in parallel to run multiple GEMMs. The NoC connecting the Flex-DPEs together is similar to that of tiled accelerator architectures, such as Tangram [16] and Simba [39].
为了从可用的乘法器中提取最大可能的并行度，SIGMA可以将多个Flex-DPE单元动态组合在一起，形成一个逻辑集合，我们称之为Flex-DPU。 一个Flex-DPU负责运行一个GEMM。 可以并行调度多个Flex-DPU，以运行多个GEMM。 将Flex-DPE连接在一起的NoC类似于诸如Tangram [16]和Simba [39]的平铺加速器体系结构。


>A simple switch is present at the intersection of each Flex-DPE to arbitrate the flow of the data. These switches are connected together in a 2D mesh. They are statically configured when mapping the GEMMs, and do not require any dynamic routing or flow-control like conventional NoCs. The amount of bandwidth on this NoC (i.e., number of unique elements of the row/column that can be transferred per-cycle) is a design-time configurable parameter.

每个Flex-DPE的交集处都有一个简单的开关，用于仲裁数据流。 这些开关以2D网格连接在一起。 它们在映射GEMM时是静态配置的，并且不需要像常规NoC一样的任何动态路由或流控制。 此NoC上的带宽量（即每个周期可以传输的行/列的唯一元素数）是设计时可配置的参数。

>Within a Flex-DPU, the switch forwards data across FlexDPEs, providing seemless multicasts of data like a bus. We describe this with an example in Sec. IV-E. Across FlexDPUs, the switches provide hop-by-hop data forwarding, similar conventional NoCs. 
在Flex-DPU中，交换机在FlexDPE之间转发数据，从而像总线一样提供数据的多播。 我们在第二节中用一个例子来描述。 我有。 跨FlexDPU，这些交换机提供逐跳数据转发，类似于传统的NoC。 

>C. Supporting Unstructured Sparsity
>Compression Format. One of the key benefits of supporting sparsity is low-memory footprint; and consequently more energy savings by avoiding zero-valued element transfers. There are a few well recognized compression formats such as CSC, CSR, COO, and RLC (Run-length compression). We use a Bitmap format within SIGMA, where each element has a corresponding bit to indicate if a given element is zero or non-zero in the corresponding matrix [17], [24], [35], [36]. Fig. 7 compares the metadata overhead of various compression formats with varying levels of sparsity. The dimensions and sparsity levels in the plot reflect what we observe in our workloads (see Sec. II). The metadata overhead for COO/ CSR/ CSC changes drastically at various sparsity regions. This is because each nonzero element require indices of log2(dimension) bits, etc. The Bitmap format has a constant meta-data overhead irrespective of sparsity, making it attractive for SIGMA which targets arbitrary unstructured sparsity. At low-levels of sparsity, we find Bitmap having lower footprint than COO/ CSR/ CSC. Bitmap has comparable overhead to RLC [10], [19], at sparse ratio of ∼30% to ∼70%. We observe that RLC is better at reducing meta-data over Bitmap at >∼70% sparsity, but is worse at <∼30% sparsity. We evaluate RLC using 4-bit (RLC-4) and 2-bit (RLC-2) run lengths. Alternate compression formats can be supported over SIGMA by only changing the front end controller to ensure proper mapping.

压缩格式。支持稀疏性的主要好处之一是内存不足。避免零值元素转移，从而节省更多能源。有一些公认的压缩格式，例如CSC，CSR，COO和RLC（行程压缩）。我们在SIGMA中使用位图格式，其中每个元素都有一个对应的位来指示给定元素在对应的矩阵[17]，[24]，[35]，[36]中是零还是非零。图7比较了具有不同稀疏性级别的各种压缩格式的元数据开销。图中的维度和稀疏度反映了我们在工作负载中观察到的情况（请参见第二节）。在各种稀疏区域中，COO / CSR / CSC的元数据开销会急剧变化。这是因为每个非零元素都需要log2（维）位的索引，等等。位图格式具有不变的元数据开销，而与稀疏性无关，这使其对以任意非结构化稀疏性为目标的SIGMA有吸引力。在稀疏性较低的级别，我们发现位图的占用空间比COO / CSR / CSC低。位图的稀疏率为〜30％到〜70％，其开销可与RLC [10]，[19]相比。我们观察到，RLC在稀疏度大于70％时比Bitmap更好地减少元数据，而在稀疏度小于30％时更差。我们使用4位（RLC-4）和2位（RLC-2）游程长度评估RLC。只需更改前端控制器以确保正确的映射，即可在SIGMA上支持其他压缩格式。


>Sparsity Controller. For each GEMM, a global controller determines how sparse matrices are decoded and mapped onto SIGMA. The controller operates on the bitmap metadata and calculates how many Flex-DPEs are needed. Internal counters and tables are implemented to determine the indices where dense computations are needed. We describe the details with a walkthrough example in Sec. IV-E.

稀疏控制器。 对于每个GEMM，全局控制器确定稀疏矩阵如何解码并映射到SIGMA。 控制器对位图元数据进行操作，并计算需要多少个Flex-DPE。 内部计数器和表用于确定需要密集计算的索引。 我们在Sec中通过演练示例来描述细节。

>D. Dataflows Supported by SIGMA

>SIGMA’s flexible substrate enables it to support myriad dataflows. For all workloads, we use both Weight (i.e., KN) stationary and Input (i.e., MK) stationary dataflows (Fig. 4d), and pick the one that provides higher efficiency. In these two dataflows, spatial dot-products of dynamic size are created depending on the matrix dimensions and sparsity of the stationary matrix. The columns/rows of the streaming matrix are reused spatially by broadcasting to the rows/columns of the stationary matrix (which are reused temporally at each multiplier). SIGMA can also run a No Local Reuse (i.e., MN-str, KN-str dataflow from Fig. 4e). This dataflow can provide 100% utilization of the compute, but comes at the cost of requiring higher interconnect bandwidth.
SIGMA的柔性基板可支持多种数据流。 对于所有工作负载，我们同时使用权重（即KN）固定数据流和输入（即MK）固定数据流（图4d），并选择能提供更高效率的数据流。 在这两个数据流中，根据矩阵尺寸和固定矩阵的稀疏性创建动态大小的空间点积。 流媒体矩阵的列/行通过广播到固定矩阵的行/列在空间上被重用（在每个乘法器上时间上被重用）。 SIGMA还可以运行“无本地重用”（即，图4e中的MN-str，KN-str数据流）。 此数据流可以提供100％的计算利用率，但以需要更高的互连带宽为代价。
![20200902101552](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902101552.png)

>E. Walkthrough Example
>The following steps (corresponding to Fig. 5) depicts a walk-through example showing how SIGMA utilizes the bitmap format to map sparse GEMMs onto Flex-DPEs. Here, the number of multipliers per Flex-DPE (Nmult) is four.
>•	Step i) Gather two bitmap-compressed matrices. In this example, MK is stationary and KN is streaming.
>•	Step ii) Conduct row-wise OR across the streaming bitmap and store the outputs to REGOR (temporary registers). Then, do element-wise AND between the corresponding REGOR row and stationary bitmap column to generate stationary’ bitmap.
>•	Step iii) The number of ones in stationary’ bitmap corresponds to the number of useful stationary value (Nstat). Since Nstat is 8 and Nmult is 4 in this example, 2-Flex-DPE units are needed to form a single Flex-DPU.
>•	Step iv) Unicast the stationary values to the multiplier buffers. The routing is straightforward, as all stationary input data travel vertically down. In this example, the input bus has a 4X bandwidth, so it is only able to fill one Flex-DPE each cycle.
>•	Step v) To generate source and destination pairs for each Flex-DPE, a counter value is assigned to each non-zero element in the stationary’ and streaming bitmaps. For stationary’ bitmap, the counter starts at 0 and increments from left-right, top-bottom. The counter resets when it reaches Nmult -1, which marks the end of one Flex-DPE. Counter values increments top-bottom in the streaming bitmap and resets at the start of each column. Then, a streaming bitmap column compares to each row of the corresponding stationary’ bitmap. If both values are 1, the counter values are stored in the Flex-DPE SRC-DEST tables. The row-id is recorded to determine partial sum regions. Additionally, an initial output bitmap is generated based on if there are any non-zero computations.
>•	Step vi) Generate distribution routing bits base on the SRC-DEST table entries. For this example, a naive routing algorithm with limited functionality is to subtract the src-index with the dest-index. Other routing algorithms have been proposed [7], [8].
>•	Step vii) Finally, the streaming values are broadcasted to all Flex-DPEs within a Flex-DPU from the routing bits calculated in Step vi. For reduction, the accumulation ID is processed and then used as the vecID in FAN, described in Section IV-A2. Multicasts, multiplications, and reductions are all happening in a pipelined fashion. Once all the columns have been streamed in and outputs are generated, the Flex-DPE units are freed up to be utilized for another GEMM operation.

以下步骤（对应于图5）描绘了一个演练示例，该示例显示了SIGMA如何利用位图格式将稀疏的GEMM映射到Flex-DPE上。在此，每个Flex-DPE（Nmult）的乘法器数量为四个。
>•步骤i）收集两个位图压缩的矩阵。在此示例中，MK是固定的，而KN是流式的。
>•步骤ii）跨流位图进行逐行或运算，并将输出存储到REGOR（临时寄存器）。然后，在相应的REGOR行和固定位图列之间进行逐元素AND生成固定位图。
>•步骤iii）固定位图中的位数等于有效固定值（Nstat）的数量。由于在此示例中Nstat为8，Nmult为4，因此需要2-Flex-DPE单元来形成单个Flex-DPU。
>•步骤iv）将固定值单播到乘法器缓冲区。路由很简单，因为所有固定输入数据都是垂直向下传播的。在此示例中，输入总线的带宽为4倍，因此每个周期只能填充一个Flex-DPE。
>•步骤v）为了为每个Flex-DPE生成源对和目标对，将计数器值分配给固定位图和流式位图中的每个非零元素。对于固定的位图，计数器从0开始，从左上右下递增。当计数器达到Nmult -1（表示一个Flex-DPE结束）时，计数器将重置。计数器值在流式位图中从上到下递增，并在每列的开头重置。然后，流式位图列会与相应固定式位图的每一行进行比较。如果两个值均为1，则计数器值存储在Flex-DPE SRC-DEST表中。记录行标识以确定部分和区域。另外，基于是否存在任何非零计算来生成初始输出位图。
>•步骤vi）根据SRC-DEST表条目生成分发路由位。对于此示例，功能有限的朴素路由算法是用dest-index减去src-index。已经提出了其他路由算法[7]，[8]。
>•步骤vii）最后，将流值从步骤vi中计算出的路由位广播到Flex-DPU中的所有Flex-DPE。为了进行减少，处理累积ID，然后将其用作FAN中的vecID，如第IV-A2节所述。多播，乘法和归约都是以流水线方式发生的。一旦所有的列都被流化并生成了输出，就释放了Flex-DPE单元以用于另一GEMM操作。

>V. IMPLEMENTATION
> Fig. 8 compares the post place-and-routed area and power of a 128×128 systolic array versus SIGMA with 128 Flex-DPEs, each of size 128. Both designs have identical input bandwidth of 128 words per cycle from SRAM. SIGMA’s key overheads are the highly flexible, non-blocking distribution and reduction networks that lead to a 37.7% area overhead. However, the performance speedups provided by SIGMA (shown later in Sec. VI-C) lead to an average 3.2× improvement in Effective TFLOPs/ Watt. We expect a further power performance gain of 7× when scaling from a 28nm design to a 12nm design. This is based on FP32 FLOPs growth between NVIDIA K20X to NVIDIA T4 where compute grew by ∼2× while power reduced by ∼3.5×. SIGMA is pipelined at 1-cycle distribution, 1-cycle multiplication, and 1-cycle for each level of reduction. The critical path for SIGMA is the distribution, but it is possible to match the maximum operating frequency of TPU by pipelining the distribution further so that the new critical path becomes the FP compute. Additionally, we estimate a global controller with 1024 AND gates, 1024 OR gates, 1024 counters, and 128 SRC-DEST tables to consume approximately 1.4mm2.

图8比较了128×128脉动阵列与SIGMA与128个Flex-DPE（每个大小为128）的后期布局和布线面积以及功率。两种设计在SRAM中每个周期具有相同的128字输入带宽。 SIGMA的主要间接费用是高度灵活，无阻塞的分销和减少网络，可导致37.7％的区域间接费用。但是，SIGMA提供的性能提升（稍后在VI-C节中显示）导致有效TFLOP / Watt平均提高3.2倍。从28nm设计扩展到12nm设计时，我们预计功率性能将进一步提高7倍。这基于NVIDIA K20X和NVIDIA T4之间的FP32 FLOP增长，其中计算增长了约2倍，而功耗却下降了约3.5倍。 SIGMA以1周期分布，1周期乘法和1周期降级的方式进行流水线传输。 SIGMA的关键路径是分布，但是可以通过进一步流水分布来匹配TPU的最大工作频率，从而使新的关键路径成为FP计算。此外，我们估计具有1024个AND门，1024个OR门，1024个计数器和128个SRC-DEST表的全局控制器消耗约1.4mm2的空间。
![20200902102006](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902102006.png)

>For 16384 total PEs, we performed a design-space exploration for sizing Flex-DPE units to find the most energy and area efficient configuration. Fig. 9 depicts that a Flex-DPE of size 128 Flex-DPE consumes the least energy, while a Flex-DPE size of 512 is the most area efficient. We decide to use Flex-DPE-128 to match the per-cycle SRAM read bandwidth of the TPU.

对于总共16384个PE，我们进行了设计空间探索，以调整Flex-DPE单元的尺寸，以找到最节能和最省电的配置。 图9描绘了大小为128的Flex-DPE消耗的能量最少，而大小为512的Flex-DPE的区域效率最高。 我们决定使用Flex-DPE-128来匹配TPU的每周期SRAM读取带宽。
![20200902102024](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902102024.png)

>VI. EVALUATION
>A. Methodology
>Target GEMMs. We use GEMM dimensions and sparsity observed during training of GNMT, Transformer, NCF and DeepBench models (described earlier in Sec. II). Input and weight sparsity were observed to be ∼10-50% and ∼80%.

A.方法论
目标GEMM。 我们使用在GNMT，Transformer，NCF和DeepBench模型训练中观察到的GEMM尺寸和稀疏度（在第二节中已有介绍）。 输入和重量稀疏度分别为〜10-50％和〜80％。

>Baseline Accelerators. We compare SIGMA against other state-of-the-art accelerators: TPU [4], EIE [19], SCNN [33], OuterSPACE [32], Eyeriss v2 [11], Packed Systolic [26], and Cambricon-X [47]. We scale the number of PEs to a constant number of 16384 in all designs. SIGMA assumes 128 FlexDPEs, each with 128 MACs, and input SRAM bandwidth of 128x (number of unique data elements that can be distributed). For our evaluations, we allow greater input bandwidth to distribute larger chunks of the streaming matrix in one cycle. For sparsity performance, all combinations of matrices and sparsity were tested and then averaged. Most of the sparse accelerators were designed for inference and specialized for convolutions; we extended them to run GEMMs by setting equal input and filter dimensions.

基准加速器。 我们将SIGMA与其他最新的加速器进行了比较：TPU [4]，EIE [19]，SCNN [33]，OuterSPACE [32]，Eyriss v2 [11]，Packed Systolic [26]和Cambricon-X [47]。 在所有设计中，我们将PE的数量扩展到恒定的16384。 SIGMA假定有128个FlexDPE，每个都有128个MAC，输入SRAM带宽为128x（可以分配的唯一数据元素的数量）。 为了进行评估，我们允许更大的输入带宽在一个周期内分配更大的流矩阵块。 对于稀疏性能，对矩阵和稀疏性的所有组合进行了测试，然后取平均值。 大多数稀疏加速器是为推理而设计的，专门用于卷积。 我们通过设置相等的输入和过滤器尺寸来扩展它们以运行GEMM。

>Simulation Infrastructure. To evaluate the performance of SIGMA and other baselines, we developed a cycle-accurate analytic model. The model evaluates the performance based on the number of compute units, buffers per compute unit, interconnect topology, input bandwidth, and dataflow. The TPU was modeled using SCALE-sim [37].

仿真基础架构。 为了评估SIGMA和其他基准的性能，我们开发了一个周期精确的分析模型。 该模型基于计算单元的数量，每个计算单元的缓冲区，互连拓扑，输入带宽和数据流来评估性能。 使用SCALE-sim [37]对TPU进行建模。

>Comparison Metrics. Table II defines comparison metrics we use across our evaluation graphs.
比较指标。 表II定义了我们在评估图中使用的比较指标。
![20200902101903](https://raw.githubusercontent.com/milk2we/picgo/master/images/20200902101903.png)


>B. Characterizing SIGMA’s Features
>We start by characterizing the benefits of the various features of SIGMA to help pick the optimal design-point.
>Dataflow Comparison Fig. 10 demonstrates the impact of dataflows when running a suite of representative GEMMs. We observe that the MK-str,KN-str dataflow, while being ideal in terms of no wasted computations, suffers in overall latency. This is because it requires extremely high bandwidth (thus serialization), due to no reuse within the Flex-DPE. For the other two dataflows, the stationary matrix maps only non-zeros, getting 100% utilization, and the overall efficiency gets determined by the sparsity of the streaming matrix. In our evaluations, we run both dataflows and report the higher performing dataflow.

>Benefits of SIGMA’s features Fig. 11 revisits the discussion from Sec. III-C and quantitatively demonstrates the benefits of SIGMA’s three key features in comparison to systolic arrays: dot-products within Flex-DPEs, (ii) scalable interconnects, namely, Benes and FAN, providing O(1) and O(log2N) distribution and reduction time respectively, and (iii) sparsity support to map only useful non-zero data.

>For sparse irregular GEMMs, TPU is required to map all elements stationary, while SIGMA maps only the nonzeros stationary. With sparsity support, SIGMA shows 100% stationary utilization. Due to increased utilization and compute efficiency, fewer cycles are needed to load and reduce data. Fig. 11 shows two versions of sparse irregular GEMMs. The M-str,N-sta example is dominated by streaming latency because the larger matrix is being streamed in, while the loading latency dominates in M-sta,N-str because the larger matrix is held stationary and leads to more folding iterations. The compute efficiency for M-sta,N-str is significantly higher because the sparser matrix is held stationary.

>C. Performance and Energy versus TPU
>Speedup. Fig. 12a and Fig. 12b evaluate dense and sparse GEMMs performance respectively. In Fig. 12a, we use three aspect ratios for the TPU. For e.g., 512×32 have 512 rows, each of which can read a data element per cycle. Either the MK or KN matrix is kept stationary. For the 2048-4096-32 GEMM, a K dimension of 32 leads to under-utilization in the 128×128 and 256×64 TPUs, but aligns with the columns of the 512×32 TPU, giving a huge performance jump. SIGMA, due to its flexibility, experiences a similar jump. The TPU overall efficiency drops steeply while operating a 1024-16-500000 sized GEMM. If a square-shaped TPU maps the KN (500000-16) matrix stationary, the low value of N leads to a 87.5% decrease in utilization. If it decides to map MK (1024-500000) stationary, numerous folds are required since there are only 16K PEs, leading to a large amount of O(N) reductions. SIGMA accelerates this GEMM by creating flexible dimensions and leveraging its O(logN) reduction structure as described in Table I. In SIGMA, the overall efficiency is close to 100% throughout, except for small GEMMs (such as 2048-1-128), where smaller sizes cause loading latency from limited bandwidth to dominate. On average, SIGMA provides speedup of 2x over TPUs on dense GEMMs, stemming from higher utilization and faster reduction. This results to an overall average efficiency of 82% compared to 59% in the TPU. In Fig. 12b we run GEMMs with varying sparsity over SIGMA. We observe that there is a roughly 6× improvement over TPU, which suffers from an average overall efficiency of less than 10% due to the mapped zeros. SIGMA maps no zeros and shows an average overall efficiency of 40%, which gets limited by the sparsity of the streaming matrix.

>Energy. 
In Fig. 13 we see that SIGMA is on an average 3× more energy efficient and 5× more area efficient than TPU for sparse workloads. Despite SIGMA consuming twice as much power (Fig. 8), the energy benefit comes from ∼6× speedup. With more sparsity induced in future workloads, we expect energy gains to be significantly more.

>D. Performance against Sparse Accelerators
>Fig. 14 presents the speedup of SIGMA over state-ofthe-art sparse accelerators. The key inefficiencies in other accelerators are presented in Table III. Of all the sparse accelerators, SIGMA is the only one that can support full spatial-reduction with arbitrary sized dot-products. For two GEMMs, we find SIGMA slower than Eyeriss v2 since the latter can buffer both operands in its local SRAM for further reuse, while SIGMA keeps only one operand stationary, and has to stream the other multiple times (even if it will be reused in future). Other designs like EIE also have local SRAM buffers, but we observe that its inter-PE communication bottleneck overshadows the memory benefits. On average, we observe SIGMA performing 3X faster than the other sparse accelerators. We tested four combinations between the matrices and sparsity level and selected the best performing one for each accelerator.

>VII. RELATED WORK
>Training. A few prior works address training on dense matrices. Pipelayer [40] and Neurocube [25] proposes ReRAM based acceleration, but does not address scalability and sparsity. Hypar [41] addresses the scaling problem in training and proposes optimal techniques to extract parallelism. Schuiki et al. [38] and Liu et al. [30] propose processing in memory approaches to combat the communication problem when scaling training. ScaleDEEP architecture was developed to target DNN training, and consists of many processing tiles with specialized memory subsystem and interconnect [44]. However none of these methods simultaneously address the irregularity, sparsity, and scalability as SIGMA does.

>Sparsity. Table III contrasts SIGMA against state-ofthe-art sparse accelerators. Other recent designs include PermDNN [13], which uses permuted diagonal matrices for inference on structured sparse DNN models. Other designs like UCNN [21] exploits sparsity and weight repetition by reusing dot products. ExTensor [22] finds intersections within compressed representations, and only operates on useful dense computations. Bit-tactical [12] targets sparsity in inference by skipping zero weights and exploiting bit level sparsity of inputs. Unlike SIGMA, Bit-tactical leverages scheduling in software to align inputs and weights. SparseReRAM [46] proposes using small operation units to exploit both weight and activation sparsity in ReRAMs. SIGMA targets acceleration of GEMMs with unstructured sparsity.

>Flexibile Interconnects. Eyeriss [10] proposes an efficient dataflow for leveraging convolutional reuse with reconfigurable buses. MAERI [27] uses tree-based interconnects for distribution and reduction which inspired the 1D FlexDPE microarchitecture in SIGMA. However, MAERI does not handle dynamic weight and activation sparsity, and is optimized for low-precision CNNs rather than high-precision GEMMs commonly used during DNN training. Eyeriss v2 [11] also uses a specialized NoC to handle sparsity, but is optimized for small mobile CNNs rather than large GEMMs.

>VIII. CONCLUSION
>The paper proposes SIGMA as an accelerator to handle emerging large, sparse, and irregular GEMMs. SIGMA provides close to 100% compute utilization via high-bandwidth non-blocking interconnect structures. The overhead for such flexibility is carefully analyzed via a place-and-routed design. Our implementation shows 5.7× performance speedup over TPU designs for sparse irregular workloads. We also observe a 3× performance speedup over other state-of-the-art sparse accelerators. Reference RTL: https://github.com/georgia-tech-synergy-lab/SIGMA






