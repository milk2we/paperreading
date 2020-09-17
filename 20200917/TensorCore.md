# 深入理解Nvidia TensorCore

---

## Volta 微体系结构

    Volta是Nvidia针对于深度学习所提出的第一代GPU加速器；Volta 体系结构由多个片上流处理器组成，并且连接多个内存分区。每个内存分区包含部分最底层的cache，并且连接GPU和片外DRAM。如Nvidia所描述的那样，每个SM内部包含着多个TensorCore。所有的SM被分为四个部分，在Volta架构中，每个部分被称为Sub-Cores。
    ![20200917111356](undefined)
    每个Sub-Cores内部包含着两个TensorCore($4\times4\times4$),一个线程束调度器、一个分配器和64kb的寄存器文件（$512\times 32 Threads \times 32 bits$）.

    除了使用TensorCore以外，Volta还使用了其他的与机器学习性能提升相关的技术。与上一代的Pascal架构相对比，在volta架构中，每一个SM具有两倍的调度单元以及独立的整数和32-bits的浮点型内核。此外，处理发散线程与上一代GPU相比也是不同的，分支以后的所有路径可以通过交错的方式被执行在一个线程束内部的线程。

---

##  线程束矩阵函数（Warp Matrix Function） WMMA API

    CUDA C++ 提供的是warp-level matrix multiply and accumulate(WMMA) API. 众所周知的是，针对于密集型矩阵使用tiling的方式可以显著的提升内存位置的影响。 使用TensorCore也是使用一种tiling的技术，可以将大型矩阵的小的tiles使用TensorCore进行处理。使用WMMA API在线程束内部的线程可以一起协作处理tiles的矩阵乘法和加法。
    
    使用Nvidia的术语，每一个tile被分割进入一个fragments，这里的fragments就是tile元素的集合。fragments被映射到单个线程的寄存器之中。因此，整个输入矩阵被分别输入到不同的线程之中，并且每一个线程仅仅存储tile的一小部分。




