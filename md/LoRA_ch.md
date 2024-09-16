## LoRA: Low-Rank Adaptation Abstract

The **LoRA (Low-Rank Adaptation)** paper introduces a method to reduce trainable parameters and memory requirements for fine-tuning large-scale models, such as **GPT-3 (175B)**. By freezing pre-trained weights and injecting low-rank decomposition matrices, LoRA decreases trainable parameters by **10,000 times** and GPU memory by **3 times**, maintaining performance parity with full fine-tuning without adding inference latency. Check out the full details at [LoRA GitHub](https://github.com/microsoft/LoRA).


LoRA 论文的摘要介绍了一种用于减少大规模预训练模型微调过程中可训练参数数量和内存需求的方法，例如拥有1750亿参数的GPT-3。LoRA 通过冻结模型权重并引入可训练的低秩分解矩阵，减少了10,000倍的可训练参数，并降低了3倍的GPU内存使用量，同时在性能上与完全微调持平，并且没有额外的推理延迟。更多信息请访问 LoRA GitHub。
 [LoRA GitHub](https://github.com/microsoft/LoRA)。

 ### LoRA（低秩适配）的主要优势

1. **共享预训练模型**：LoRA 通过冻结预训练模型，并高效替换低秩矩阵，实现任务间的切换。
2. **提升效率**：LoRA 仅需训练注入的小型低秩矩阵，减少高达三倍的硬件需求。
3. **无推理延迟**：通过将可训练的矩阵与冻结的权重融合，不会引入额外的推理延迟。
4. **兼容性**：LoRA 与诸如前缀微调的许多方法兼容，增加了应用中的灵活性。

### 术语和约定：
这一部分介绍了 LoRA 论文中使用的术语和约定，包括自注意力机制中的投影矩阵 $W_q$, $W_k$, $W_v$, 和 $W_o$，以及预训练权重矩阵 $W_0$，和梯度更新 $\Delta W$。

### 完全微调过程：


在完全微调过程中，模型初始化为预训练的权重 $\Phi_0$，并通过梯度下降反复更新为 $\Phi_0 + \Delta \Phi$，以最大化条件语言建模的目标函数：


$$
\max_{\Phi} \sum_{(x,y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left( P_{\Phi} (y_t | x, y_{<t}) \right)
$$


其中一个主要缺点是，对于每个下游任务，必须学习一组不同的参数 $\Delta \Phi$，其维度等于 $|\Phi_0|$。因此，如果预训练模型很大（例如 GPT-3 具有约 1750 亿参数），存储和部署多个独立的微调模型将非常具有挑战性，甚至不可行。


为了解决这一问题，本文采用了一种更加高效的参数化方法，任务特定的参数增量 $\Delta \Phi = \Delta \Phi (\Theta)$ 被进一步编码为一个更小的参数集 $\Theta$，其维度 $|\Theta| \ll |\Phi_0|$。优化 $\Delta \Phi$ 的任务变为优化 $\Theta$：


$$
\max_{\Theta} \sum_{(x,y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left( p_{\Phi_0 + \Delta \Phi (\Theta)} (y_t | x, y_{<t}) \right)
$$

在随后的部分中，我们提出了一种使用低秩表示来编码 $\Delta \Phi$，这既高效又节省内存。对于 GPT-3 这种 1750 亿参数的预训练模型，可训练的参数 $\Theta$ 数量可以小至 $\Phi_0$ 的 0.01%。

## 低秩参数化更新矩阵

神经网络包含许多执行矩阵乘法的全连接层。这些层中的权重矩阵通常具有全秩。在适应特定任务时，Aghajanyan 等（2020）指出，预训练语言模型具有低“内在维度”，即便在投影到较小子空间时仍能有效学习。

基于此，我们假设权重的更新在适应过程中也具有低“内在秩”。对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，我们通过低秩分解 $W_0 + \Delta W = W_0 + BA$ 来约束其更新，其中 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$，且秩 $r \ll \min(d, k)$。

在训练过程中，$W_0$ 被冻结且不接受梯度更新，而 $A$ 和 $B$ 包含可训练参数。注意，$W_0$ 和 $\Delta W = BA$ 使用相同输入进行乘法运算，输出向量按坐标相加。对于 $h = W_0 x$，我们修改后的前向传递变为：

$$ h = W_0 x + \Delta W x = W_0 x + BA x $$

我们在图 1 中展示了这种重新参数化方法。我们为 $A$ 使用随机高斯初始化，并将 $B$ 初始化为零，因此在训练开始时 $\Delta W = BA$ 为零。然后我们通过 $\frac{\alpha}{r}$ 缩放 $\Delta W x$，其中 $\alpha$ 是与 $r$ 成比例的常数。当使用 Adam 优化时，调节 $\alpha$ 与调节学习率基本相同。因此，我们简单地将 $\alpha$ 设置为我们尝试的第一个 $r$，且不进行微调。此缩放有助于减少在变化 $r$ 时重新调节超参数的需要。


这种高效的低秩方法大大减少了参数数量，使得在保持性能的同时可以进行高效微调。

## 更广泛的微调

LoRA 引入了一种更广泛的微调方法，允许我们仅训练预训练参数的一部分，而不需要积累梯度更新以使权重矩阵在适应过程中达到全秩。通过设置 LoRA 秩 \( r \) 等于预训练权重矩阵的秩，我们可以大致恢复完整微调的表现能力。随着可训练参数的增加，LoRA 训练逐渐逼近原始模型的训练结果，而其他基于适配器的方法则趋向于一个无法处理长输入的简单 MLP。

## 无额外推理延迟


LoRA 在推理过程中没有额外的延迟。我们可以显式计算并存储 $W = W_0 + BA$，并像往常一样执行推理。当需要切换任务时，我们可以通过减去 $BA$ 并添加不同的 $B'A'$ 来恢复 $W_0$，这是一个高效的操作，几乎没有内存开销。这确保了在推理过程中，不会引入比微调模型更多的延迟。

## 4.2 应用于 Transformer 的 LoRA

原则上，我们可以将 LoRA 应用于神经网络中的任何权重矩阵子集，以减少可训练参数的数量。在 Transformer 架构中，自注意力模块中的四个权重矩阵 \( W_q \), \( W_k \), \( W_v \), \( W_o \) 以及 MLP 模块中的两个矩阵被视为维度为 \( d_{\text{model}} \times d_{\text{model}} \) 的单一矩阵，尽管输出维度通常会被切割成注意力头。我们将研究仅限于**适配注意力权重**，冻结 MLP 模块（因此它们不会在下游任务中被训练），以简化操作并提高参数效率。我们进一步研究了不同类型的注意力权重矩阵在 Transformer 中的适配效果，详见 [Section 7.1]。至于适配 MLP 层、LayerNorm 层以及偏差权重的实证研究，则留待未来工作。

### 实际的优势与限制

最显著的优势来自于减少了内存和存储的使用。对于使用 Adam 优化器训练的大型 Transformer，VRAM 使用量可以减少最多 \( \frac{2}{3} \)，如果 \( r \ll d_{\text{model}} \)，因为我们无需存储被冻结参数的优化器状态。在 GPT-3 175B 模型上，我们将训练期间的 VRAM 消耗从 1.2TB 降低到 350GB。对于 \( r = 4 \)，并且仅适配查询和值投影矩阵，检查点大小减少了约 \( 10,000 \times \) （从 350GB 到 35MB）
。这使得我们可以使用显著更少的 GPU 进行训练，并避免 I/O 瓶颈。另一个好处是，我们可以通过仅交换 LoRA 权重而不是所有参数，在任务之间动态切换。这使得可以创建许多定制模型，这些模型可以在机器上即时进行切换，存储在 VRAM 中的预训练权重不会被影响。在 GPT-3 175B 的训练过程中，与全微调相比，LoRA 还观察到了 25% 的加速，因为我们不需要为大多数参数计算梯度。

LoRA 也有其局限性。例如，将输入批量化到不同的任务并非易事。如果在前向传递中选择将 \( A \) 和 \( B \) 吸收到 \( W \) 中以消除额外的推理延迟，这会有一定的限制。尽管可以选择不合并权重，并动态选择 LoRA 模块以在延迟不重要的场景中使用批量样本。


![alt text](image-3.png)

该表（表 4）展示了在三个任务上应用各种适应方法的 GPT-3 175B 的性能：WikiSQL、MultiNLI-matched（MNLI-m）和 SAMSum。主要度量指标是 WikiSQL 和 MNLI-m 上的验证准确率，以及 SAMSum 上的 Rouge-1/2/L 得分。

以下是结果的详细说明：

- **GPT-3 (FT)**（完全微调）是基准方法。它对整个 175B 参数进行训练，并在 WikiSQL 上达到 73.8% 的准确率，在 MNLI-m 上达到 89.5%，在 SAMSum 上的 Rouge-1/2/L 得分为 52.0/28.0/44.5。

- **GPT-3 (BitFit)** 仅训练了 14.2M 参数，在 MNLI-m 上表现相似（91.0%），但在 WikiSQL 上的准确率略低（71.3%）。Rouge 得分也比完全微调略低。

- **GPT-3 (PreEmbed)** 和 **GPT-3 (PreLayer)** 代表基于预训练嵌入和层的方法。这些方法在 WikiSQL 和 MNLI-m 上的表现相对较差，尤其是 PreEmbed 模型，与 LoRA 和完全微调相比得分显著较低。

- **GPT-3 (Adapter)** 方法相较于完全微调训练的参数较少。高秩适配器（AdapterH）在 WikiSQL 上达到 73.2% 的准确率，在 MNLI-m 上达到最高的 91.5% 准确率，并且在 SAMSum 上表现良好（53.2/29.0/45.1）。

- **GPT-3 (LoRA)**：LoRA（低秩适应）与其他方法相比表现特别出色。LoRA 仅训练了 4.7M 参数，在 WikiSQL 上达到接近完全微调的准确率（73.4%），在 MNLI-m（91.7%）和 SAMSum（53.8/29.8/45.9）上表现更好。对于 37.7M 参数的 LoRA 模型，WikiSQL 得分更高（74.0%）。

### 总结：
- LoRA 在大多数任务上都优于其他适应方法，甚至在某些任务上超过了完全微调的性能，同时训练的参数远远少于完全微调。
- LoRA 减少了大规模微调的需求，同时在测试任务上仍能达到较高的准确率和 Rouge 得分。
