# 基于大模型微调与对比上下文学习的细粒度中文仇恨识别

本仓库为NLP大作业 **《基于大模型微调与对比上下文学习的细粒度中文仇恨识别方法》** 的代码。

该项目旨在完成 **CCL25-Eval 任务10：细粒度中文仇恨识别评测**，通过利用大语言模型（LLM）从中文文本中识别并结构化地提取仇恨言论的各个要素。

---

## 📖 目录

* [项目概述](#-项目概述)
* [核心方法](#-核心方法)
* [项目结构](#-项目结构)
* [环境配置与依赖](#-环境配置与依赖)
* [使用流程](#-使用流程)

---

##  项目概述

随着社交媒体的日益普及，网络仇恨言论的识别与治理已成为自然语言处理领域的一项关键挑战。本方法详述了针对“CCL25-Eval 任务10：细粒度中文仇恨识别评测”的技术方案，深入阐述构建思路、技术细节及理论基础。鉴于任务的复杂性及结构化输出要求，本方法使用了大语言模型（Large Language Model, LLM）作为基础架构。初步零样本（Zero-Shot）实验表明，LLM直接处理此任务效果欠佳。为此，本方法使用QLoRA（Quantized Low-Rank Adapter）对LLM进行参数高效微调，提升基础任务能力，同时在推理阶段设计对比式上下文学习策略：利用LLM零样本生成存在偏差的“伪标签”作为负面对比例，构建包含“原始文本-负面示例-修正指令”的提示模板，迫使模型在推理时通过对比机制辨别正确答案与错误答案的细微差异。实验表明，该方法提升了模型对仇恨言论四元组的结构化提取性能与错误答案鲁棒性，评价分数从零样本生成的0.2736提升至0.3410。

---

##  核心方法

技术路径主要包含以下两个阶段：

### 1. 基于 QLoRA 的监督微调 (SFT)

为了高效地使大语言模型适配细粒度仇恨言论识别任务，我们采用了 **QLoRA (Quantized Low-Rank Adaptation)** 技术。QLoRA 通过将基础模型量化为 4-bit 并仅训练少量低秩适配器（Adapter）的方式，极大地降低了微调过程中的显存消耗。此步骤旨在让模型掌握任务的基础能力，理解并生成所需格式的四元组。



### 2. 用于推理的对比式上下文学习

本方法的核心创新在于推理阶段的策略。在尝试微调策略后，我们利用模型本身为给定的文本生成一个“伪标签”。这个初步的预测结果通常接近正确答案但包含一些细微的偏差，因此是一个完美的 **负面示例**。

接着，我们构建一个包含原始文本、有偏的负面示例以及修正指令的特定提示（Prompt）。

**提示词模板 (Prompt Template):**

```
new_input_for_prompt = (
    f"原始文本内容：\n\"{original_text_content}\"\n\n"
    f"一个AI助手针对以上文本给出了如下可能是错误或不完善的四元组提取结果：\n"
    f"\"{negative_pseudo_quad_str}\"\n\n"
    f"请你忽略上述AI助手的提取结果（它可能包含错误），并严格按照指令，根据“原始文本内容”重新分析并给出正确的四元组。"
)
```

这种对比机制迫使模型进行更深层次的思考，通过比较正确与错误示例的差异，识别并修正那些细微的错误，从而产出更精准、更鲁棒的最终预测。含有负面示例的生成过程由 `Aug.py` 和 `Aug.ipynb` 脚本实现。 

---

## 项目结构

NLPtask/ 

├── model/                      

│   └── aug/ # 这里存放的是我训练好的模型适配器和分词器文件 

│       ├── adapter_config.json 

│       ├── merges.txt 

│       ├── special_tokens_map.json 

│       ├── tokenizer_config.json 

│       ├── vocab.json 

│      

 └── Qwen3-8B #下载模型位置


├── Aug.ipynb                   # 对比上下文学习的 Notebook 

├── Aug.py                      # 对比上下文学习的脚本 

├── postprocessing.py           # 用于后处理、清洗和格式化模型输出的脚本 

├── README.md                   # 本说明文件



---

 ##  环境配置与依赖

1. 克隆本仓库：    ```bash    git clone <https://github.com/Irrises/NLPtask.git>    cd NLPtask    ```

2. 在终端中安装所需的 Python 依赖包。基于 QLoRA 方法，主要依赖项如下：  

     ```pip install torch transformers peft bitsandbytes accelerate sentencepiece hugging face   ```

3. 下载模型到本地部署

     ```huggingface-cli download --resume-download Qwen/Qwen3-8B --local-dir ./model/Qwen3-8B --local-dir-use-symlinks False  ```

   

   本次实验在以下硬件平台上进行：

   平台：AutoDL算力平台

   GPU: RTX 4090 * 1（24GB）

   CPU型号: 16 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz

   内存大小: 120GB

   主要框架及库: PyTorch 2.5.1、Python 3.12(ubuntu22.04)、CUDA 12.4

   ---

   

   ##  使用流程

   按要求设置好路径以及修改文件中的保存目录后，就可以直接运行py脚本
   或是在服务器中利用jupyter notebook运行ipynb文件便于分析和调试

### 注意:
模型路径需与本地实际路径保持一致。
