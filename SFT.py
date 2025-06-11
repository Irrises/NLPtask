#%%
import os
import json
import re
from collections import defaultdict
import difflib

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, # 注意：使用 AutoModelForCausalLM
    TrainingArguments,    # 注意：使用 TrainingArguments
    Trainer,              # 注意：使用 Trainer
    DataCollatorForSeq2Seq, 
    EarlyStoppingCallback,
    BitsAndBytesConfig    # 用于量化加载
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from sklearn.model_selection import train_test_split
import numpy as np
# import evaluate # Hugging Face evaluate 库，如果需要标准指标如BLEU/ROUGE

# 检查是否有可用的GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
#%%
# --- 配置参数 ---
MODEL_NAME = "/root/autodl-tmp/models/Qwen3-8B" 


# LoRA 配置 (如果启用)
USE_LORA = True # 是否启用LoRA进行参数高效微调
LORA_R = 16 # LoRA的秩 (rank)
LORA_ALPHA = 32 # LoRA的alpha参数 (缩放因子)
LORA_DROPOUT = 0.05 # LoRA层的dropout率
# LoRA作用的目标模块，对于Qwen1.5/Qwen2模型，常见的模块包括 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
# 您可以通过 print(model) 查看模型结构来确定正确的模块名称。
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# 量化配置 (可选, 如果显存不足)
USE_QUANTIZATION = True # 是否使用4-bit/8-bit量化加载模型以节省显存
QUANTIZATION_TYPE = "nf4" # "nf4" (4-bit NormalFloat), "fp4" 或 "int8"

# 训练相关参数
OUTPUT_DIR = "/root/autodl-tmp/qwen_hate_speech_finetuned-1.7B" # 微调后模型的输出和保存目录
# 定义训练文件路径，请确保该文件存在于您的目录中
TRAIN_FILE_PATH = "./train_formatted_for_llm.jsonl" 

TRAIN_BATCH_SIZE = 5 # Causal LM通常需要更小的批次大小，根据您的GPU显存进行调整
EVAL_BATCH_SIZE = 2  # 评估时的批次大小
NUM_TRAIN_EPOCHS = 1 # 训练的总轮数，可根据收敛情况调整 (对于大模型和LoRA，较少轮数可能就够了)
LEARNING_RATE = 2e-4 # LoRA微调时学习率通常可以稍大一些
WEIGHT_DECAY = 0.01  # 权重衰减参数
MAX_INPUT_LENGTH = 1024 # 输入序列（包括提示和输出）的最大token长度
MAX_TARGET_LENGTH = 256 # 目标输出（四元组字符串）的最大token长度  
# MAX_TARGET_LENGTH 在Causal LM中不太直接使用，因为输入和输出合并了
GRADIENT_ACCUMULATION_STEPS = 4 # 梯度累积步骤，有效扩大批次大小
WARMUP_RATIO = 0.03 # 学习率预热比例
LR_SCHEDULER_TYPE = "cosine" # 学习率调度器类型

SEED = 42 # 随机种子，用于保证实验的可复现性

# 特殊标记定义
END_TOKEN = "[END]" # 单个四元组的结束标记
SEP_TOKEN = "[SEP]" # 多个四元_组之间的分隔标记
# 目标群体的所有类别，确保与您的任务描述和数据标注一致
TARGET_GROUPS = ["Region", "Racism", "Sexism", "LGBTQ", "others", "non-hate"] 
# 是否构成仇恨言论的两种状态
HATEFUL_STATUS = ["hate", "non-hate"]

# 定义用于Causal LM的提示模板结构
PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
你是一个专业的中文社交媒体内容分析助手，专门用于细粒度片段级仇恨言论识别。请根据用户提供的文本，识别其中存在的仇恨言论或非仇恨的评论性言论，并按照以下格式输出一个或多个四元组：
评论对象 (Target) | 论点 (Argument) | 目标群体 (Targeted Group) | 是否仇恨 (Hateful) [END]
详细说明：
1.  **评论对象 (Target)：** 帖子中被评论或提及的具体人物、群体、事物或概念。如果是针对文本中隐含的、没有明确指出的对象，或者评论是泛指，则设为 "NULL"。
2.  **论点 (Argument)：** 针对“评论对象”所发表的核心观点、描述或行为，应为文本中的一个关键信息片段。
3.  **目标群体 (Targeted Group)：** 指该“评论对象-论点”所涉及或指向的社会群体。重点注意：目标仇恨群体有可能不止一种，可能同时存在多种仇恨
    如：她们 | 支持她们找黑哥哥 | Sexism, Racism | hate [END]
    但总的群体必须从以下预设类别中选择：
    * `Region`：针对特定地域（国家、省份、城市等）人群的评论。
    * `Racism`：针对特定种族或民族人群的评论。
    * `Sexism`：针对特定性别人群（男性、女性）的评论，或性别歧视、刻板印象。
    * `LGBTQ`：针对性少数群体的评论（如同性恋、跨性别等）。
    * `others`：针对上述四类之外的特定群体（如特定职业、疾病群体、政治立场群体等）或不构成对特定社会群体的攻击，而是个人攻击、观点评论等。
    * `non-hate`：不存在攻击群体。
4.  **是否仇恨 (Hateful)：** 判断该“评论对象-论点”是否构成了对“目标群体”的仇恨言论。
    * `hate`：构成仇恨。
    * `non-hate`：不构成仇恨（包括中性、积极、或一般性负面评论但未达到仇恨程度）。
格式要求：
* 四元组内各元素之间用 " | "（空格竖杠空格）分隔。
* 每个四元组必须以 " [END]"（空格[END]）结尾。
* 如果一条评论中识别出多个独立的评论对象和论点，应输出多个四元组，不同四元组之间用 " [SEP] "（空格[SEP]空格）分隔。

现在，请处理以下新的输入内容：
<</SYS>>

用户提供的文本如下：
{input_text} [/INST]
模型输出：
"""
# 模型应该在此之后接上四元组字符串
#%%
def load_and_prepare_data(file_path, test_size=0.01, random_state=SEED):
    """
    加载 .jsonl 格式的数据文件，并将其划分为训练集和验证集。
    此函数专门适配包含 "messages" 列表的数据格式，从中提取 "user" 和 "assistant" 的内容。
    
    参数:
    - file_path (str): .jsonl 数据文件的路径。
    - test_size (float): 分配给验证集的比例。
    - random_state (int): 随机种子，用于可复现的划分。
    """
    input_texts_from_user = []      # 用于存储从 "user" 角色提取的输入文本
    target_quadruples_from_assistant = [] # 用于存储从 "assistant" 角色提取的目标四元组字符串
    system_prompts_from_data = []   # 可选：存储数据中提供的系统提示，供后续分析或使用
    
    # 检查文件是否存在，以避免后续错误
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 训练文件 '{file_path}' 未找到。请检查路径是否正确。")
        
    print(f"开始从 '{file_path}' 加载数据 (适配 'messages' 格式)...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1): # 从1开始计数行号，便于调试
            try:
                data_item = json.loads(line) # 解析当前行JSON数据
                
                if "messages" not in data_item or not isinstance(data_item["messages"], list):
                    print(f"警告: 跳过行 (行号 {line_num})，因为缺少 'messages' 键或其值不是列表: {line.strip()}")
                    continue

                messages_list = data_item["messages"]
                user_content = None
                assistant_content = None
                system_content_in_item = None # 当前条目中的系统提示

                for message_dict in messages_list:
                    if "role" in message_dict and "content" in message_dict:
                        if message_dict["role"] == "user":
                            user_content = message_dict["content"]
                        elif message_dict["role"] == "assistant":
                            assistant_content = message_dict["content"]
                        elif message_dict["role"] == "system":
                            system_content_in_item = message_dict["content"]
                    else:
                        print(f"警告: 跳过 'messages' 列表中的无效条目 (行号 {line_num})，缺少 'role' 或 'content': {message_dict}")
                
                # 确保成功提取了user和assistant的内容
                if user_content is not None and assistant_content is not None:
                    input_texts_from_user.append(user_content)
                    target_quadruples_from_assistant.append(assistant_content)
                    if system_content_in_item: # 如果当前数据条目中找到了system prompt
                        system_prompts_from_data.append(system_content_in_item)
                else:
                    print(f"警告: 跳过行 (行号 {line_num})，未能从 'messages' 中同时找到 'user' 和 'assistant' 的有效内容。")
                    if user_content is None:
                        print(f"  - 缺失 'user' 内容。")
                    if assistant_content is None:
                        print(f"  - 缺失 'assistant' 内容。")

            except json.JSONDecodeError:
                # 如果某行JSON格式错误，打印警告并跳过该行
                print(f"警告: 跳过无效的JSON行 (行号 {line_num}): {line.strip()}")
            except Exception as e: # 捕获其他潜在错误
                print(f"警告: 处理行 (行号 {line_num}) 时发生未知错误 '{e}': {line.strip()}")
                continue 
    
    # 检查是否成功加载了数据
    if not input_texts_from_user or not target_quadruples_from_assistant:
        raise ValueError(f"错误: 未能从 '{file_path}' 加载任何有效的 'user'/'assistant' 对话数据。请检查文件格式、内容和角色标签是否正确。")
    
    print(f"成功从 '{file_path}' 加载了 {len(input_texts_from_user)} 条有效的对话记录。")
    if system_prompts_from_data:
        print(f"（其中 {len(system_prompts_from_data)} 条记录包含系统提示）")
        # print(f"  数据中发现的第一个系统提示示例: '{system_prompts_from_data[0]}'") # 可选打印

    # 使用 sklearn 的 train_test_split 函数划分训练集和验证集
    print(f"正在将数据划分为训练集和验证集 (验证集比例: {test_size})...")
    # 注意：这里传递给Dataset的键名仍然是 "text" 和 "quadruples_str" 以便后续单元格代码兼容
    # input_texts_from_user 对应 "text"
    # target_quadruples_from_assistant 对应 "quadruples_str"
    train_texts, val_texts, train_quads, val_quads = train_test_split(
        input_texts_from_user, target_quadruples_from_assistant, 
        test_size=test_size, random_state=random_state
    )
    print(f"划分完成: 训练集 {len(train_texts)} 条, 验证集 {len(val_texts)} 条。")

    # 将划分后的数据转换为 Hugging Face Dataset 对象
    # 使用与之前代码兼容的键名 "text" 和 "quadruples_str"
    train_dataset = Dataset.from_dict({"text": train_texts, "quadruples_str": train_quads})
    val_dataset = Dataset.from_dict({"text": val_texts, "quadruples_str": val_quads})
    
    # 将训练集和验证集包装在 DatasetDict 中返回
    return DatasetDict({"train": train_dataset, "validation": val_dataset})
#%%
print(f"准备从文件 '{TRAIN_FILE_PATH}' 加载数据...")
try:
    raw_datasets = load_and_prepare_data(TRAIN_FILE_PATH)
    print("\n数据加载和初步划分成功:")
    print(raw_datasets) 
    
    if raw_datasets and 'train' in raw_datasets and len(raw_datasets['train']) > 0:
        print(f"\n训练集中的第一个样本示例:")
        print(f"  输入文本 (text): {raw_datasets['train'][0]['text']}")
        print(f"  目标标签 (quadruples_str): {raw_datasets['train'][0]['quadruples_str']}")
    else:
        print("\n警告: 加载后的 'raw_datasets' 为空或 'train' 部分不完整。请检查数据加载过程。")
except Exception as e:
    print(f"\n数据加载或准备过程中发生严重错误: {e}")
    # raise e
#%%
print(f"正在从 '{MODEL_NAME}' 加载Tokenizer...")
# trust_remote_code=True 对于某些模型（如Qwen）是必要的
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Tokenizer 加载完成。")

# Qwen tokenizer 可能没有默认的 pad_token。如果需要填充，通常将其设置为 eos_token。
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 
    print(f"Tokenizer的pad_token未设置，已将其设置为eos_token: '{tokenizer.eos_token}'")

# 对于Causal LM, 我们需要将输入和目标合并，并正确处理标签以仅计算目标部分的损失
def preprocess_function_causal(examples):
    """
    对批量数据进行tokenize和预处理，适配Causal LM。
    输入和输出将被合并，并创建标签以仅对输出部分计算损失。
    """
    full_prompts = []
    input_texts_for_prompt = examples["text"]
    target_outputs = examples["quadruples_str"]

    for input_text, target_output in zip(input_texts_for_prompt, target_outputs):
        # 构建包含指令、输入和预期输出的完整文本
        # 模型在推理时只会看到 PROMPT_TEMPLATE.format(input_text=input_text) 这部分
        # 训练时，我们将完整输出也加进去，并添加eos_token表示序列结束
        full_text = PROMPT_TEMPLATE.format(input_text=input_text) + target_output + tokenizer.eos_token
        full_prompts.append(full_text)

    # Tokenize 完整文本
    model_inputs = tokenizer(
        full_prompts,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False, # 先不填充，DataCollator会处理，或者可以设为 "max_length"
        return_attention_mask=True # 需要attention_mask
    )

    # 创建标签，初始时与input_ids相同
    labels = [list(ids) for ids in model_inputs["input_ids"]] # 深拷贝

    # 关键步骤：屏蔽掉提示部分的标签，使其在损失计算中被忽略 (设为-100)
    # 我们只希望模型学习预测 "模型输出："之后的内容
    for i in range(len(examples["text"])):
        prompt_only_text = PROMPT_TEMPLATE.format(input_text=examples["text"][i])
        # token_response_keyword = "模型输出：" # 定位输出开始的关键词
        # prompt_only_text_until_response = prompt_only_text.split(token_response_keyword)[0] + token_response_keyword
        
        # Tokenize不包含答案的提示部分，以确定需要屏蔽的长度
        # Qwen的tokenizer在tokenize提示时可能会有所不同，这里需要精确匹配
        # 更稳妥的方法是找到 "模型输出：" 之后token的起始位置
        
        # 找到 "模型输出：" 在完整提示中的位置，并获取其tokenize后的长度
        # 这里用一个近似方法：tokenize不包含答案的提示部分
        # 注意：Qwen tokenizer 对于 chat template 有特定处理，直接 format 可能不完全等同于 chat template 的tokenize结果
        # 但对于我们自定义的 PROMPT_TEMPLATE，这种方式是可行的。
        
        # 找到 "模型输出：" 在完整tokenize序列中的位置
        # 这是一个复杂点，因为 "模型输出：" 本身会被tokenize
        # 一个更简单的方法是，我们知道答案是从 PROMPT_TEMPLATE.format(...) 之后开始的
        
        temp_inputs_for_prompt_only = tokenizer(
            PROMPT_TEMPLATE.format(input_text=examples["text"][i]),
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
            add_special_tokens=False # 通常在构建完整序列时，首尾的特殊token由整体控制
                                     # 但Qwen tokenizer可能在内部添加，需要实验
        )
        prompt_length = len(temp_inputs_for_prompt_only["input_ids"])
        
        # 屏蔽提示部分的标签
        for j in range(prompt_length):
            if j < len(labels[i]): # 确保不越界
                 labels[i][j] = -100
            else: # 如果prompt_length超过了当前样本的总长度（可能因为截断），则停止
                 break
        
        # 确保 eos_token 不被屏蔽（如果它在答案的末尾）
        # 因为我们添加了 eos_token 到 target_output 之后，它应该在计算损失的范围内
        # 如果 tokenizer 自动在末尾添加 eos_token，且未包含在 target_output + eos_token 中，
        # 那么 labels[i] 的最后一个非-100元素之后直到序列末尾都应是-100（除了真正的eos_token）

    model_inputs["labels"] = labels
    return model_inputs
#%%
print("开始对数据集进行tokenize和预处理 (适配Causal LM)...")
if 'raw_datasets' not in locals() or not raw_datasets['train']: 
    print("错误: 'raw_datasets' 未定义或为空，无法进行tokenize。请先成功执行数据加载单元格。")
else:
    # 使用之前为Causal LM定义的预处理函数
    tokenized_datasets = raw_datasets.map(
        preprocess_function_causal, 
        batched=True, # 批处理以提高效率
        remove_columns=raw_datasets["train"].column_names 
    )
    print("\n数据tokenize和预处理完成:")
    print(tokenized_datasets) 

    if tokenized_datasets and 'train' in tokenized_datasets and len(tokenized_datasets['train']) > 0:
        print(f"\nTokenize后的训练集样本 (检查input_ids和labels的屏蔽情况):")
        sample_idx = 0
        print(f"  原始输入文本 (text): {raw_datasets['train'][sample_idx]['text']}")
        print(f"  原始目标输出 (quadruples_str): {raw_datasets['train'][sample_idx]['quadruples_str']}")
        
        print(f"\n  Tokenized input_ids (前60): {tokenized_datasets['train'][sample_idx]['input_ids'][:60]}")
        print(f"  Decoded input_ids (前60): {tokenizer.decode(tokenized_datasets['train'][sample_idx]['input_ids'][:60])}")
        
        print(f"\n  Tokenized labels (前60, -100表示已屏蔽): {tokenized_datasets['train'][sample_idx]['labels'][:60]}")
        # 找到第一个非-100的标签，解码该部分以验证
        first_label_idx = -1
        for idx, lbl_id in enumerate(tokenized_datasets['train'][sample_idx]['labels']):
            if lbl_id != -100:
                first_label_idx = idx
                break
        if first_label_idx != -1:
            print(f"  Decoded labels from first non-masked token (部分): {tokenizer.decode([l for l in tokenized_datasets['train'][sample_idx]['labels'][first_label_idx:first_label_idx+30] if l != -100])}")
        else:
            print("  注意：该样本的所有标签都被屏蔽了，可能存在问题或该样本答案部分被截断。")
            
        if 'attention_mask' in tokenized_datasets['train'][sample_idx]:
             print(f"\n  Attention_mask (前60): {tokenized_datasets['train'][sample_idx]['attention_mask'][:60]}")
    else:
        print("\n警告: Tokenize后的数据集为空或不完整。")
#%%
print(f"准备从 '{MODEL_NAME}' 加载预训练的Causal LM...")

# 量化配置 (如果启用)
bnb_config = None
if USE_QUANTIZATION:
    if QUANTIZATION_TYPE == "nf4" or QUANTIZATION_TYPE == "fp4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=QUANTIZATION_TYPE, # "nf4" 或 "fp4"
            bnb_4bit_compute_dtype=torch.bfloat16, # 计算时使用的类型，bfloat16 更稳定
            bnb_4bit_use_double_quant=True, # 双量化
        )
        print(f"使用4-bit量化 ({QUANTIZATION_TYPE}) 加载模型。")
    elif QUANTIZATION_TYPE == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("使用8-bit量化加载模型。")
    else:
        print(f"警告：不支持的量化类型 '{QUANTIZATION_TYPE}'，将不使用量化加载。")


# 加载预训练的Causal LM (如Qwen)
# trust_remote_code=True 对很多HF上的模型是必要的
# device_map="auto" 可用于多GPU或显存不足时自动分配模型层
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config if USE_QUANTIZATION else None,
    trust_remote_code=True,
    #device_map="auto" # 自动将模型分布到可用设备，对大模型友好
    # torch_dtype=torch.bfloat16 # 如果不量化，可以尝试用bfloat16加载以节省显存并加速 (在支持的GPU上)
)
print(f"模型 '{MODEL_NAME}' 加载完成。")

# 如果tokenizer的pad_token被设置为eos_token，模型的config中也最好同步
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    model.config.pad_token_id = model.config.eos_token_id
    print(f"模型配置的pad_token_id已设置为eos_token_id: {model.config.eos_token_id}")


if USE_LORA:
    print("\n启用LoRA进行参数高效微调。")
    # 如果使用了k-bit量化(4-bit/8-bit)，需要先准备模型
    if USE_QUANTIZATION:
        model = prepare_model_for_kbit_training(model)
        print("模型已为k-bit训练准备就绪 (LoRA适配)。")

    lora_config = LoraConfig(
        r=LORA_R, 
        lora_alpha=LORA_ALPHA, 
        target_modules=LORA_TARGET_MODULES, # 确保这些模块在您的Qwen模型中存在
        lora_dropout=LORA_DROPOUT, 
        bias="none", 
        task_type=TaskType.CAUSAL_LM # 任务类型设置为CAUSAL_LM
    )
    print("LoRA配置已创建:")
    print(lora_config)
    
    model = get_peft_model(model, lora_config) 
    print("\nLoRA适配器已应用到模型。")
    model.print_trainable_parameters() 
else:
    print("\n未启用LoRA，将进行全参数微调 (如果资源允许且未量化)。")

# 注意：如果使用了 device_map="auto"，模型可能已部分或全部在GPU上，无需再手动 .to(DEVICE)
# 但如果未使用 device_map 或希望确保在特定主设备，可以取消下面行的注释（但要小心与device_map冲突）
target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 或者直接使用之前定义的 DEVICE
model.to(target_device)
print(f"\n模型已手动移动到设备: {model.device}") # 现在应该显示 cuda:0

print(f"当前模型所在设备（通过model.device）: {model.device}")
if hasattr(model, 'hf_device_map'):
    print(f"模型层设备分布 (hf_device_map): {model.hf_device_map}")
#%%
# 解析和F1计算函数与之前Seq2Seq版本类似，因为它们处理的是文本字符串
def parse_quadruples(text_str):
    """
    将模型生成的单个目标字符串解析回结构化的四元组列表。
    """
    quadruples = [] 
    if not isinstance(text_str, str) or not text_str.strip(): 
        return [] 
        
    parts = text_str.split(SEP_TOKEN) 
    for part_idx, part in enumerate(parts):
        part_cleaned = part.strip() 
        
        if part_cleaned.endswith(END_TOKEN):
            part_cleaned = part_cleaned[:-len(END_TOKEN)].strip() 
        elif not part_cleaned and part_idx == len(parts) -1 : 
             continue

        if not part_cleaned: 
            continue
            
        elements = [e.strip() for e in part_cleaned.split("|")] 
        
        if len(elements) == 4:
            quadruples.append(elements)
        # else:
            # print(f"解析警告: 无法将部分 '{part_cleaned}' 解析为4个元素。实际得到 {len(elements)} 个元素: {elements}")
    return quadruples


def calculate_f1_metrics(preds_quads_list, labels_quads_list):
    """
    根据预测的四元组列表和真实的四元组列表，计算硬匹配和软匹配的F1分数。
    """
    true_positives_hard = 0
    predicted_positives_hard = 0 
    actual_positives_hard = 0    

    true_positives_soft = 0
    predicted_positives_soft = 0
    actual_positives_soft = 0

    for pred_quads_for_sample, gold_quads_for_sample in zip(preds_quads_list, labels_quads_list):
        predicted_positives_hard += len(pred_quads_for_sample)
        actual_positives_hard += len(gold_quads_for_sample)
        predicted_positives_soft += len(pred_quads_for_sample)
        actual_positives_soft += len(gold_quads_for_sample)

        matched_gold_indices_hard = set()
        for p_quad in pred_quads_for_sample:
            for i, g_quad in enumerate(gold_quads_for_sample):
                if i in matched_gold_indices_hard: 
                    continue
                if p_quad == g_quad: 
                    true_positives_hard += 1
                    matched_gold_indices_hard.add(i)
                    break 
        
        matched_gold_indices_soft = set()
        for p_quad in pred_quads_for_sample:
            if len(p_quad) != 4: continue
            for i, g_quad in enumerate(gold_quads_for_sample):
                if len(g_quad) != 4: continue 
                if i in matched_gold_indices_soft:
                    continue
                if p_quad[2] == g_quad[2] and p_quad[3] == g_quad[3]:
                    sim_target = difflib.SequenceMatcher(None, p_quad[0], g_quad[0]).ratio()
                    sim_argument = difflib.SequenceMatcher(None, p_quad[1], g_quad[1]).ratio()
                    if sim_target > 0.5 and sim_argument > 0.5: 
                        true_positives_soft += 1
                        matched_gold_indices_soft.add(i)
                        break 

    precision_hard = true_positives_hard / predicted_positives_hard if predicted_positives_hard > 0 else 0
    recall_hard = true_positives_hard / actual_positives_hard if actual_positives_hard > 0 else 0
    f1_hard = 2 * (precision_hard * recall_hard) / (precision_hard + recall_hard) if (precision_hard + recall_hard) > 0 else 0

    precision_soft = true_positives_soft / predicted_positives_soft if predicted_positives_soft > 0 else 0
    recall_soft = true_positives_soft / actual_positives_soft if actual_positives_soft > 0 else 0
    f1_soft = 2 * (precision_soft * recall_soft) / (precision_soft + recall_soft) if (precision_soft + recall_soft) > 0 else 0
    
    avg_f1 = (f1_hard + f1_soft) / 2

    return {
        "f1_hard": f1_hard, "precision_hard": precision_hard, "recall_hard": recall_hard,
        "f1_soft": f1_soft, "precision_soft": precision_soft, "recall_soft": recall_soft,
        "avg_f1": avg_f1
    }


def compute_metrics_causal(eval_preds):
    """
    Trainer在评估时调用的函数，用于计算Causal LM的自定义指标。
    eval_preds: 一个包含 predictions 和 label_ids 的元组。
                predictions 是模型生成（或logits），label_ids 是真实标签。
    """
    # predictions 是模型生成（或logits），label_ids 是真实标签。
    # 对于CausalLM，如果 Trainer 中没有特殊设置，predictions 可能是 logits。
    # 但如果使用了 generation_config 或类似设置，或者在 SFTTrainer 中，它可能是生成的 token ID。
    # 这里假设 predictions 是生成的 token ID 序列 (因为我们会在 TrainingArguments 中启用 generation)
    generated_token_ids, label_ids_from_input = eval_preds 
    
    # 将 label_ids 中的 -100 (用于在损失计算中忽略的填充token) 替换为 tokenizer 的 pad_token_id，以便正确解码
    processed_label_ids = np.where(label_ids_from_input != -100, label_ids_from_input, tokenizer.pad_token_id)
    
    # 解码生成的 token ID
    # skip_special_tokens=True 会移除解码结果中的特殊token
    # clean_up_tokenization_spaces=True 会清理tokenization过程中可能产生的额外空格
    decoded_preds_str = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # 解码真实的标签 ID (只解码答案部分)
    # 注意：对于Causal LM，label_ids_from_input 包含了整个输入序列（提示+答案）
    # 我们只关心答案部分的真实标签。
    # 而 decoded_preds_str 应该是模型仅生成的答案部分（不含提示）。
    # 所以，我们需要从原始数据中获取真实的“答案”字符串进行比较。
    
    # 这里有一个不匹配：decoded_preds_str 是模型生成的纯答案。
    # 但 processed_label_ids 解码后会包含提示+答案。
    # 我们需要真实的“答案”字符串，这在原始数据中是 quadruples_str。
    # Trainer 的 eval_dataset 通常不直接传递原始字符串。
    # 解决方案：在评估时，我们主要关心模型生成的质量。
    # 真实的 quadruples_str 需要从原始验证集中获取，这在 compute_metrics 中有点麻烦。
    
    # 简化的方法：假设 label_ids_from_input 只包含答案部分（如果数据加载时已处理）
    # 或者，更标准的方法是，模型生成时，我们只给它提示，它生成答案。
    # 然后，我们将生成的答案与原始数据中的 quadruples_str 比较。

    # 当前的 preprocess_function_causal 使 label_ids_from_input 对应完整序列，但提示部分是-100
    # 所以，解码 processed_label_ids 会得到 提示+答案。我们需要从中提取答案。
    
    decoded_labels_full_str = tokenizer.batch_decode(processed_label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # 从解码的完整标签中提取真实的答案部分
    # 这需要知道提示的结构，或者找到 "模型输出：" 之后的文本
    actual_target_strs = []
    for full_label_text in decoded_labels_full_str:
        # 找到 "模型输出：" 之后的内容
        # 注意：解码后的文本可能与原始提示不完全一致（由于tokenize和decode过程）
        # 最可靠的方式是使用原始未tokenize的 quadruples_str，但这不易在 compute_metrics 中直接获得
        # 这里我们尝试从解码后的完整标签中提取
        if "模型输出：" in full_label_text:
            actual_target_strs.append(full_label_text.split("模型输出：", 1)[-1].strip())
        else: # 如果关键词未找到，可能意味着这个样本的标签部分为空或被完全截断
            actual_target_strs.append("") 


    # 解析四元组
    pred_quads_list = [parse_quadruples(p_str) for p_str in decoded_preds_str]
    label_quads_list = [parse_quadruples(l_str) for l_str in actual_target_strs] # 使用提取的或原始的真实答案
    
    results = calculate_f1_metrics(pred_quads_list, label_quads_list)
    return results

print("评估指标相关函数 (parse_quadruples, calculate_f1_metrics, compute_metrics_causal) 已定义。")
#%%
import transformers # 导入 transformers 主模块以检查版本
import torch # 导入 torch 检查版本
# from transformers.trainer_utils import IntervalStrategy # 如果直接用 "steps" 字符串，则不需要导入这个

# 再次确认 Transformers 和 Torch 版本 (在 Notebook 单元格内执行，确保是内核使用的版本)
print(f"DEBUG: 当前Jupyter内核实际使用的 Hugging Face Transformers 版本: {transformers.__version__}")
print(f"DEBUG: 当前Jupyter内核实际使用的 Torch 版本: {torch.__version__}")

# --- 计算每个epoch的步数 ---
# 这个计算需要在 tokenized_datasets 加载完成之后 (通常在之前的单元格完成)
# 并在这里再次确认或计算，以确保 TrainingArguments 获得正确的值。

# 设置一个默认值，以防之前的计算步骤因某种原因未正确执行或变量丢失
CALCULATED_STEPS_PER_EPOCH = 500 

if 'tokenized_datasets' in locals() and \
   'train' in tokenized_datasets and \
   tokenized_datasets['train'] is not None and \
   len(tokenized_datasets['train']) > 0:
    
    # 确保 TRAIN_BATCH_SIZE 和 GRADIENT_ACCUMULATION_STEPS 是正整数
    if 'TRAIN_BATCH_SIZE' in globals() and TRAIN_BATCH_SIZE > 0 and \
       'GRADIENT_ACCUMULATION_STEPS' in globals() and GRADIENT_ACCUMULATION_STEPS > 0:
        
        # 在单GPU或未使用显式分布式训练（如DDP）时，world_size为1
        # effective_train_batch_size_per_step = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())
        # 为简化，并考虑到 device_map="auto" 的常见用法，我们假设并行处理由Trainer内部管理，此处基于单进程计算
        effective_train_batch_size_per_step = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        
        CALCULATED_STEPS_PER_EPOCH = len(tokenized_datasets["train"]) // effective_train_batch_size_per_step
        if CALCULATED_STEPS_PER_EPOCH == 0: 
            CALCULATED_STEPS_PER_EPOCH = 1 # 至少为1步，防止除零或无效值
        print(f"DEBUG: 根据计算，每个epoch大约有 {CALCULATED_STEPS_PER_EPOCH} 个更新步骤。")
    else:
        print(f"DEBUG: TRAIN_BATCH_SIZE 或 GRADIENT_ACCUMULATION_STEPS 未定义或无效，steps_per_epoch 使用默认值 {CALCULATED_STEPS_PER_EPOCH}。")
else:
    print(f"DEBUG: 'tokenized_datasets' 信息不足或训练集为空，steps_per_epoch 使用默认值 {CALCULATED_STEPS_PER_EPOCH}。")


print(f"DEBUG: 尝试初始化 TrainingArguments (使用最终确认的参数组合)...")
try:
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        # --- 核心配置：使用被证明在您环境中有效的参数名和策略 ---
        do_eval=True,                 # 启用评估
        eval_strategy="steps",        # 使用 "steps" 字符串作为策略名
        eval_steps=CALCULATED_STEPS_PER_EPOCH, # 设置评估步数
        
        save_strategy="steps",        # 使用 "steps" 字符串作为策略名
        save_steps=CALCULATED_STEPS_PER_EPOCH, # 设置保存步数，与评估步数一致
        # ----------------------------------------------------
        
        save_total_limit=2, # 最多保存的检查点数量
        
        logging_dir=f"{OUTPUT_DIR}/logs", # TensorBoard等日志的输出目录
        logging_strategy="steps", # 日志记录策略
        # 日志记录步数，例如每epoch记录10次，或至少每50步（如果epoch太短）
        logging_steps=max(1, CALCULATED_STEPS_PER_EPOCH // 10 if CALCULATED_STEPS_PER_EPOCH > 10 else 50),
        
        load_best_model_at_end=True, # 训练结束后加载在验证集上性能最佳的模型
        metric_for_best_model="avg_f1", # 用于选择最佳模型的指标名称 (应与compute_metrics返回的键匹配)
        greater_is_better=True,      # 上述指标是否越大越好
        
        # 混合精度训练配置 (根据您的 USE_QUANTIZATION 设置)
        fp16=(torch.cuda.is_available() and not USE_QUANTIZATION),
        bf16=(torch.cuda.is_bf16_supported() and not USE_QUANTIZATION),

        lr_scheduler_type=LR_SCHEDULER_TYPE, # 学习率调度器类型
        warmup_ratio=WARMUP_RATIO,           # 学习率预热的比例 (相对于总训练步数)
        
        report_to=["tensorboard"], # 将训练指标报告给哪些平台 (例如 "tensorboard", "wandb")
        seed=SEED,                 # 全局随机种子，保证可复现性
        
        optim="adamw_torch", # 使用的优化器 ("adamw_torch", "adamw_hf", "adafactor" 等)
        remove_unused_columns=True, # 是否自动移除数据集中模型forward方法不使用的列 (通常推荐True)
    )
    print(f"训练参数 (TrainingArguments) 配置完成。评估和保存策略均设置为 'steps'，每 {CALCULATED_STEPS_PER_EPOCH} 步执行一次。")

except Exception as e: 
    print(f"DEBUG: TrainingArguments 初始化时捕获到错误: {e}")
    # 如果这里仍然出错，请仔细检查错误信息和所有传入的参数值是否合理
    raise e


# 初始化数据整理器 (Data Collator)
# 这部分代码通常在 TrainingArguments 成功初始化后执行
if 'training_args' in locals() and training_args is not None:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model, 
        label_pad_token_id=-100, # 使用-100填充标签，以便在损失计算中被忽略
        pad_to_multiple_of=8 if (training_args.fp16 or training_args.bf16) else None # 对于fp16/bf16训练，填充到8的倍数可能提高效率
    )
    print("数据整理器 (DataCollatorForSeq2Seq) 初始化完成。")
else:
    print("DEBUG: training_args 未能成功初始化，跳过 DataCollator 初始化。")
#%%
# 初始化 Trainer
# 注意：这里使用的是 Trainer，而不是 Seq2SeqTrainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=tokenized_datasets["train"] if tokenized_datasets else None, 
    eval_dataset=tokenized_datasets["validation"] if tokenized_datasets else None, 
    tokenizer=tokenizer,                 
    data_collator=data_collator,         
    compute_metrics=compute_metrics_causal, # 使用为Causal LM调整的评估函数
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)] 
)
print("Trainer 初始化完成。")

if not tokenized_datasets or not tokenized_datasets["train"]:
    print("警告: 由于tokenized_datasets为空或不完整，Trainer可能没有正确设置训练集。请检查之前的步骤。")
#%%
print("即将开始模型训练...")
if trainer.train_dataset is None:
    print("错误: 训练数据集未设置，无法开始训练。请检查数据加载和预处理步骤。")
else:
    try:
        # 为了在评估时让 Trainer 使用 model.generate()，我们需要确保它知道这是一个生成任务。
        # 这通常通过 SFTTrainer 或在 TrainingArguments 中设置相关参数（如 generation_config）来完成。
        # 对于普通的 Trainer，compute_metrics 将接收模型的 logits 输出（或如果模型本身在forward中生成，则为生成结果）。
        # 为了确保 compute_metrics_causal 接收到生成的 token ID 而不是 logits，
        # 我们需要在 TrainingArguments 中设置一些与生成相关的参数，
        # 或者在 compute_metrics_causal 内部进行 model.generate() 调用（但这更复杂）。
        # 一个简单的方法是，如果模型本身是 PeftModel，它通常会正确传递调用给基础模型的 generate。
        # TrainingArguments 没有直接的 predict_with_generate，但 Trainer.evaluate 会尝试生成。
        # 我们需要在 compute_metrics_causal 内部确认 eval_preds[0] 是生成的 token ID。
        # Trainer 在调用 compute_metrics 前会进行 prediction_step，如果模型是生成式的，
        # 且 eval_dataset 提供了 input_ids（不含labels），它应该会调用 generate。
        # 我们的 tokenized_datasets["validation"] 包含 labels，Trainer 会用它来计算损失，
        # 并且如果模型是生成模型，也会生成预测。

        # 查看一下 Trainer 的 prediction_loop 逻辑，确保它为 Causal LM 生成文本。
        # Trainer 会在 evaluation_loop 中调用 prediction_loop。
        # prediction_loop 会调用 model(**inputs) 或 model.generate(**inputs, generation_config)
        # 如果 labels is not None，它会计算损失。
        # 它也会返回 logits 或生成的序列。
        # 如果是 AutoModelForCausalLM，并且有 generation_config，它会生成。
        
        # 确保模型有 generation_config
        if model.generation_config is None:
            from transformers import GenerationConfig
            model.generation_config = GenerationConfig.from_model_config(model.config)
            print("已为模型设置默认的GenerationConfig。")
        
        # Trainer 会使用 model.generation_config.max_length 等参数
        # 我们可以覆盖这些，例如在 TrainingArguments 中使用 generation_max_length
        # (但 TrainingArguments 没有这个参数，它是在 Seq2SeqTrainingArguments 中)
        # 所以，依赖于 model.generation_config，或者在 predict 时手动传入
        model.generation_config.max_new_tokens = MAX_TARGET_LENGTH # 控制生成答案的最大长度
        model.generation_config.num_beams = 3
        model.generation_config.early_stopping = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id # 确保pad_token_id正确
        model.generation_config.eos_token_id = tokenizer.eos_token_id

        print(f"模型评估时将使用以下生成配置: num_beams={model.generation_config.num_beams}, max_new_tokens={model.generation_config.max_new_tokens}")

        train_result = trainer.train()
        print("\n模型训练完成!")

        print("正在保存模型 (LoRA adapter)...")
        trainer.save_model(OUTPUT_DIR) # 对于PEFT，这通常只保存adapter
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"模型适配器和tokenizer已保存到 '{OUTPUT_DIR}'。")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics) 
        trainer.save_metrics("train", metrics) 
        trainer.save_state() 
        print("\n训练指标已记录和保存。")
        print(f"训练统计指标: {metrics}")

    except Exception as e:
        print(f"\n模型训练过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
#%%
# 加载训练好的模型进行预测/推理
# Trainer.model 应该是训练结束后性能最佳的模型 (如果 load_best_model_at_end=True)

# 如果需要手动加载 PEFT 模型：
# from peft import PeftModel
# print(f"正在从 '{MODEL_NAME}' 加载基础模型 (用于推理)...")
# base_model_for_inference = AutoModelForCausalLM.from_pretrained(
# MODEL_NAME,
# quantization_config=bnb_config if USE_QUANTIZATION else None, # 与训练时一致的量化
# trust_remote_code=True,
# device_map="auto" # 或者将其移至特定设备
# )
# if tokenizer.pad_token_id == tokenizer.eos_token_id: # 确保pad_token_id一致
# base_model_for_inference.config.pad_token_id = base_model_for_inference.config.eos_token_id

# print(f"正在从 '{OUTPUT_DIR}' 加载LoRA适配器...")
# model_to_predict = PeftModel.from_pretrained(base_model_for_inference, OUTPUT_DIR)
# print("LoRA适配器加载完成。")
# model_to_predict = model_to_predict.merge_and_unload() # 可选: 合并权重并卸载LoRA层，得到一个标准模型
# print("LoRA权重已合并 (如果执行了merge_and_unload)。")

# 这里我们直接使用 trainer.model (因为它应该是最好的，并且已经是PeftModel)
model_to_predict = trainer.model 
model_to_predict.eval() # 设置为评估模式
# 如果未使用 device_map="auto" 或者模型不在GPU上，需要手动移动
# if DEVICE.type == 'cuda' and not hasattr(model_to_predict, 'hf_device_map'):
# model_to_predict.to(DEVICE)
print(f"用于预测的模型已准备好，当前设备: {model_to_predict.device}")


def predict_quadruples_causal(text_list, model, tokenizer_pred):
    """
    使用微调后的Causal LM模型对一批文本进行预测。
    """
    generated_quadruples_str = []
    parsed_results_list = []

    for text_input in text_list:
        # 1. 构建不包含答案的提示
        prompt_for_inference = PROMPT_TEMPLATE.format(input_text=text_input)
        
        # 2. Tokenize提示
        # 对于Qwen等模型，tokenizer(..., add_special_tokens=True) 通常是推荐的
        # 但如果PROMPT_TEMPLATE已包含所有必要的特殊token (如<s>, </s>, [INST]), 则可能设为False
        # Qwen的Chat模型通常期望特定的对话格式，可以通过tokenizer.apply_chat_template处理
        # 但我们这里用的是自定义的PROMPT_TEMPLATE，所以直接tokenize
        inputs = tokenizer_pred(
            prompt_for_inference, 
            return_tensors="pt", 
            truncation=True, 
            max_length=MAX_INPUT_LENGTH - MAX_TARGET_LENGTH, # 给答案留出空间
            padding=False # 单个样本推理不需要padding
        ).to(model.device) # 将输入移动到模型所在设备

        # 3. 使用模型生成输出
        with torch.no_grad():
            # 设置生成参数
            generation_config = model.generation_config
            generation_config.max_new_tokens = 1024 # 控制生成答案的最大长度
            generation_config.num_beams = 3
            generation_config.early_stopping = True
            #generation_config.do_sample = True # 如果想要采样而不是beam search
            #generation_config.temperature = 0.1
            # generation_config.top_k = 50
            # generation_config.pad_token_id = tokenizer_pred.eos_token_id # 重要：用于beam search

            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # 4. 解码生成的token ID
        # outputs包含完整的序列 (提示+答案)，我们需要提取答案部分
        # generated_ids = outputs[0] # 对于batch_size=1
        # prompt_tokens_count = inputs.input_ids.shape[1]
        # answer_tokens = generated_ids[prompt_tokens_count:]
        
        # 更简单的方式：直接解码整个输出，然后通过字符串处理移除提示部分
        # 或者，如果tokenizer.decode能正确处理，它可能只解码新生成的部分
        # (取决于模型的generate实现和skip_special_tokens)
        full_generated_text = tokenizer_pred.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # 从完整生成文本中提取答案部分
        # 假设答案在 "模型输出：" 之后
        answer_part_str = ""
        if "模型输出：" in full_generated_text:
            answer_part_str = full_generated_text.split("模型输出：", 1)[-1].strip()
        else: # 如果模型没有生成 "模型输出："，则取最后一部分作为尝试
            # 这部分可能需要根据模型实际输出进行调整
            answer_part_str = full_generated_text.replace(prompt_for_inference.replace("{input_text}", text_input).split("模型输出：")[0]+"模型输出：", "").strip()


        generated_quadruples_str.append(answer_part_str)
        
        # 5. 解析生成的四元组字符串
        parsed_quads = parse_quadruples(answer_part_str)
        parsed_results_list.append({
            "original_text": text_input,
            "full_generated_text": full_generated_text, # 包含提示的完整输出，用于调试
            "extracted_answer_string": answer_part_str,
            "parsed_quadruples": parsed_quads
        })
        
    return parsed_results_list

print("预测/推理相关函数 (predict_quadruples_causal) 已定义。")
#%%
# 示例预测
sample_test_texts_for_prediction = [
    "那些同性恋真恶心，败坏社会风气。",
    "这道菜味道不错，下次还来。",
    "上海人就是排外，看不起外地人。",
    "黑人都是罪犯，应该被赶走。",
    "你可真是头蠢驴，这都做不好。",
    "我是支持的理中客和鉴权♂太多早该砸砸了还有那种乱开黄腔然后后面个狗头的低能"
]

print("\n开始运行示例预测...")
if 'model_to_predict' not in locals() or model_to_predict is None:
    print("错误: 'model_to_predict' 未定义。请确保模型已成功训练并加载。")
else:
    predictions = predict_quadruples_causal(sample_test_texts_for_prediction, model_to_predict, tokenizer)
    print("\n示例预测结果:")
    for item in predictions:
        print(f"原始文本 (Original Text): {item['original_text']}")
        # print(f"模型完整输出 (Full Generated Text): {item['full_generated_text']}") # 用于调试
        print(f"提取的答案字符串 (Extracted Answer): {item['extracted_answer_string']}")
        print(f"解析后的四元组 (Parsed Quadruples): {item['parsed_quadruples']}")
        print("-" * 30)
#%%
import json # 确保导入json库
import os   # 确保导入os库

# --- 如何加载官方测试数据并生成提交文件的示例 ---

def load_official_test_data(file_path):
    """
    加载官方测试数据。
    假设文件是一个JSON，其顶级结构是一个列表，列表中的每个元素是一个包含 "id" 和 "content" 键的字典。
    
    参数:
    - file_path (str): 测试数据JSON文件的路径。
    
    返回:
    - list: 包含所有 "content" 字符串的列表。
    - list: 包含所有对应 "id" 的列表 (可选, 如果需要id进行映射或调试)。
    """
    texts_to_predict = []
    ids_from_test_data = [] # 可选，用于追踪ID

    if not os.path.exists(file_path):
        print(f"错误: 测试文件 '{file_path}' 未找到。")
        return texts_to_predict, ids_from_test_data # 返回空列表

    print(f"正在从 '{file_path}' 加载官方测试数据...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # 整个文件是一个JSON列表
            if not isinstance(data, list):
                print(f"错误: 测试文件 '{file_path}' 的顶级结构不是一个列表。请检查文件格式。")
                return texts_to_predict, ids_from_test_data

            for item_num, item in enumerate(data, 1):
                if isinstance(item, dict) and "content" in item and "id" in item:
                    texts_to_predict.append(item["content"])
                    ids_from_test_data.append(item["id"])
                else:
                    print(f"警告: 测试文件 '{file_path}' 中的第 {item_num} 项格式不正确或缺少 'id'/'content' 键，已跳过: {item}")
        
        print(f"成功从 '{file_path}' 加载了 {len(texts_to_predict)} 条测试数据。")

    except json.JSONDecodeError:
        print(f"错误: 解析测试文件 '{file_path}' 时发生JSON解码错误。请检查文件是否为有效的JSON格式。")
    except Exception as e:
        print(f"加载测试文件 '{file_path}' 时发生其他错误: {e}")
        
    return texts_to_predict, ids_from_test_data

# --- 开始处理测试数据并生成提交文件 ---

# 确保 'model_to_predict' 和 'tokenizer' 已经从之前的单元格成功加载和设置
if 'model_to_predict' not in locals() or model_to_predict is None:
    print("错误: 'model_to_predict' 未定义。请确保模型已成功训练并赋值给此变量。")
elif 'tokenizer' not in locals() or tokenizer is None:
    print("错误: 'tokenizer' 未定义。请确保Tokenizer已成功加载。")
else:
    # 选择要处理的测试文件 (例如 test1.json 或 test2.json)
    # official_test_file_path = "/kaggle/input/nlptrain/test1.json" 
    official_test_file_path = "./test1.json" # 或者选择 test2.json

    if os.path.exists(official_test_file_path):
        print(f"\n开始处理官方测试文件: {official_test_file_path}")
        
        # 加载测试数据
        official_test_texts, official_test_ids = load_official_test_data(official_test_file_path)
        
        if official_test_texts:
            submission_outputs_strings = []
            # 为了提高效率，可以分批处理官方测试数据
            # 推理时的批次大小，根据您的显存和模型大小调整
            # (应与单元格2中的 EVAL_BATCH_SIZE 或一个适合推理的值一致)
            inference_batch_size = EVAL_BATCH_SIZE 

            print(f"开始对 {len(official_test_texts)} 条测试数据进行预测 (批次大小: {inference_batch_size})...")
            for i in range(0, len(official_test_texts), inference_batch_size):
                batch_texts = official_test_texts[i : i + inference_batch_size]
                
                current_batch_num = (i // inference_batch_size) + 1
                total_batches = (len(official_test_texts) + inference_batch_size - 1) // inference_batch_size
                print(f"  正在预测批次 {current_batch_num} / {total_batches}...")
                
                # 调用您在单元格12中定义的预测函数
                # predict_quadruples_causal 函数返回一个字典列表，
                # 每个字典包含 'original_text', 'full_generated_text', 'extracted_answer_string', 'parsed_quadruples'
                batch_predictions = predict_quadruples_causal(batch_texts, model_to_predict, tokenizer)
                
                for item_prediction in batch_predictions:
                    # 我们需要的是模型生成的、仅包含四元组格式的字符串
                    submission_outputs_strings.append(item_prediction['extracted_answer_string'])
            
            # 将预测结果按demo.txt的格式保存到 submission.txt 文件
            submission_file_path = "./submission2.txt" # Kaggle工作目录
            try:
                with open(submission_file_path, "w", encoding="utf-8") as f:
                    for line_num, line_content in enumerate(submission_outputs_strings):
                        f.write(line_content + "\n")
                print(f"\n提交文件已成功生成: {submission_file_path}")
                print(f"该文件包含 {len(submission_outputs_strings)} 行预测。")
                print("请检查文件内容是否符合demo.txt的格式。")
            except Exception as e:
                print(f"写入提交文件 '{submission_file_path}' 时发生错误: {e}")
        else:
            print(f"未能从 '{official_test_file_path}' 加载任何测试数据进行预测。")
    else:
        print(f"测试文件路径 '{official_test_file_path}' 不存在。跳过提交文件生成。")
#%%
