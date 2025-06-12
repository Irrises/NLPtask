#%%
# --- 单元格 1: 环境设置和库导入 🛠️ ---


# 在Notebook内部验证和提示bitsandbytes的安装
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes 版本: {bnb.__version__} 已成功导入。")
except ImportError:
    print("错误: bitsandbytes 未安装或导入失败。")
    print("请尝试在新的单元格中运行: !pip install -U bitsandbytes")
    print("或者，如果您使用的是特定CUDA版本，可能需要查找特定的bitsandbytes安装命令。")
    print("安装后务必重启Jupyter Kernel！")
    raise

import os
import json
import re
from collections import defaultdict
import difflib
import gc 
from tqdm import tqdm 

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq, 
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from sklearn.model_selection import train_test_split
import numpy as np

# 检查可用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    # 设置 PYTORCH_CUDA_ALLOC_CONF 来减少显存碎片
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
#%%
# --- 单元格 2: 配置参数 ⚙️ ---
MODEL_NAME = "/root/autodl-tmp/models/Qwen3-1.7B" 
# LoRA 配置
USE_LORA = True 
LORA_R = 16 
LORA_ALPHA = 32 
LORA_DROPOUT = 0.05 
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 量化配置 
USE_QUANTIZATION = True 
QUANTIZATION_TYPE = "nf4" 

# 训练相关参数 (因为经常OOM所以进行保守设置)
OUTPUT_DIR = "/root/autodl-tmp/qwen_hate_speech_finetuned_llm_aug" # 输出目录名
TRAIN_FILE_PATH = "./train_formatted_for_llm.jsonl" 

TRAIN_BATCH_SIZE = 2 # 非常小的批次大小以避免OOM
EVAL_BATCH_SIZE = 3  
NUM_TRAIN_EPOCHS = 3 
LEARNING_RATE = 2e-4 
WEIGHT_DECAY = 0.01  
MAX_INPUT_LENGTH = 1024 # 
MAX_TARGET_LENGTH = 256 # 生成目标（四元组字符串）的最大token长度
GRADIENT_ACCUMULATION_STEPS = 8 # 增大梯度累积以补偿小批次大小
WARMUP_RATIO = 0.03 
LR_SCHEDULER_TYPE = "cosine" 

SEED = 42 

# 特殊标记定义
END_TOKEN = "[END]" 
SEP_TOKEN = "[SEP]" 
TARGET_GROUPS = ["Region", "Racism", "Sexism", "LGBTQ", "others", "non-hate"] 
HATEFUL_STATUS = ["hate", "non-hate"]

# 定义提示模板结构 
PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
你是一个专业的中文社交媒体内容分析助手，专门用于细粒度片段级仇恨言论识别。请根据用户提供的文本，识别其中存在的仇恨言论或非仇恨的评论性言论，并按照以下格式输出一个或多个四元组：
评论对象 (Target) | 论点 (Argument) | 目标群体 (Targeted Group) | 是否仇恨 (Hateful) [END]
详细说明：
1.  **评论对象 (Target)：** 帖子中被评论或提及的具体人物、群体、事物或概念。如果是针对文本中隐含的、没有明确指出的对象，或者评论是泛指，则设为 "NULL"。
2.  **论点 (Argument)：** 针对“评论对象”所发表的核心观点、描述或行为，应为文本中的一个关键信息片段。
3.  **目标群体 (Targeted Group)：** 指该“评论对象-论点”所涉及或指向的社会群体。其中，目标群体可以有多项，但必须从以下预设类别中选择：
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
#%%
# --- 单元格 3: 数据加载与准备函数  📄 ---
def load_and_prepare_data(file_path, test_size=0.1, random_state=SEED):
    input_texts_from_user = []      
    target_quadruples_from_assistant = [] 
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 训练文件 '{file_path}' 未找到。请检查路径是否正确。")
        
    print(f"开始从 '{file_path}' 加载数据 (适配 'messages' 格式)...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1): 
            try:
                data_item = json.loads(line) 
                
                if "messages" not in data_item or not isinstance(data_item["messages"], list):
                    # print(f"警告: 跳过行 (行号 {line_num})，因为缺少 'messages' 键或其值不是列表: {line.strip()}")
                    continue

                messages_list = data_item["messages"]
                user_content = None
                assistant_content = None
                
                for message_dict in messages_list:
                    if "role" in message_dict and "content" in message_dict:
                        if message_dict["role"] == "user":
                            user_content = message_dict["content"]
                        elif message_dict["role"] == "assistant":
                            assistant_content = message_dict["content"]
                
                if user_content is not None and assistant_content is not None:
                    input_texts_from_user.append(user_content)
                    target_quadruples_from_assistant.append(assistant_content)
                # else:
                    # print(f"警告: 跳过行 (行号 {line_num})，未能从 'messages' 中同时找到 'user' 和 'assistant' 的有效内容。")

            except json.JSONDecodeError:
                # print(f"警告: 跳过无效的JSON行 (行号 {line_num}): {line.strip()}")
                pass 
            except Exception: 
                # print(f"警告: 处理行 (行号 {line_num}) 时发生未知错误 '{e}': {line.strip()}")
                pass 
    
    if not input_texts_from_user or not target_quadruples_from_assistant:
        raise ValueError(f"错误: 未能从 '{file_path}' 加载任何有效的 'user'/'assistant' 对话数据。")
    
    print(f"成功从 '{file_path}' 加载了 {len(input_texts_from_user)} 条有效的对话记录。")

    print(f"正在将数据划分为训练集和验证集 (验证集比例: {test_size})...")
    train_texts, val_texts, train_quads, val_quads = train_test_split(
        input_texts_from_user, target_quadruples_from_assistant, 
        test_size=test_size, random_state=random_state
    )
    print(f"划分完成: 训练集 {len(train_texts)} 条, 验证集 {len(val_texts)} 条。")

    train_dataset = Dataset.from_dict({"text": train_texts, "quadruples_str": train_quads})
    val_dataset = Dataset.from_dict({"text": val_texts, "quadruples_str": val_quads})
    
    return DatasetDict({"train": train_dataset, "validation": val_dataset})
#%%
# --- 单元格 4: 加载并检查原始数据 🧐 ---
print(f"准备从文件 '{TRAIN_FILE_PATH}' 加载数据...")
raw_datasets = None # 初始化
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
#%%
# --- 单元格 3.5 (新增): 利用LLM零样本能力生成伪标签数据 🤖🏷️ ---
# --- 配置用于生成伪标签的LLM ---
GENERATOR_MODEL_NAME_FOR_PSEUDO = "/root/autodl-tmp/models/Qwen3-8B" # 示例：使用与微调相同的模型路径，或另一个更强的模型
GENERATOR_USE_QUANTIZATION_FOR_PSEUDO = True 
GENERATOR_QUANTIZATION_TYPE_FOR_PSEUDO = "nf4"

# 用于生成伪标签的提示模板 (与微调的PROMPT_TEMPLATE类似，但不包含 "模型输出：" 后的答案部分)
# 注意：这里的 GENERATOR_PROMPT_TEMPLATE 与主 PROMPT_TEMPLATE 几乎一致，
# 确保 "模型输出：" 之后是空的，以便LLM填充。
GENERATOR_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
你是一个专业的中文社交媒体内容分析助手，专门用于细粒度片段级仇恨言论识别。请根据用户提供的文本，识别其中存在的仇恨言论或非仇恨的评论性言论，并按照以下格式输出一个或多个四元组：
评论对象 (Target) | 论点 (Argument) | 目标群体 (Targeted Group) | 是否仇恨 (Hateful) [END]
详细说明：
1.  **评论对象 (Target)：** 帖子中被评论或提及的具体人物、群体、事物或概念。如果是针对文本中隐含的、没有明确指出的对象，或者评论是泛指，则设为 "NULL"。
2.  **论点 (Argument)：** 针对“评论对象”所发表的核心观点、描述或行为，应为文本中的一个关键信息片段。
3.  **目标群体 (Targeted Group)：** 指该“评论对象-论点”所涉及或指向的社会群体。必须从以下预设类别中选择：
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

pseudo_labels_list = []
texts_for_pseudo_generation = []

if raw_datasets and 'train' in raw_datasets and raw_datasets['train'] is not None:
    texts_for_pseudo_generation = list(raw_datasets['train']['text'])
    print(f"准备为 {len(texts_for_pseudo_generation)} 条训练文本生成伪标签...")

    # --- 加载生成器LLM和Tokenizer ---
    # 为避免与主模型冲突，使用不同的变量名
    generator_model_instance = None
    generator_tokenizer_instance = None
    print(f"正在从 '{GENERATOR_MODEL_NAME_FOR_PSEUDO}' 加载用于生成伪标签的LLM和Tokenizer...")
    try:
        generator_tokenizer_instance = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME_FOR_PSEUDO, trust_remote_code=True)
        
        generator_bnb_config = None
        if GENERATOR_USE_QUANTIZATION_FOR_PSEUDO:
            if GENERATOR_QUANTIZATION_TYPE_FOR_PSEUDO == "nf4" or GENERATOR_QUANTIZATION_TYPE_FOR_PSEUDO == "fp4":
                generator_bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type=GENERATOR_QUANTIZATION_TYPE_FOR_PSEUDO,
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                )
            elif GENERATOR_QUANTIZATION_TYPE_FOR_PSEUDO == "int8":
                generator_bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            print(f"生成器LLM将使用量化: {GENERATOR_QUANTIZATION_TYPE_FOR_PSEUDO if generator_bnb_config else '无'}")

        generator_model_instance = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL_NAME_FOR_PSEUDO,
            quantization_config=generator_bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
        generator_model_instance.eval() 

        if generator_tokenizer_instance.pad_token is None:
            generator_tokenizer_instance.pad_token = generator_tokenizer_instance.eos_token
            if generator_model_instance.config.pad_token_id is None:
                 generator_model_instance.config.pad_token_id = generator_tokenizer_instance.pad_token_id
            print(f"生成器Tokenizer的pad_token已设置为eos_token: '{generator_tokenizer_instance.eos_token}'")
        
        print("生成器LLM和Tokenizer加载成功。")

        GENERATION_BATCH_SIZE = 4 # 伪标签生成批次大小
        
        generation_config_pseudo = GenerationConfig(
            max_new_tokens=MAX_TARGET_LENGTH, 
            num_beams=1, 
            do_sample=False, 
            pad_token_id=generator_tokenizer_instance.pad_token_id if generator_tokenizer_instance.pad_token_id is not None else generator_tokenizer_instance.eos_token_id,
            eos_token_id=generator_tokenizer_instance.eos_token_id
        )

        for i in tqdm(range(0, len(texts_for_pseudo_generation), GENERATION_BATCH_SIZE), desc="生成伪标签"):
            batch_texts = texts_for_pseudo_generation[i : i + GENERATION_BATCH_SIZE]
            batch_prompts = [GENERATOR_PROMPT_TEMPLATE.format(input_text=text) for text in batch_texts]
            
            inputs = generator_tokenizer_instance(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=MAX_INPUT_LENGTH - MAX_TARGET_LENGTH 
            ).to(generator_model_instance.device)

            with torch.no_grad():
                outputs = generator_model_instance.generate(**inputs, generation_config=generation_config_pseudo)
            
            # 解码并提取答案部分
            # outputs 包含完整的序列 (提示+答案)。我们需要提取模型生成的部分。
            # input_len = inputs.input_ids.shape[1]
            # generated_ids_batch = outputs[:, input_len:] # 获取每个样本新生成的token
            # decoded_answers = generator_tokenizer_instance.batch_decode(generated_ids_batch, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # 或者，使用之前的方法从完整解码文本中分割
            full_decoded_outputs = generator_tokenizer_instance.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            keyword_separator_pseudo = "模型输出：" # 与GENERATOR_PROMPT_TEMPLATE末尾一致
            
            for full_output_text in full_decoded_outputs:
                answer_part_str = ""
                if keyword_separator_pseudo in full_output_text:
                    answer_part_str = full_output_text.split(keyword_separator_pseudo, 1)[-1].strip()
                else: # 后备方案
                    original_prompt_text_no_answer = GENERATOR_PROMPT_TEMPLATE.format(input_text="DUMMY").split(keyword_separator_pseudo)[0] # 获取提示头
                    # 这是一个粗略的移除，可能不完美
                    if full_output_text.startswith(original_prompt_text_no_answer.split("用户提供的文本如下：")[0]): # 尝试匹配系统提示部分
                         answer_part_str = full_output_text # 如果无法清晰分割，保留完整输出，后续质量评估时处理
                    else:
                         answer_part_str = full_output_text
                pseudo_labels_list.append(answer_part_str)
        
        print(f"成功为 {len(pseudo_labels_list)} 条文本生成了伪标签。")

    except Exception as e:
        print(f"加载生成器LLM或生成伪标签过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("将使用空的伪标签列表。")
        pseudo_labels_list = [] 
    finally:
        # 清理生成器模型以释放显存
        if 'generator_model_instance' in locals() and generator_model_instance is not None:
            del generator_model_instance
        if 'generator_tokenizer_instance' in locals() and generator_tokenizer_instance is not None:
            del generator_tokenizer_instance
        if 'inputs' in locals() and inputs is not None: del inputs
        if 'outputs' in locals() and outputs is not None: del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("生成器LLM及相关资源已尝试清理。")
else:
    print("警告: 原始数据集 'raw_datasets' 未加载，无法生成伪标签。")
    pseudo_labels_list = []

if pseudo_labels_list:
    print("\n生成的一些伪标签样本:")
    for i in range(min(3, len(pseudo_labels_list))):
        print(f"  原始文本 (部分): {texts_for_pseudo_generation[i][:50]}...")
        print(f"  生成伪标签: {pseudo_labels_list[i]}")
else:
    print("\n未能生成或加载任何伪标签。")
#%%
# --- 单元格 3.6: 利用LLM生成的负例进行对比增强SFT数据 ---

# 是否启用基于LLM负例的对比增强
ENABLE_CONTRASTIVE_AUGMENTATION_WITH_NEGATIVES = True

# 确保 parse_quadruples 函数已定义 (通常在单元格8)
# 如果单元格8的还未执行, 使用临时占位符 (主要用于结构完整性, 实际运行前应确保已定义)
if 'parse_quadruples' not in globals(): 
    def parse_quadruples_placeholder(text_str_dummy): 
        if not text_str_dummy: return []
        quads = []
        parts = text_str_dummy.split(SEP_TOKEN if 'SEP_TOKEN' in globals() else "[SEP]") # 使用全局变量或默认值
        for part in parts:
            part_c = part.strip()
            if part_c.endswith(END_TOKEN if 'END_TOKEN' in globals() else "[END]"):
                part_c = part_c[:-len(END_TOKEN if 'END_TOKEN' in globals() else "[END]")].strip()
            if not part_c: continue
            elements = [e.strip() for e in part_c.split(" | ")]
            if len(elements) == 4:
                quads.append(elements)
        return quads
    parse_quadruples_fn_to_use = parse_quadruples_placeholder
    print("警告：单元格8的 'parse_quadruples' 函数定义先于此单元格执行。将使用临时占位符。")
else:
    parse_quadruples_fn_to_use = parse_quadruples


if ENABLE_CONTRASTIVE_AUGMENTATION_WITH_NEGATIVES and \
   'raw_datasets' in locals() and raw_datasets and \
   'pseudo_labels_list' in locals() and \
   len(pseudo_labels_list) == len(raw_datasets['train']): # 确保伪标签列表与训练数据对齐

    print(f"开始基于LLM生成的伪标签（作为负例）进行对比数据增强...")
    
    original_train_texts = list(raw_datasets['train']['text'])
    original_train_quads = list(raw_datasets['train']['quadruples_str']) # 真实标签 (正例)
    
    # pseudo_labels_list 包含的是LLM生成的伪标签 (将被视为负例或干扰项)
    
    contrastive_augmented_texts = []
    contrastive_augmented_quads = [] # 目标输出始终是真实的四元组
    
    num_augmented_samples_created = 0

    for i in tqdm(range(len(original_train_texts)), desc="创建对比增强SFT数据"):
        original_text_content = original_train_texts[i]
        true_quad_str = original_train_quads[i]         # 正例输出
        negative_pseudo_quad_str = pseudo_labels_list[i] # LLM生成的，作为负例/干扰项

        # 1. 添加标准的SFT样本： (原始提示 -> 真实四元组)
        #    PROMPT_TEMPLATE 中的 {input_text} 将直接使用 original_text_content
        contrastive_augmented_texts.append(original_text_content) 
        contrastive_augmented_quads.append(true_quad_str)
        
        # 2. 创建对比增强的SFT样本：
        #    (包含负例的复杂提示 -> 真实四元组)
        #    只有当伪标签与真实标签确实不同时，这种增强才有意义
        if negative_pseudo_quad_str and negative_pseudo_quad_str.strip() and \
           negative_pseudo_quad_str.strip() != true_quad_str.strip():
            
            # 构建包含原始文本和“错误提案”（负例）的新输入文本
            # 这个 new_input_for_prompt 会被填入主 PROMPT_TEMPLATE 的 {input_text} 占位符
            new_input_for_prompt = (
                f"原始文本内容：\n\"{original_text_content}\"\n\n"
                f"一个AI助手针对以上文本给出了如下可能是错误或不完善的四元组提取结果：\n"
                f"\"{negative_pseudo_quad_str}\"\n\n"
                f"请你忽略上述AI助手的提取结果（它可能包含错误），并严格按照指令，根据“原始文本内容”重新分析并给出正确的四元组。"
            )
            
            contrastive_augmented_texts.append(new_input_for_prompt)
            contrastive_augmented_quads.append(true_quad_str) # 目标仍然是真实的四元组
            num_augmented_samples_created += 1

    print(f"对比数据增强完成。")
    print(f"原始训练样本数: {len(original_train_texts)}")
    print(f"额外创建了 {num_augmented_samples_created} 个对比增强样本。")
    
    if num_augmented_samples_created > 0 or len(contrastive_augmented_texts) != len(original_train_texts) :
        contrastive_augmented_train_dataset = Dataset.from_dict({
            "text": contrastive_augmented_texts, # "text" 字段现在包含原始文本或增强后的复杂提示
            "quadruples_str": contrastive_augmented_quads # 目标始终是真实的四元组
        })
        
        # 更新 raw_datasets 中的训练集
        # 验证集保持不变，用于评估原始任务性能
        raw_datasets['train'] = contrastive_augmented_train_dataset
        print(f"对比增强SFT数据准备完成。训练集现在包含 {len(raw_datasets['train'])} 条样本。")
        if len(raw_datasets['train']) > 0:
            print(f"增强后训练集的第一个样本 text (可能为原始): {raw_datasets['train'][0]['text'][:150]}...") 
            print(f"增强后训练集的第一个样本 quadruple: {raw_datasets['train'][0]['quadruples_str']}")
            if len(raw_datasets['train']) > len(original_train_texts): # 如果确实添加了增强样本
                 print(f"一个对比增强样本的 text (部分): {raw_datasets['train'][-1]['text'][:250]}...") # 打印最后一个（可能是增强的）
                 print(f"该增强样本的目标 quadruple: {raw_datasets['train'][-1]['quadruples_str']}")
    else:
        print("没有新的对比增强样本被添加到训练集（可能因为所有伪标签都与真实标签相同，或者伪标签为空）。")

else:
    if not ENABLE_CONTRASTIVE_AUGMENTATION_WITH_NEGATIVES:
        print("基于LLM负例的对比数据增强被禁用。")
    else:
        print("警告: 未执行基于LLM负例的对比数据增强，因为 'raw_datasets' 或 'pseudo_labels_list' 未正确准备或数量不匹配。")
#%%
# --- 单元格 5: Tokenizer 初始化和数据预处理函数 📝 ---
print(f"正在从 '{MODEL_NAME}' 加载用于微调的Tokenizer...") # 主微调模型的Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("微调Tokenizer加载完成。")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 
    print(f"微调Tokenizer的pad_token未设置，已将其设置为eos_token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
else:
    print(f"微调Tokenizer的pad_token已设置为: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

def preprocess_function_causal(examples):
    full_prompts = []
    input_texts_for_prompt = examples["text"]
    target_outputs = examples["quadruples_str"]

    for input_text, target_output in zip(input_texts_for_prompt, target_outputs):
        input_text_str = str(input_text) if input_text is not None else ""
        target_output_str = str(target_output) if target_output is not None else ""
        
        prompt_part = PROMPT_TEMPLATE.format(input_text=input_text_str)
        full_text = prompt_part + target_output_str + tokenizer.eos_token
        full_prompts.append(full_text)

    model_inputs = tokenizer(
        full_prompts,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False, 
        return_attention_mask=True 
    )

    labels = [list(ids) for ids in model_inputs["input_ids"]] 

    for i in range(len(examples["text"])):
        # input_text_str = str(examples["text"][i]) if examples["text"][i] is not None else ""
        # prompt_part_only = PROMPT_TEMPLATE.format(input_text=input_text_str)
        current_input_ids = model_inputs["input_ids"][i]
        current_labels = labels[i]
        
        answer_part_str = str(examples["quadruples_str"][i]) if examples["quadruples_str"][i] is not None else ""
        # Tokenize答案部分（不加特殊token，因为它们已在full_text中处理）
        answer_tokens = tokenizer(answer_part_str + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        
        len_to_mask = len(current_input_ids) - len(answer_tokens)
        
        if len_to_mask < 0: 
            # print(f"警告: 样本 {i} 的计算屏蔽长度为负 ({len_to_mask})。Input: '{str(examples['text'][i])[:50]}...', Target: '{answer_part_str}'")
            # print(f"  Input IDs len: {len(current_input_ids)}, Answer tokens len: {len(answer_tokens)}")
            # 如果答案比整个序列还长（或因截断导致不匹配），则不屏蔽任何内容，或仅屏蔽BOS
            if current_input_ids and current_input_ids[0] == tokenizer.bos_token_id:
                 len_to_mask = 1 # 只屏蔽BOS
            else:
                 len_to_mask = 0 # 不屏蔽
        
        for j in range(min(len_to_mask, len(current_labels))): 
            current_labels[j] = -100
        
        if answer_part_str and all(l == -100 for l in current_labels):
            # print(f"警告: 样本 {i} 的所有标签都被屏蔽，但目标输出不为空 ('{answer_part_str}')。屏蔽长度: {len_to_mask}")
            if current_labels:
                 current_labels[-1] = current_input_ids[-1]

    model_inputs["labels"] = labels
    return model_inputs
#%%
# --- 单元格 6: 对数据集进行Tokenize 🧩 ---
print("开始对数据集进行tokenize和预处理 (适配Causal LM)...")
if 'raw_datasets' in locals() and raw_datasets and 'train' in raw_datasets and raw_datasets['train'] is not None:
    tokenized_datasets = raw_datasets.map(
        preprocess_function_causal, 
        batched=True, 
        remove_columns=raw_datasets["train"].column_names 
    )
    print("\n数据tokenize和预处理完成:")
    print(tokenized_datasets) 

    if tokenized_datasets and 'train' in tokenized_datasets and len(tokenized_datasets['train']) > 0:
        print(f"\nTokenize后的训练集样本 (检查input_ids和labels的屏蔽情况):")
        sample_idx = 0 
        if sample_idx < len(tokenized_datasets['train']) and sample_idx < len(raw_datasets['train']):
            print(f"  原始/增强后输入文本 (text): {raw_datasets['train'][sample_idx]['text']}") 
            print(f"  原始目标输出 (quadruples_str): {raw_datasets['train'][sample_idx]['quadruples_str']}")
            
            tokenized_sample = tokenized_datasets['train'][sample_idx]
            print(f"\n  Tokenized input_ids (前60): {tokenized_sample['input_ids'][:60]}")
            print(f"  Decoded input_ids (前60): {tokenizer.decode(tokenized_sample['input_ids'][:60])}")
            
            print(f"\n  Tokenized labels (前60, -100表示已屏蔽): {tokenized_sample['labels'][:60]}")
            
            first_label_idx = -1
            for idx, lbl_id in enumerate(tokenized_sample['labels']):
                if lbl_id != -100:
                    first_label_idx = idx
                    break
            
            if first_label_idx != -1:
                decoded_label_part = tokenizer.decode([l for l in tokenized_sample['labels'][first_label_idx:] if l != -100])
                print(f"  Decoded labels from first non-masked token (部分): {decoded_label_part}")
            else:
                print("  注意：该样本的所有标签都被屏蔽了。")
                if raw_datasets['train'][sample_idx]['quadruples_str']: 
                     print(f"  原始目标输出非空 ('{raw_datasets['train'][sample_idx]['quadruples_str']}'), 但所有标签被屏蔽，请仔细检查preprocess_function_causal中的屏蔽逻辑。")
        else:
            print(f"警告：选择的样本索引 {sample_idx} 超出训练集范围。")
    else:
        print("\n警告: Tokenize后的数据集为空或不完整。")
else:
    print("错误: 'raw_datasets' 或其训练集未定义/为空，无法进行tokenize。请先成功执行数据加载和（可选的）增强单元格。")

#%%
# --- 单元格 7: 模型加载与PEFT (LoRA) 配置 🧱 ---
# 主微调模型使用 MODEL_NAME
print(f"准备从 '{MODEL_NAME}' 加载用于微调的Causal LM...")

bnb_config_finetune = None # 与生成器LLM的bnb_config区分开
if USE_QUANTIZATION: # 使用主配置中的量化设置
    if QUANTIZATION_TYPE == "nf4" or QUANTIZATION_TYPE == "fp4":
        bnb_config_finetune = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=QUANTIZATION_TYPE, 
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, 
        )
        print(f"微调模型将使用4-bit量化 ({QUANTIZATION_TYPE}) 加载。")
    elif QUANTIZATION_TYPE == "int8":
        bnb_config_finetune = BitsAndBytesConfig(load_in_8bit=True)
        print("微调模型将使用8-bit量化加载。")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config_finetune,
    trust_remote_code=True,
    device_map="auto" 
)
print(f"用于微调的模型 '{MODEL_NAME}' 加载完成。")

if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"微调模型配置的pad_token_id已设置为tokenizer的pad_token_id: {tokenizer.pad_token_id}")

if hasattr(model, 'config') and model.config.model_type and "qwen2" in model.config.model_type.lower() and hasattr(model, 'enable_input_require_grads'):
    try:
        model.enable_input_require_grads()
        print("已为Qwen2微调模型调用 enable_input_require_grads()")
    except Exception as e_grad:
        print(f"为Qwen2微调模型调用 enable_input_require_grads() 时发生错误 (可能不需要或不适用): {e_grad}")

if USE_LORA:
    print("\n为微调模型启用LoRA。")
    # training_args 现在应该在Cell 9中定义，这里我们假设它会被定义
    # 为了更安全，可以在 prepare_model_for_kbit_training 调用时直接传递布尔值
    use_grad_ckpt_for_lora = True # 默认启用，除非在TrainingArguments中显式关闭
    if 'training_args' in locals() and hasattr(training_args, 'gradient_checkpointing'):
        use_grad_ckpt_for_lora = training_args.gradient_checkpointing

    if hasattr(model, "is_loaded_in_8bit") or hasattr(model, "is_loaded_in_4bit") or (USE_QUANTIZATION and bnb_config_finetune is not None):
        print("检测到微调模型已量化加载，准备k-bit训练...")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_grad_ckpt_for_lora) 
        print(f"微调模型已为k-bit训练准备就绪。梯度检查点将{'启用' if use_grad_ckpt_for_lora else '禁用'}。")
    
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, 
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM 
    )
    print("LoRA配置已创建:")
    print(lora_config)
    
    model = get_peft_model(model, lora_config) 
    print("\nLoRA适配器已应用到微调模型。")
    model.print_trainable_parameters() 
else:
    print("\n未启用LoRA，将进行全参数微调。")

print(f"当前微调模型所在设备: {model.device}")
if hasattr(model, 'hf_device_map'):
    print(f"微调模型层设备分布: {model.hf_device_map}")
#%%
# --- 单元格 8: 评估指标计算函数 📊 ---
# (保持不变，但确保 parse_quadruples 在此定义或之前已定义)
def parse_quadruples(text_str): # 确保这是完整的定义
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
            
        elements = [e.strip() for e in part_cleaned.split(" | ")] # 注意分隔符中的空格
        
        if len(elements) == 4:
            quadruples.append(elements)
    return quadruples

def calculate_f1_metrics(preds_quads_list, labels_quads_list):
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
                if i in matched_gold_indices_hard: continue
                if p_quad == g_quad: 
                    true_positives_hard += 1
                    matched_gold_indices_hard.add(i)
                    break 
        
        matched_gold_indices_soft = set()
        for p_quad in pred_quads_for_sample:
            if len(p_quad) != 4: continue
            for i, g_quad in enumerate(gold_quads_for_sample):
                if len(g_quad) != 4: continue 
                if i in matched_gold_indices_soft: continue
                if p_quad[2].strip().lower() == g_quad[2].strip().lower() and \
                   p_quad[3].strip().lower().startswith(g_quad[3].strip().lower().split(" ")[0]): # 比较主要部分
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
    generated_token_ids, label_ids_from_input = eval_preds 
    # label_ids_from_input 包含了 -100 用于屏蔽提示部分
    # generated_token_ids 是模型生成的序列，可能也包含提示部分（如果generate未正确配置只输出新token）
    # 通常，Trainer的generate会处理好，只返回新生成的tokens，或者我们需要从完整序列中提取

    # 假设 generated_token_ids 是模型新生成的token (不含提示)
    # 如果它包含了提示，我们需要从模型输出中移除提示部分
    # decoded_preds_str = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # 一个更鲁棒的方法是，我们知道label_ids_from_input是完整的，包含-100
    # 而 generated_token_ids 是模型针对这些输入生成的完整序列（提示+答案）
    # 我们需要从 generated_token_ids 中提取答案部分，或者从 decoded_preds_str 中提取

    # 方案1: 假设 generated_token_ids 是完整的（提示+答案）
    decoded_preds_full_str = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # 从解码的完整预测中提取真实的答案部分
    pred_answer_strs = []
    keyword_separator = "模型输出：" # 与PROMPT_TEMPLATE一致
    for full_pred_text in decoded_preds_full_str:
        if keyword_separator in full_pred_text:
            pred_answer_strs.append(full_pred_text.split(keyword_separator, 1)[-1].strip())
        else: # 如果模型没按套路出牌
            # print(f"警告: 预测结果中未找到分隔符 '{keyword_separator}'. Full pred: '{full_pred_text[:100]}...'")
            # 尝试移除已知的提示头（这比较脆弱）
            # prompt_head_approx = PROMPT_TEMPLATE.split("{input_text}")[0].split("<<SYS>>")[-1].strip() # 取系统提示之后的部分
            # if full_pred_text.startswith(prompt_head_approx):
            #     pred_answer_strs.append(full_pred_text[len(prompt_head_approx):].strip())
            # else:
            pred_answer_strs.append(full_pred_text) # 后备：使用全部，可能包含提示

    # 解码真实标签（答案部分）
    processed_label_ids = np.where(label_ids_from_input != -100, label_ids_from_input, tokenizer.pad_token_id)
    decoded_labels_full_str = tokenizer.batch_decode(processed_label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    actual_target_strs = []
    for full_label_text in decoded_labels_full_str:
        if keyword_separator in full_label_text:
            actual_target_strs.append(full_label_text.split(keyword_separator, 1)[-1].strip())
        else:
            # print(f"警告: 解码的标签中未找到分隔符 '{keyword_separator}'. Full label: '{full_label_text[:100]}...'")
            actual_target_strs.append("") 

    pred_quads_list = [parse_quadruples(p_str) for p_str in pred_answer_strs]
    label_quads_list = [parse_quadruples(l_str) for l_str in actual_target_strs] 
    
    results = calculate_f1_metrics(pred_quads_list, label_quads_list)
    return results

print("评估指标相关函数 (parse_quadruples, calculate_f1_metrics, compute_metrics_causal) 已定义。")
# 更新 parse_quadruples_fn_to_use (如果单元格3.6在单元格8之前运行了)
if 'parse_quadruples_fn_to_use' in globals() and parse_quadruples_fn_to_use.__name__ == 'parse_quadruples_placeholder':
    parse_quadruples_fn_to_use = parse_quadruples
    print("DEBUG: 已将 parse_quadruples_fn_to_use 更新为单元格8的完整定义。")
#%%
# --- 单元格 9: 训练参数配置与数据整理器 (最终确认版) 📋 ---
# (保持与您确认可用的版本一致)
if 'CALCULATED_STEPS_PER_EPOCH' not in locals(): CALCULATED_STEPS_PER_EPOCH = 500 # 安全默认值
print(f"DEBUG: 最终确认的 CALCULATED_STEPS_PER_EPOCH: {CALCULATED_STEPS_PER_EPOCH}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    do_eval=True,                 
    eval_strategy="steps",        
    eval_steps=CALCULATED_STEPS_PER_EPOCH, 
    save_strategy="steps",        
    save_steps=CALCULATED_STEPS_PER_EPOCH, 
    save_total_limit=2, 
    logging_dir=f"{OUTPUT_DIR}/logs", 
    logging_strategy="steps", 
    logging_steps=max(1, CALCULATED_STEPS_PER_EPOCH // 10 if CALCULATED_STEPS_PER_EPOCH > 10 else 50),
    load_best_model_at_end=True, 
    metric_for_best_model="avg_f1", 
    greater_is_better=True,      
    fp16=(torch.cuda.is_available() and not USE_QUANTIZATION), # fp16 与 4-bit/8-bit 量化通常不一起用
    bf16=(torch.cuda.is_bf16_supported() and not USE_QUANTIZATION), # bf16 同上
    lr_scheduler_type=LR_SCHEDULER_TYPE, 
    warmup_ratio=WARMUP_RATIO,           
    report_to=["tensorboard"], 
    seed=SEED,                 
    optim="paged_adamw_8bit" if USE_QUANTIZATION else "adamw_torch",
    remove_unused_columns=True, # 推荐设置为True
    gradient_checkpointing=True, # 为节省显存启用
    gradient_checkpointing_kwargs={"use_reentrant": False}, # 推荐的梯度检查点设置
)
print(f"训练参数 (TrainingArguments) 配置完成。评估和保存策略均设置为 'steps'，每 {CALCULATED_STEPS_PER_EPOCH} 步执行一次。")

if 'training_args' in locals() and training_args is not None:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, 
        pad_to_multiple_of=8 if (training_args.fp16 or training_args.bf16) else None 
    )
    print("数据整理器 (DataCollatorForSeq2Seq) 初始化完成。")
#%%
# --- 单元格 10: 初始化 Trainer 👨‍🏫 ---
# (保持不变)
trainer = Trainer(
    model=model, args=training_args,                  
    train_dataset=tokenized_datasets["train"] if 'tokenized_datasets' in locals() and tokenized_datasets and "train" in tokenized_datasets else None, 
    eval_dataset=tokenized_datasets["validation"] if 'tokenized_datasets' in locals() and tokenized_datasets and "validation" in tokenized_datasets else None, 
    tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics_causal, 
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)] 
)
print("Trainer 初始化完成。")
if not ('tokenized_datasets' in locals() and tokenized_datasets and "train" in tokenized_datasets and tokenized_datasets["train"]):
    print("警告: Trainer的训练集未正确设置。")
#%%
# --- 单元格 11: 开始模型训练 🚀 ---
# (保持不变)
print("即将开始模型训练...")
if trainer.train_dataset is None:
    print("错误: 训练数据集未设置，无法开始训练。")
else:
    try:
        if model.generation_config is None: 
            model.generation_config = GenerationConfig.from_model_config(model.config)
            print("已为模型设置默认的GenerationConfig。")
        
        model.generation_config.max_new_tokens = MAX_TARGET_LENGTH 
        model.generation_config.num_beams = 3 
        model.generation_config.early_stopping = True
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None: # 确保eos_token_id也设置
             model.generation_config.eos_token_id = tokenizer.eos_token_id

        print(f"模型评估时将使用以下生成配置: num_beams={model.generation_config.num_beams}, max_new_tokens={model.generation_config.max_new_tokens}")
        train_result = trainer.train()
        print("\n模型训练完成!")
        print("正在保存模型 (LoRA adapter)...")
        trainer.save_model(OUTPUT_DIR) 
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
# --- 单元格 12: 预测/推理函数设置 🔮 ---
# (保持不变，但确保 MAX_INPUT_LENGTH 和 MAX_TARGET_LENGTH 从Cell 2正确传递)
model_to_predict = trainer.model if 'trainer' in locals() and hasattr(trainer, 'model') else None
if model_to_predict:
    model_to_predict.eval() 
    print(f"用于预测的模型已准备好，当前设备: {model_to_predict.device}")
else:
    print("警告: 'trainer.model' 未找到，无法设置 model_to_predict。示例预测和提交文件生成可能失败。")


def predict_quadruples_causal(text_list, model, tokenizer_pred, max_input_len_pred, max_target_gen_len_pred):
    parsed_results_list = []
    if model is None or tokenizer_pred is None:
        print("错误: 预测所需的模型或tokenizer未提供。")
        return [{"original_text": t, "extracted_answer_string": "ERROR: Model/Tokenizer missing", "parsed_quadruples": []} for t in text_list]

    for text_input in text_list:
        prompt_for_inference = PROMPT_TEMPLATE.format(input_text=text_input)
        
        # 确保 max_length 对于提示是合理的
        max_prompt_len = max_input_len_pred - max_target_gen_len_pred
        if max_prompt_len <= 0 : max_prompt_len = max_input_len_pred // 2 # 至少给提示一半空间

        inputs = tokenizer_pred(
            prompt_for_inference, return_tensors="pt", truncation=True, 
            max_length=max_prompt_len, padding=False 
        ).to(model.device) 

        with torch.no_grad():
            # 复制并更新生成配置，以防修改全局配置
            current_gen_config = GenerationConfig(**model.generation_config.to_dict())
            current_gen_config.max_new_tokens = max_target_gen_len_pred
            # num_beams, early_stopping 等应已在 model.generation_config 中设置

            outputs = model.generate(**inputs, generation_config=current_gen_config)
        
        full_generated_text = tokenizer_pred.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        answer_part_str = ""
        keyword_separator = "模型输出：" 
        split_parts = full_generated_text.split(keyword_separator, 1)
        if len(split_parts) > 1:
            answer_part_str = split_parts[1].strip()
        else: 
            prompt_head_for_removal = PROMPT_TEMPLATE.format(input_text=text_input).split(keyword_separator)[0] + keyword_separator
            decoded_prompt_head = tokenizer_pred.decode(tokenizer_pred.encode(prompt_head_for_removal.split("用户提供的文本如下：")[0], add_special_tokens=False), skip_special_tokens=True) # 尝试解码提示头部分
            
            # 这是一个更复杂的尝试，试图移除提示部分
            # 如果解码后的完整文本以解码后的提示头开始，则移除它
            # 这非常依赖于解码的一致性
            # if full_generated_text.startswith(decoded_prompt_head.strip()):
            #    answer_part_str = full_generated_text[len(decoded_prompt_head.strip()):].strip()
            #    if answer_part_str.startswith(keyword_separator): # 再次检查
            #         answer_part_str = answer_part_str[len(keyword_separator):].strip()
            # else:
            # print(f"警告：无法在生成文本中通过关键词 '{keyword_separator}' 定位答案，且提示头不匹配。Full: '{full_generated_text[:200]}...'")
            answer_part_str = full_generated_text # 作为最后的手段，可能包含提示

        parsed_quads = parse_quadruples(answer_part_str) # 使用Cell 8定义的parse_quadruples
        parsed_results_list.append({
            "original_text": text_input, "full_generated_text": full_generated_text, 
            "extracted_answer_string": answer_part_str, "parsed_quadruples": parsed_quads
        })
    return parsed_results_list

print("预测/推理相关函数 (predict_quadruples_causal) 已定义。")
#%%
# (保持不变)
sample_test_texts_for_prediction = [
    "那些同性恋真恶心，败坏社会风气。", "这道菜味道不错，下次还来。",
    "上海人就是排外，看不起外地人。", "黑人都是罪犯，应该被赶走。",
    "你可真是头蠢驴，这都做不好。"
]
print("\n开始运行示例预测...")
if 'model_to_predict' in locals() and model_to_predict is not None:
    predictions = predict_quadruples_causal(
        sample_test_texts_for_prediction, model_to_predict, tokenizer,
        MAX_INPUT_LENGTH, MAX_TARGET_LENGTH 
    )
    print("\n示例预测结果:")
    for item in predictions:
        print(f"原始文本: {item['original_text']}")
        print(f"提取答案: {item['extracted_answer_string']}")
        print(f"解析四元组: {item['parsed_quadruples']}")
        print("-" * 30)
else:
    print("错误: 'model_to_predict' 未定义或为None。无法运行示例预测。")

#%%
# --- 单元格 14: 加载官方测试数据并生成提交文件 📤 ---
# (保持不变，但确保路径和变量名正确)
# official_test_file_path = "/kaggle/input/nlptrain/test1.json" # 您的测试文件路径
# official_test_file_path = "./test1.json" # 假设在当前目录
# official_test_file_path = "./test2.json" # 或者 test2.json

# 确保以下变量已定义：
# official_test_file_path, model_to_predict, tokenizer, EVAL_BATCH_SIZE, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH

# 示例路径，请根据您的实际情况修改
# official_test_file_path_to_use = "./test1.json" 
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

official_test_file_path_to_use = "./test1.json" # 或您的 test2.json 路径

if 'model_to_predict' not in locals() or model_to_predict is None:
    print("错误: 'model_to_predict' 未定义。无法进行官方测试数据预测。")
elif 'tokenizer' not in locals() or tokenizer is None:
    print("错误: 'tokenizer' 未定义。无法进行官方测试数据预测。")
elif not os.path.exists(official_test_file_path_to_use):
    print(f"错误: 测试文件路径 '{official_test_file_path_to_use}' 不存在。")
else:
    print(f"\n开始处理官方测试文件: {official_test_file_path_to_use}")
    official_test_texts, official_test_ids = load_official_test_data(official_test_file_path_to_use) # load_official_test_data 在您之前的代码中定义
    
    if official_test_texts:
        submission_outputs_strings = []
        inference_batch_size = EVAL_BATCH_SIZE 
        print(f"开始对 {len(official_test_texts)} 条测试数据进行预测 (批次大小: {inference_batch_size})...")
        for i in tqdm(range(0, len(official_test_texts), inference_batch_size), desc="官方测试集预测"):
            batch_texts = official_test_texts[i : i + inference_batch_size]
            batch_predictions = predict_quadruples_causal(
                batch_texts, model_to_predict, tokenizer,
                MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
            )
            for item_prediction in batch_predictions:
                submission_outputs_strings.append(item_prediction['extracted_answer_string'])
        
        #submission_file_path = "/kaggle/working/submission.txt" # Kaggle 工作目录
        submission_file_path = "./newsubmission.txt" # 或者本地路径
        try:
            with open(submission_file_path, "w", encoding="utf-8") as f:
                for line_content in submission_outputs_strings:
                    f.write(line_content + "\n")
            print(f"\n提交文件已成功生成: {submission_file_path}")
            print(f"该文件包含 {len(submission_outputs_strings)} 行预测。")
        except Exception as e:
            print(f"写入提交文件 '{submission_file_path}' 时发生错误: {e}")
    else:
        print(f"未能从 '{official_test_file_path_to_use}' 加载任何测试数据进行预测。")
#%%
