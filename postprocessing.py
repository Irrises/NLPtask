import re # 导入正则表达式模块

def extract_quadruplets(input_file_path, output_file_path):
    """
    从 input_file_path (输入文件路径) 读取数据，提取符合规范的四元组行，
    并将其写入 output_file_path (输出文件路径)。

    一个四元组行的定义是：
    元素1 | 元素2 | 元素3 | 元素4 [END]
    同一行中的多个四元组由 " [SEP] " 分隔。
    """
    quadruplets_found = [] # 用于存储找到的四元组

    # 用于识别单个四元组结构的正则表达式。
    # 它会寻找：
    # - 任意字符（非贪婪模式）作为元素1，后跟 " | "
    # - 任意字符（非贪婪模式）作为元素2，后跟 " | "
    # - 任意字符（非贪婪模式）作为元素3，后跟 " | "
    # - 任意字符（非贪婪模式）作为元素4，后跟 " [END]"
    # 这个模式确保在 "[END]" 之前恰好有三个 " | " 分隔符
    quadruplet_pattern = re.compile(r"^[^|]+\|[^|]+\|[^|]+\|[^|]+ \[END\]$")

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile: # 以只读模式和UTF-8编码打开输入文件
            for line in infile: # 遍历文件中的每一行
                line = line.strip() # 移除行首和行尾的空白字符

                # 跳过明显不是数据或元指令的行 (如空行、注释行等)
                if not line or \
                   line.startswith("输出：") or \
                   line.startswith("待分析文本：") or \
                   line.startswith("```") or \
                   "<think>" in line or \
                   line == "例结束]" or \
                   line == "[示例结束]" or \
                   line == "文本：" or \
                   line == "输入文本：" or \
                   line == "现在，请分析以下文本：":
                    continue # 继续处理下一行

                # 分割可能包含多个四元组的行 (以 " [SEP] " 为分隔符)
                parts = line.split(" [SEP] ")
                for part in parts: # 遍历分割后的各个部分
                    part = part.strip() # 清理每个部分前后的空白字符
                    # 检查该部分是否精确匹配四元组结构
                    if quadruplet_pattern.match(part):
                        quadruplets_found.append(part) # 如果匹配，则添加到结果列表中
    except FileNotFoundError:
        print(f"错误：输入文件 '{input_file_path}' 未找到。")
        return
    except Exception as e:
        print(f"读取文件 '{input_file_path}' 时发生错误：{e}")
        return

    if not quadruplets_found:
        print("未找到符合条件的四元组数据。")
        # 即使没有找到数据，也创建一个空的输出文件或不创建，取决于需求
        # 这里选择不创建空文件，如果需要空文件，可以取消下面两行的注释
        # with open(output_file_path, 'w', encoding='utf-8') as outfile:
        #     pass # 创建一个空文件
        return

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile: # 以写入模式和UTF-8编码打开输出文件
            for quad in quadruplets_found:
                outfile.write(quad + "\n") # 将每个找到的四元组写入新行
        print(f"已成功提取 {len(quadruplets_found)} 个四元组到 '{output_file_path}'")
    except IOError:
        print(f"错误：无法写入到输出文件 '{output_file_path}'。")
    except Exception as e:
        print(f"写入文件 '{output_file_path}' 时发生错误：{e}")

# --- 如何使用 ---
if __name__ == "__main__":
    # 1. 将您的数据保存到一个文件中，例如："your_input_data.txt"。
    #    确保文件内容与您之前提供的数据格式一致，并且使用UTF-8编码。
    # 2. 将此 Python 脚本保存为一个 .py 文件 (例如："process_data_zh.py")。
    # 3. 根据您的实际文件名修改下面的 input_filename 和 output_filename。
    # 4. 从您的终端运行该脚本: python process_data_zh.py

    input_filename = "zeroshot_submission.txt"   # 请替换为您的实际输入文件名
    output_filename = "zeroshot.txt" # 这是期望的输出文件名

    print(f"开始处理文件：'{input_filename}'...")
    extract_quadruplets(input_filename, output_filename)
    print("处理完成。")

    # 可选：如果您想在脚本执行后立即看到输出文件的内容（少量数据时适用）
    # try:
    #     with open(output_filename, 'r', encoding='utf-8') as f:
    #         print(f"\n--- '{output_filename}' 的内容预览 ---")
    #         print(f.read())
    #         print(f"--- 内容预览结束 ---")
    # except FileNotFoundError:
    #     print(f"提示：输出文件 '{output_filename}' 未创建或为空。")
    # except Exception as e:
    #     print(f"预览输出文件时出错: {e}")