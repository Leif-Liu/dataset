import json
import os

def process_chunks_json(input_file="chunks_json/datasets-chunks-2.json", output_dir="processed_dataset"):
    """
    处理text-chunks.json文件，将summary与content组合生成预训练数据集
    
    Args:
        input_file: 输入的JSON文件路径
        output_dir: 输出目录
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 读取JSON文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        print(f"成功读取 {input_file}，包含 {len(chunks_data)} 个chunks")
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"错误：JSON解析失败 - {e}")
        return
    
    # 处理数据集的不同格式
    processed_data = []
    
    # 1. 生成组合格式（summary + content）
    combined_dataset = []
    for idx, chunk in enumerate(chunks_data):
        combined_text = f"{chunk.get('summary', '')}\n\n{chunk.get('content', '')}"
        
        combined_item = {
            "text": combined_text,
        }
        combined_dataset.append(combined_item)
    
    # 2. 生成问答格式
    qa_dataset = []
    for idx, chunk in enumerate(chunks_data):
        qa_item = {
            "id": idx + 1,
            "question": f"请总结以下内容：\n{chunk.get('content', '')}",
            "answer": chunk.get('summary', ''),
            "source": chunk.get('fileName', ''),
            "name": chunk.get('name', '')
        }
        qa_dataset.append(qa_item)
    
    # 3. 生成纯文本格式（用于语言模型预训练）
    text_dataset = []
    for idx, chunk in enumerate(chunks_data):
        # 格式1：直接组合
        text_item1 = {
            "id": f"{idx + 1}_combined",
            "text": f"{chunk.get('summary', '')}\n\n{chunk.get('content', '')}"
        }
        
        # 格式2：结构化组合
        text_item2 = {
            "id": f"{idx + 1}_structured",
            "text": f"文档: {chunk.get('fileName', '')}\n摘要: {chunk.get('summary', '')}\n\n详细内容:\n{chunk.get('content', '')}"
        }
        
        text_dataset.extend([text_item1, text_item2])
    
    # 保存不同格式的数据集
    datasets = {
        "combined_dataset.json": combined_dataset,
        "qa_dataset.json": qa_dataset,
        "text_dataset.json": text_dataset
    }
    
    for filename, dataset in datasets.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"已保存 {filename}: {len(dataset)} 条记录")
    
    # 生成纯文本文件（用于某些预训练框架）
    text_output_path = os.path.join(output_dir, "training_text.txt")
    with open(text_output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks_data:
            # 写入结构化文本
            f.write(f"=== {chunk.get('fileName', 'Unknown')} ===\n")
            f.write(f"摘要: {chunk.get('summary', '')}\n\n")
            f.write(f"{chunk.get('content', '')}\n")
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"已保存纯文本文件: training_text.txt")
    
    # 生成统计信息
    stats = {
        "total_chunks": len(chunks_data),
        "total_characters": sum(len(chunk.get('content', '') + chunk.get('summary', '')) for chunk in chunks_data),
        "average_size": sum(chunk.get('size', 0) for chunk in chunks_data) / len(chunks_data) if chunks_data else 0,
        "unique_files": len(set(chunk.get('fileName', '') for chunk in chunks_data)),
        "projects": list(set(chunk.get('projectId', '') for chunk in chunks_data))
    }
    
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 数据集统计信息 ===")
    print(f"总chunks数量: {stats['total_chunks']}")
    print(f"总字符数: {stats['total_characters']:,}")
    print(f"平均大小: {stats['average_size']:.1f}")
    print(f"唯一文件数: {stats['unique_files']}")
    print(f"项目数: {len(stats['projects'])}")
    print(f"统计信息已保存到: {stats_path}")

def generate_training_samples(input_file="chunks_json/text-chunks.json", output_file="processed_dataset/training_samples.jsonl"):
    """
    生成适合机器学习训练的样本格式（JSONL格式）
    """
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, chunk in enumerate(chunks_data):
            # 创建多种训练样本格式
            
            # 样本1：摘要生成任务
            sample1 = {
                "instruction": "请为以下内容生成摘要：",
                "input": chunk.get('content', ''),
                "output": chunk.get('summary', ''),
                "task_type": "summarization"
            }
            f.write(json.dumps(sample1, ensure_ascii=False) + '\n')
            
            # 样本2：内容理解任务
            sample2 = {
                "instruction": "根据摘要，这个文档可能包含什么详细内容？",
                "input": chunk.get('summary', ''),
                "output": chunk.get('content', ''),
                "task_type": "content_expansion"
            }
            f.write(json.dumps(sample2, ensure_ascii=False) + '\n')
            
            # 样本3：文档分类任务（基于文件名）
            file_type = chunk.get('fileName', '').split('_')[0] if '_' in chunk.get('fileName', '') else 'general'
            sample3 = {
                "instruction": "这个文档属于什么类型？",
                "input": f"{chunk.get('summary', '')}\n\n{chunk.get('content', '')}",
                "output": file_type,
                "task_type": "classification"
            }
            f.write(json.dumps(sample3, ensure_ascii=False) + '\n')
    
    print(f"已生成训练样本文件: {output_file}")

def load_and_validate_dataset(dataset_path="processed_dataset/combined_dataset.json"):
    """
    加载并验证处理后的数据集，为后续训练做准备
    """
    print(f"\n=== 加载训练数据集 ===")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"✅ 成功加载数据集: {dataset_path}")
        print(f"📊 数据集大小: {len(dataset)} 条记录")
        
        # 统计信息
        total_chars = sum(len(item['text']) for item in dataset)
        avg_length = total_chars / len(dataset) if dataset else 0
        
        print(f"📝 总字符数: {total_chars:,}")
        print(f"📏 平均长度: {avg_length:.1f} 字符/条")
        
        # 显示前几个样本
        print(f"\n📋 数据样本预览（前3条）:")
        for i, item in enumerate(dataset[:3]):
            text_preview = item['text'][:200] + "..." if len(item['text']) > 200 else item['text']
            print(f"\n样本 {i+1}:")
            print(f"  长度: {len(item['text'])} 字符")
            print(f"  内容: {text_preview}")
        
        # 长度分布分析
        lengths = [len(item['text']) for item in dataset]
        lengths.sort()
        
        print(f"\n📈 长度分布统计:")
        print(f"  最短: {min(lengths)} 字符")
        print(f"  最长: {max(lengths)} 字符")
        print(f"  中位数: {lengths[len(lengths)//2]} 字符")
        print(f"  75%分位数: {lengths[int(len(lengths)*0.75)]} 字符")
        print(f"  95%分位数: {lengths[int(len(lengths)*0.95)]} 字符")
        
        return dataset
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到数据集文件 {dataset_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON解析失败 - {e}")
        return None
    except Exception as e:
        print(f"❌ 错误：加载数据集时出现异常 - {e}")
        return None

def prepare_for_training(dataset, output_file="processed_dataset/training_ready.json"):
    """
    为训练准备最终的数据集格式
    """
    print(f"\n=== 准备训练数据 ===")
    
    # 准备训练格式的数据
    training_data = []
    for i, item in enumerate(dataset):
        training_item = {
            "id": i + 1,
            "text": item["text"],
            "length": len(item["text"]),
            "word_count": len(item["text"].split())
        }
        training_data.append(training_item)
    
    # 保存训练准备数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 训练数据已保存到: {output_file}")
        print(f"🎯 数据格式: 每条记录包含 id, text, length, word_count 字段")
        return training_data
        
    except Exception as e:
        print(f"❌ 保存训练数据时出错: {e}")
        return None

if __name__ == "__main__":
    print("开始处理text-chunks.json文件...")
    
    # 处理基本数据集
    process_chunks_json()
    
    print("\n" + "="*50)
    
    # 生成训练样本
    print("生成机器学习训练样本...")
    generate_training_samples()
    
    print("\n" + "="*50)
    
    # 加载并验证数据集
    dataset = load_and_validate_dataset()
    
    if dataset:
        # 准备训练数据
        training_data = prepare_for_training(dataset)
        
        if training_data:
            print(f"\n🎉 数据集处理完成！可用于训练的数据集已准备就绪。")
    
    print("\n📁 生成的文件清单：")
    print("- processed_dataset/combined_dataset.json      (主要训练数据)")
    print("- processed_dataset/qa_dataset.json           (问答格式)")
    print("- processed_dataset/text_dataset.json         (纯文本格式)")
    print("- processed_dataset/training_text.txt         (文本文件)")
    print("- processed_dataset/training_samples.jsonl    (JSONL格式)")
    print("- processed_dataset/training_ready.json       (训练就绪数据)")
    print("- processed_dataset/dataset_stats.json        (统计信息)")
    
    print("\n🚀 数据集已准备完毕，可以开始训练！") 

    