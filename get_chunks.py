from ragflow_sdk import RAGFlow
import os

rag_object = RAGFlow(api_key="ragflow-dOaxETaP66RCzPgPNYKmKyORPPAaO6P96TVSgmOrA5E", base_url="http://10.10.11.13:9380")

# 前端可配置参数
TOTAL_PAGES = 100  # 要遍历的总页数
CHUNKS_PER_PAGE = 50  # 每页chunk数量

# 创建输出目录
output_dir = "ragflow_chunks_Glean_VCU"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")

# 先获取所有数据集
datasets = rag_object.list_datasets()
print(f"找到 {len(datasets)} 个数据集")

# 查找指定ID的数据集
target_dataset_id = "65825a60ffe411f08398ce8499fba1bf"    ##"d483981bfb6511f0a535065a3d57bbcc"
target_dataset = None

for dataset in datasets:
    print(f"数据集ID: {dataset.id}, 名称: {dataset.name}")
    if dataset.id == target_dataset_id:
        target_dataset = dataset
        break

if target_dataset is None:
    print(f"未找到ID为 {target_dataset_id} 的数据集")
    exit(1)

print(f"使用数据集: {target_dataset.name} (ID: {target_dataset.id})")

# 获取文档并遍历chunks
docs = target_dataset.list_documents()
print(f"数据集中有 {len(docs)} 个文档")

if len(docs) > 0:
    dataset_chunks_total = 0
    # 遍历所有文档
    for doc_index, doc in enumerate(docs):
        print(f"\n=== 处理文档 {doc_index + 1}/{len(docs)}: {doc.name} ===")
        
        try:
            doc_chunks_total = 0
            
            # 遍历所有页面
            for page_num in range(1, TOTAL_PAGES + 1):
                print(f"\n  --- 文档 '{doc.name}' 获取第 {page_num} 页 ---")
                
                try:
                    chunks = doc.list_chunks(page=page_num)
                    
                    if len(chunks) == 0:
                        print(f"  第 {page_num} 页没有chunks，可能已到达最后一页")
                        break  # 如果当前页没有chunks，说明已经到达最后一页
                    
                    print(f"  第 {page_num} 页包含 {len(chunks)} 个chunks")
                    doc_chunks_total += len(chunks)
                    
                    # 遍历当前页的所有chunks
                    for chunk_index, chunk in enumerate(chunks):
                        doc_chunk_index = (page_num - 1) * CHUNKS_PER_PAGE + chunk_index + 1
                        #print(f"    Chunk {doc_chunk_index}: {chunk}")
                        print(f"    Chunk {doc_chunk_index}")
                        
                        # 提取content字段内容
                        if hasattr(chunk, 'content') and chunk.content:
                            content = chunk.content
                            
                            # 生成文件名
                            safe_doc_name = "".join(c for c in doc.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                            if not safe_doc_name:
                                safe_doc_name = f"doc_{doc_index + 1}"
                            
                            filename = f"{safe_doc_name}_chunk_{doc_chunk_index}.md"
                            filepath = os.path.join(output_dir, filename)
                            
                            # 写入文件
                            try:
                                with open(filepath, 'w', encoding='utf-8') as f:
                                    # 写入内容
                                    f.write(content)
                                    
                                print(f"      已保存内容到文件: {filepath}")
                                
                            except Exception as write_error:
                                print(f"      写入文件时出错: {write_error}")
                        else:
                            print(f"      Chunk {doc_chunk_index} 没有content字段或content为空")
                        
                except Exception as page_error:
                    print(f"  获取第 {page_num} 页时出错: {page_error}")
                    if "not found" in str(page_error).lower() or "404" in str(page_error):
                        print(f"  第 {page_num} 页不存在，已达到最后一页")
                        break
                    continue
            
            dataset_chunks_total += doc_chunks_total
            print(f"\n文档 '{doc.name}' 总计包含 {doc_chunks_total} 个chunks")
                
        except Exception as e:
            print(f"  处理文档 '{doc.name}' 时出错: {e}")
            continue
            
    print(f"\n总计处理了 {len(docs)} 个文档")
    print(f"数据集总计包含 {dataset_chunks_total} 个chunks")
    print(f"所有chunk内容已保存到目录: {output_dir}")
else:
    print("数据集中没有文档")
