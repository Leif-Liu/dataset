#!/usr/bin/env python3
"""
Interactive Q&A system that uses RAGFlow SDK to answer questions.
Questions can be loaded from local dataset or input manually.
Based on the RAGFlow SDK example.
"""

import json
import os
from typing import List, Dict
from ragflow_sdk import RAGFlow, Agent

# ==================== 配置区域 ====================
# RAGFlow 连接配置
API_KEY  = "ragflow-g1ZGRhNjQyNTYzZTExZjA4ZjZiODY2Nj"
BASE_URL = "http://10.10.11.7:9380"
AGENT_ID = "673a2bda7bf911f0a13f2e823edec181"

# 文件路径配置
INPUT_DATASET_PATH  = "/home/liufeng/sdk-ragflow/chunks_json/datasets-jd-0820.json"   # 输入问题数据集    /home/liufeng/sdk-ragflow/chunks_json/datasets-qa-0811.json
OUTPUT_ANSWERS_PATH = "/home/liufeng/sdk-ragflow/chunks_json/resumes-answers-0820.json"  # 输出答案文件
# ==================================================

class RAGFlowQASystem:
    def __init__(self, api_key: str, base_url: str, agent_id: str, dataset_path: str = INPUT_DATASET_PATH):
        """Initialize the RAGFlow Q&A system."""
        self.dataset_path = dataset_path
        self.qa_data = []
        self.load_dataset()
        
        # Initialize RAGFlow connection
        self.rag_object = RAGFlow(api_key=api_key, base_url=base_url)
        self.agent_id = agent_id
        self.agent = None
        self.session = None
        self.connect_to_ragflow()
    
    def load_dataset(self):
        """Load the Q&A dataset from JSON file."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.qa_data = json.load(f)
            print(f"Loaded {len(self.qa_data)} Q&A pairs from {self.dataset_path}")
        except FileNotFoundError:
            print(f"Warning: Dataset file {self.dataset_path} not found! Only manual input will be available.")
            self.qa_data = []
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {self.dataset_path}: {e}. Only manual input will be available.")
            self.qa_data = []
    
    def connect_to_ragflow(self):
        """Connect to RAGFlow and create session."""
        try:
            print(f"正在连接 RAGFlow...")
            print(f"Agent ID: {self.agent_id}")
            
            # 获取所有agents列表
            print("正在获取 agents 列表...")
            agents_list = self.rag_object.list_agents(id=self.agent_id)
            print(f"Found {len(agents_list)} agents with ID '{self.agent_id}'")
            
            if not agents_list:
                print(f"ERROR: No agent found with ID '{self.agent_id}'")
                print("正在获取所有可用的 agents...")
                try:
                    all_agents = self.rag_object.list_agents()
                    print(f"Total available agents: {len(all_agents)}")
                    print("Available agents:")
                    for i, agent in enumerate(all_agents[:5]):  # 显示前5个
                        agent_name = getattr(agent, 'name', getattr(agent, 'title', 'Unknown'))
                        print(f"  {i+1}. ID: {agent.id}, Name: {agent_name}")
                except Exception as e:
                    print(f"  Unable to list agents: {e}")
                exit(1)
            
            self.agent = agents_list[0]
            agent_name = getattr(self.agent, 'name', getattr(self.agent, 'title', 'Unknown'))
            print(f"Using agent: {agent_name} (ID: {self.agent.id})")
            
            print("正在创建会话...")
            self.session = self.agent.create_session()
            print("Successfully connected to RAGFlow!")
            
        except Exception as e:
            print(f"Error connecting to RAGFlow: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    def ask_ragflow(self, question: str, save_to_file: bool = True, start_time: str = None):
        """Ask question to RAGFlow and get streaming response."""
        if not question.strip():
            print("Please provide a valid question.")
            return ""
        
        try:
            cont = ""
            first_token = True
            first_token_time = None
            for ans in self.session.ask(question, stream=True):
                if first_token and ans.content:
                    first_token_time = self.get_current_timestamp()
                    if start_time:
                        elapsed_time = self.calculate_time_diff(start_time, first_token_time)
                        print(f"[首个token时间: {first_token_time}] [首个token耗时: {elapsed_time}]")
                    else:
                        print(f"[首个token时间: {first_token_time}]")
                    first_token = False
                print(ans.content[len(cont):], end='', flush=True)
                cont = ans.content
            
            # Add completion timestamp and calculate response time
            completion_time = self.get_current_timestamp()
            if first_token_time:
                response_time = self.calculate_time_diff(first_token_time, completion_time)
                print(f"\n[回答完成时间: {completion_time}] [回答耗时: {response_time}]")
            else:
                print(f"\n[回答完成时间: {completion_time}]")
            
            # Save to file if requested
            if save_to_file and cont.strip():
                self.save_qa_to_file(question, cont)
            
            return cont
        except Exception as e:
            error_msg = f"Error getting response from RAGFlow: {e}"
            print(error_msg)
            print(f"Error type: {type(e).__name__}")
            
            # 保存错误信息到文件
            if save_to_file:
                error_response = f"**ERROR**: {str(e)}"
                self.save_qa_to_file(question, error_response)
            
            return error_msg
    
    def save_qa_to_file(self, question: str, answer: str):
        """Save question and answer to JSON file."""
        output_file = OUTPUT_ANSWERS_PATH
        
        # Create the entry
        qa_entry = {
            "question": question,
            "ragflow_answer": answer,
            "timestamp": self.get_current_timestamp()
        }
        
        # Load existing data or create new list
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []
        
        # Add new entry
        existing_data.append(qa_entry)
        
        # Save back to file
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            print(f"[保存] 问答已保存到 {output_file}")
        except Exception as e:
            print(f"[错误] 保存失败: {e}")
    
    def get_current_timestamp(self):
        """Get current timestamp in readable format."""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def calculate_time_diff(self, start_time_str: str, end_time_str: str):
        """Calculate time difference between two timestamp strings."""
        import datetime
        try:
            start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
            diff = end_time - start_time
            total_seconds = diff.total_seconds()
            return f"{total_seconds:.2f}秒"
        except Exception as e:
            return f"计算错误: {e}"
    
    def ask_all_questions_from_dataset(self):
        """Ask all questions from the dataset to RAGFlow."""
        if not self.qa_data:
            print("No questions available in the dataset.")
            return
        
        print(f"Processing {len(self.qa_data)} questions from the dataset...\n")
        
        # Ask user for processing mode
        print("Choose processing mode:")
        print("1. Interactive mode (pause after each question)")
        print("2. Continuous mode (process all questions automatically)")
        
        while True:
            mode_input = input("Enter your choice (1 or 2): ").strip()
            if mode_input in ['1', '2']:
                continuous_mode = (mode_input == '2')
                break
            else:
                print("Please enter 1 or 2.")
        
        if continuous_mode:
            print(f"\n[连续模式] 将连续处理所有 {len(self.qa_data)} 个问题...\n")
        else:
            print(f"\n[交互模式] 将逐个处理 {len(self.qa_data)} 个问题，每个问题后暂停...\n")
        
        for i, qa_pair in enumerate(self.qa_data, 1):
            question = qa_pair.get('question', '')
            if not question:
                continue
                
            current_time = self.get_current_timestamp()
            print(f"\n{'='*60}")
            print(f"[{current_time}] Question {i}/{len(self.qa_data)}: {question}")
            print('='*60)
            print(f"\n==== RAGFlow Response (开始时间: {current_time}) ====\n")
            
            self.ask_ragflow(question, start_time=current_time)
            
            # Handle user interaction based on mode
            if not continuous_mode and i < len(self.qa_data):
                user_input = input("\nPress Enter to continue, 'q' to quit, 's' to skip to interactive mode, or 'c' to switch to continuous mode: ").strip().lower()
                if user_input == 'q':
                    break
                elif user_input == 's':
                    return
                elif user_input == 'c':
                    continuous_mode = True
                    print("\n[切换到连续模式] 将连续处理剩余问题...\n")
            elif continuous_mode and i < len(self.qa_data):
                next_time = self.get_current_timestamp()
                print(f"\n[连续模式 - {next_time}] 自动进入下一个问题... ({i}/{len(self.qa_data)} 完成)")
        
        print(f"\n{'='*60}")
        print(f"批量处理完成！共处理了 {len([qa for qa in self.qa_data if qa.get('question')])} 个问题。")
        print('='*60)
    
    def list_available_questions(self, limit: int = 10) -> List[str]:
        """List available questions from the dataset."""
        questions = [qa['question'] for qa in self.qa_data[:limit] if qa.get('question')]
        return questions
    
    def search_questions(self, keyword: str) -> List[str]:
        """Search for questions containing a specific keyword."""
        matching_questions = []
        for qa in self.qa_data:
            question = qa.get('question', '')
            if question and keyword.lower() in question.lower():
                matching_questions.append(question)
        return matching_questions


def main():
    """Main interactive loop."""
    print("Initializing RAGFlow Q&A System...")
    print(f"输入数据集: {INPUT_DATASET_PATH}")
    print(f"输出文件: {OUTPUT_ANSWERS_PATH}")
    
    # Initialize the RAGFlow Q&A system
    try:
        qa_system = RAGFlowQASystem(
            api_key=API_KEY,
            base_url=BASE_URL,
            agent_id=AGENT_ID
        )
    except SystemExit:
        return
    
    print("\n===== Miss R (RAGFlow Q&A System) ====\n")
    print("Hello. What can I do for you?")
    print("\nAvailable commands:")
    print(f"- Type your question normally (answers auto-saved to {os.path.basename(OUTPUT_ANSWERS_PATH)})")
    print("- Type 'list' to see sample questions from dataset")
    print("- Type 'search <keyword>' to find questions with a keyword")
    print("- Type 'batch' to process all questions from dataset")
    print("- Type 'quit' or 'exit' to end the session")
    
    while True:
        question = input("\n===== User ====\n> ").strip()
        
        if not question:
            continue
            
        if question.lower() in ['quit', 'exit']:
            print("\n==== Miss R ====\n")
            print("Goodbye!")
            break
        
        if question.lower() == 'list':
            print("\n==== Miss R ====\n")
            print("Here are some sample questions from the dataset:")
            questions = qa_system.list_available_questions(10)
            for i, q in enumerate(questions, 1):
                print(f"{i}. {q}")
            continue
        
        if question.lower() == 'batch':
            print("\n==== Miss R ====\n")
            qa_system.ask_all_questions_from_dataset()
            continue
        
        if question.lower().startswith('search '):
            keyword = question[7:].strip()
            if keyword:
                print("\n==== Miss R ====\n")
                matching_questions = qa_system.search_questions(keyword)
                if matching_questions:
                    print(f"Found {len(matching_questions)} questions containing '{keyword}':")
                    for i, q in enumerate(matching_questions[:10], 1):
                        print(f"{i}. {q}")
                else:
                    print(f"No questions found containing '{keyword}'.")
            else:
                print("\n==== Miss R ====\n")
                print("Please provide a keyword to search for.")
            continue
        
        current_time = qa_system.get_current_timestamp()
        print(f"\n==== Miss R (回答时间: {current_time}) ====\n")
        
        # Ask question to RAGFlow
        qa_system.ask_ragflow(question, start_time=current_time)


if __name__ == "__main__":
    main()
