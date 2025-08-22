#!/usr/bin/env python3
"""
Interactive Q&A system that uses RAGFlow SDK to answer questions.
Questions can be loaded from local dataset or input manually.
Based on the RAGFlow SDK example.
"""

import json
import os
import requests
from typing import List, Dict, Optional
from ragflow_sdk import RAGFlow, Agent

# ==================== 配置区域 ====================
# RAGFlow 连接配置
API_KEY  = "ragflow-g1ZGRhNjQyNTYzZTExZjA4ZjZiODY2Nj"
BASE_URL = "http://10.10.11.7:9380"
AGENT_ID = "673a2bda7bf911f0a13f2e823edec181"

# OpenAI 配置
OPENAI_API_KEY = "vllm"  # 请替换为您的OpenAI API密钥
OPENAI_BASE_URL = "http://10.10.11.7:11541/v1"  # 或者使用其他兼容的API端点
OPENAI_MODEL = "openai-mirror/gpt-oss-20b"  # 或 gpt-4, gpt-4-turbo 等

# 文件路径配置
INPUT_DATASET_PATH  = "/home/liufeng/sdk-ragflow/chunks_json/datasets-jd-0822.json"   # 输入问题数据集    /home/liufeng/sdk-ragflow/chunks_json/datasets-qa-0811.json
OUTPUT_ANSWERS_PATH = "/home/liufeng/sdk-ragflow/chunks_json/retrieval-resume-0820.json"  # 输出答案文件
# ==================================================

class OpenAIClient:
    """OpenAI API客户端类，模拟openai库的接口"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.chat = self.Chat(self)
    
    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.client.ChatCompletions(client)
    
    class ChatCompletions:
        def __init__(self, client):
            self.client = client
        
        def create(self, model: str, messages: List[Dict], temperature: float = 0.7, 
                  max_tokens: Optional[int] = None, stream: bool = False, **kwargs):
            """
            创建聊天完成请求
            
            Args:
                model: 模型名称
                messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
                temperature: 温度参数
                max_tokens: 最大token数
                stream: 是否流式响应
                **kwargs: 其他参数
            """
            url = f"{self.client.base_url}/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.client.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
                **kwargs
            }
            
            if max_tokens is not None:
                data["max_tokens"] = max_tokens
            
            try:
                response = requests.post(url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                
                if stream:
                    # 简化版流式处理，实际应用中需要处理Server-Sent Events
                    return self._handle_stream_response(response)
                else:
                    result = response.json()
                    return self._create_response_object(result)
                    
            except requests.exceptions.RequestException as e:
                print(f"OpenAI API请求失败: {e}")
                return self._create_error_response(str(e))
        
        def _handle_stream_response(self, response):
            """处理流式响应（简化版）"""
            # 这里简化处理，实际应用中需要解析SSE格式
            try:
                result = response.json()
                return self._create_response_object(result)
            except:
                return self._create_error_response("流式响应解析失败")
        
        def _create_response_object(self, result):
            """创建响应对象"""
            class Choice:
                def __init__(self, choice_data):
                    self.message = self.Message(choice_data.get('message', {}))
                    self.finish_reason = choice_data.get('finish_reason')
                    self.index = choice_data.get('index', 0)
                
                class Message:
                    def __init__(self, message_data):
                        self.role = message_data.get('role', 'assistant')
                        self.content = message_data.get('content', '')
            
            class Usage:
                def __init__(self, usage_data):
                    self.prompt_tokens = usage_data.get('prompt_tokens', 0)
                    self.completion_tokens = usage_data.get('completion_tokens', 0)
                    self.total_tokens = usage_data.get('total_tokens', 0)
            
            class Response:
                def __init__(self, result):
                    self.id = result.get('id', '')
                    self.object = result.get('object', 'chat.completion')
                    self.created = result.get('created', 0)
                    self.model = result.get('model', '')
                    self.choices = [Choice(choice) for choice in result.get('choices', [])]
                    self.usage = Usage(result.get('usage', {}))
            
            return Response(result)
        
        def _create_error_response(self, error_message):
            """创建错误响应"""
            class ErrorResponse:
                def __init__(self, error_msg):
                    self.choices = []
                    self.error = error_msg
                    
                    # 创建一个错误消息的choice
                    class ErrorChoice:
                        def __init__(self, error_msg):
                            self.message = self.ErrorMessage(error_msg)
                            self.finish_reason = "error"
                            self.index = 0
                        
                        class ErrorMessage:
                            def __init__(self, error_msg):
                                self.role = "assistant"
                                self.content = f"**ERROR**: {error_msg}"
                    
                    self.choices = [ErrorChoice(error_msg)]
            
            return ErrorResponse(error_message)


class RAGFlowQASystem:
    def __init__(self, api_key: str, base_url: str, agent_id: str, dataset_path: str = INPUT_DATASET_PATH,
                 openai_api_key: str = OPENAI_API_KEY, openai_base_url: str = OPENAI_BASE_URL):
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
        
        # Initialize OpenAI client
        self.openai_client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url)
        print(f"OpenAI客户端已初始化，使用端点: {openai_base_url}")
    
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
            first_token_elapsed_str = None
            response_elapsed_str = None
            
            for ans in self.session.ask(question, stream=True):
                if first_token and ans.content:
                    first_token_time = self.get_current_timestamp()
                    if start_time:
                        elapsed_time = self.calculate_time_diff(start_time, first_token_time)
                        first_token_elapsed_str = f"首个token耗时: {elapsed_time}"
                        print(f"[首个token时间: {first_token_time}] [{first_token_elapsed_str}]")
                    else:
                        print(f"[首个token时间: {first_token_time}]")
                    first_token = False
                print(ans.content[len(cont):], end='', flush=True)
                cont = ans.content
            
            # Add completion timestamp and calculate response time
            completion_time = self.get_current_timestamp()
            if first_token_time:
                response_time = self.calculate_time_diff(first_token_time, completion_time)
                response_elapsed_str = f"回答耗时: {response_time}"
                print(f"\n[回答完成时间: {completion_time}] [{response_elapsed_str}]")
            else:
                print(f"\n[回答完成时间: {completion_time}]")
            
            # Process with OpenAI if we got a valid answer
            openai_answer = None
            if cont.strip() and not cont.startswith("**ERROR"):
                print(f"\n{'='*60}")
                print("正在使用OpenAI大模型进一步处理答案...")
                print('='*60)
                openai_answer = self.process_with_openai(question, cont)
                print(f"\n==== OpenAI处理后的答案 ====")
                print(openai_answer)
                print('='*60)
            
            # Save to file if requested
            if save_to_file and cont.strip():
                self.save_qa_to_file(question, cont, first_token_elapsed_str, response_elapsed_str, openai_answer)
            
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
    
    def save_qa_to_file(self, question: str, answer: str, first_token_elapsed: str = None, 
                       response_elapsed: str = None, openai_answer: str = None):
        """Save question and answer to JSON file."""
        output_file = OUTPUT_ANSWERS_PATH
        
        # Create the entry
        qa_entry = {
            "question": question,
            "ragflow_answer": answer,
            "openai_answer": openai_answer,
            "timestamp": self.get_current_timestamp(),
            "first_token_elapsed": first_token_elapsed,
            "response_elapsed": response_elapsed
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
    
    def process_with_openai(self, question: str, ragflow_answer: str, 
                          custom_prompt: str = None) -> str:
        """
        使用OpenAI大模型进一步处理RAGFlow的答案
        
        Args:
            question: 原始问题
            ragflow_answer: RAGFlow的答案
            custom_prompt: 自定义提示词，如果不提供则使用默认提示词
        
        Returns:
            OpenAI处理后的答案
        """
        if not ragflow_answer or ragflow_answer.strip() == "":
            return "无法处理空的RAGFlow答案"
        
        # 默认提示词
        if custom_prompt is None:
            custom_prompt = """
你是一个 专业的招聘顾问，你的任务是根据用户输入的 职位需求，从 RAG 检索到的候选简历信息 中，筛选和推荐最符合岗位要求的简历。

请严格遵循以下规则：

只参考 RAG 提供的候选简历信息，不要凭空编造。

逐一对比职位需求与候选简历的技能、经验、教育背景，分析其匹配度。

输出时必须包含以下结构化内容：

候选人姓名，联系方式，电话号码

关键技能与岗位需求的匹配点

潜在不足或差距（如果有）

候选人简历的Document name

总体匹配度评分（0-100）

最终请按照 匹配度从高到低排序，并输出前 N 个最优简历（默认 N=5）。

Notes on scoring methodology

Experience – 5 pts per year (capped at 65 pts).

Education – BE = 5 pts, MSc = 10 pts.

Certifications – 5 pts each.

Projects / Impact – 5 pts per notable project or measurable outcome.

Base score – 50 pts for meeting the core requirement.

Maximum score – 100 pts.

保持 专业、客观、中立 的风格，不做主观情绪化评价。

RAG检索到的内容：{ragflow_answer}

用户输入的职位需求：{question}

"""
        
        # 格式化提示词
        formatted_prompt = custom_prompt.format(
            question=question,
            ragflow_answer=ragflow_answer
        )
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]
        
        try:
            print("\n[OpenAI] 正在调用大模型处理答案...")
            
            # 调用OpenAI API
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=131072
            )
            
            if response.choices and len(response.choices) > 0:
                openai_answer = response.choices[0].message.content
                print(f"[OpenAI] 处理完成")
                return openai_answer
            else:
                error_msg = "OpenAI未返回有效回答"
                print(f"[OpenAI] 错误: {error_msg}")
                return f"**OpenAI处理失败**: {error_msg}"
                
        except Exception as e:
            error_msg = f"OpenAI API调用失败: {e}"
            print(f"[OpenAI] 错误: {error_msg}")
            return f"**OpenAI处理失败**: {error_msg}"
    
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
