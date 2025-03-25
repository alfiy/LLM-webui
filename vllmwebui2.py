from flask import Flask, request, jsonify, send_from_directory
import gradio as gr
import chardet
from openai import OpenAI  # 假设我们使用一个类似的库来模拟OpenAI客户端
import os
import subprocess
import threading
import time
import requests


app = Flask(__name__)

# 设置VLLM API的API key和base URL
openai_api_key = os.getenv("OPENAI_API_KEY", "EMPTY")  # 如果VLLM API不需要API key，可以保持为空
openai_api_base = os.getenv("OPENAI_API_BASE", "http://0.0.0.0:8000/v1")  # 使用0.0.0.0以便可以从网络访问

# 初始化OpenAI风格的客户端
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def generate_text(prompt, content):
    """
    向VLLM服务发送请求以生成文本。
    """
    try:
        messages = [
            {"role": "system",
             "content": "You are Qwen, created by Alibaba Cloud. You are a helpful and friendly assistant."},
            {"role": "system",
             "content": "You should always provide accurate and reliable information. If you don't know the answer, you should say so. Avoid making up information or providing misleading answers."},
            {"role": "system",
             "content": "You should maintain a polite and professional tone in all your interactions. Use clear and concise language, and avoid using offensive or inappropriate content."},
            {"role": "system", "content": prompt},
            {"role": "assistant", "content": content},  # 注意这里的角色是assistant，可能需要根据实际情况调整
        ]
        
        
        chat_response = client.chat.completions.create(
            model="../qwen2.5/qwen2.5-14B-Instruct",
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            max_tokens=4096,  # 或者其他需要设置的参数
            extra_body={
                "repetition_penalty": 1.05,
            },
        )

        print("Chat response:", chat_response)  # 调试输出

        # 解析响应并返回生成的文本
        response_text = chat_response.choices[0].message.content if chat_response.choices else ""
        return response_text
    except Exception as e:
        return str(e)


def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']


def read_file_content(file_obj):
    """根据文件对象读取内容"""
    if hasattr(file_obj, 'data'):
        # 如果文件对象包含.data属性，则假设它是已经读取的字节流
        detected_encoding = chardet.detect(file_obj.data)['encoding']
        try:
            return file_obj.data.decode(detected_encoding)
        except UnicodeDecodeError:
            return file_obj.data.decode('gb2312', errors='ignore')
    else:
        # 假设这是一个本地文件路径
        detected_encoding = detect_encoding(file_obj.name)
        try:
            with open(file_obj.name, 'r', encoding=detected_encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_obj.name, 'r', encoding='gb2312', errors='ignore') as f:
                return f.read()


def process_uploaded_file(prompt, user_input, file_obj):
    """
    处理上传的文件，并使用其内容与用户输入结合作为模型的输入。
    """
    content = ""
    if file_obj is not None:
        # Gradio上传的文件是字典类型，包含文件名和文件数据
        # 使用read_file_content函数读取文件内容
        content = read_file_content(file_obj)
    else:
        content = user_input

    # 如果用户输入不为空，则将其附加到文件内容后面
    if user_input.strip():
        content += "\n" + user_input

    # 使用内容作为prompt的一部分调用generate_text
    response_text = generate_text(prompt, content)

    return response_text


# 创建Gradio接口
iface = gr.Interface(
    fn=lambda prompt, user_input, file,: process_uploaded_file(prompt, user_input, file),
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="User Input"),
        gr.File(label="Upload File (Optional)"),

    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Qwen Text Generator with VLLM",
    description="Generate text using the Qwen model with VLLM."
)


def run_vllm_api():
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "../qwen2.5/qwen2.5-14B-Instruct",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "2"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 打印VLLM服务启动日志
    def log_subprocess_output(pipe):
        for line in iter(pipe.readline, b''):
            print(line.decode().strip())

    threading.Thread(target=log_subprocess_output, args=(process.stdout,)).start()
    threading.Thread(target=log_subprocess_output, args=(process.stderr,)).start()

    wait_for_service_ready("http://localhost:8000/v1/models")
    return process

def wait_for_service_ready(url, timeout=120):
    """等待服务启动"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                print("Service is ready.")
                return True
            else:
                print(f"Service responded with status code {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        time.sleep(5)
    raise TimeoutError("Service did not become ready within the expected time.")


# 启动Gradio应用
def run_gradio():
    iface.launch(share=True, server_name='0.0.0.0', server_port=5000)  # 使用0.0.0.0以便可以从网络访问


# 创建API路由
@app.route('/generate', methods=['POST'])
def api_generate_text():
    """
    接收JSON输入，生成文本并返回结果。
    输入JSON格式:
    {
        "prompt": "提示词",
        "content": "用户输入的文本内容"
    }
    """
    try:
        data = request.json
        if not data or 'prompt' not in data or 'content' not in data:
            return jsonify({"error": "Invalid input. 'prompt' and 'content' are required."}), 400
        prompt = data['prompt']
        content = data['content']
        response_text = generate_text(prompt, content)
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    vllm_process = run_vllm_api()

    gradio_thread = threading.Thread(target=run_gradio)
    gradio_thread.start()

    try:
        app.run(host='0.0.0.0', port=5001)
    finally:
        vllm_process.terminate()
        gradio_thread.join()
