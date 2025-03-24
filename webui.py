from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import gradio as gr
import os
import chardet
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

app = Flask(__name__)

# 加载模型和分词器
model_name = "../qwen2.5/qwen2.5-14B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_text(prompt, content):
    """
    生成文本的函数。
    """
    try:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful and friendly assistant."},
            {"role": "system", "content": "You should always provide accurate and reliable information. If you don't know the answer, you should say so. Avoid making up information or providing misleading answers."},
            {"role": "system", "content": "You should maintain a polite and professional tone in all your interactions. Use clear and concise language, and avoid using offensive or inappropriate content."},
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=4096,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response_text
    except Exception as e:
        return str(e)
    finally:
        cleaned_resources = []
        if 'model_inputs' in locals():
            del model_inputs
        if 'generated_ids' in locals():
            del generated_ids
        gc.collect()
        torch.cuda.empty_cache()

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

def process_uploaded_file(file_obj, user_input):
    """
    处理上传的文件，并使用其内容与用户输入结合作为模型的输入。
    """
    prompt = "请根据以下文档内容回答问题或提供帮助:"
    
    if file_obj is None:
        content = user_input
    else:
        content = read_file_content(file_obj)
    
    # 如果用户输入不为空，则将其附加到文件内容后面
    if user_input.strip():
        content += "\n" + user_input
    
    # 使用内容作为prompt的一部分调用generate_text
    response_text = generate_text(prompt, content)
    
    return response_text

iface = gr.Interface(
    fn=lambda file, user_input: process_uploaded_file(file, user_input),
    inputs=[
        gr.File(label="Upload File (Optional)"),
        gr.Textbox(label="User Input")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Qwen Text Generator",
    description="Generate text using the Qwen model with optional uploaded files."
)

if __name__ == '__main__':
    iface.launch(share=True, server_name='192.168.60.121', server_port=5000)
