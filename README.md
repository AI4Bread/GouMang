
# XiXiLM 

<div align="center">

<img src="https://github.com/AI4Bread/GouMang/blob/main/assets/goumang_logoallnew.png?raw=true" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
<!--     <b><font size="5">XiXiLM</font></b> -->
    <sup>
      <a href="http://www.ai4bread.com">
      </a>
    </sup>
    <div>&nbsp;</div>
  </div>
  

[💻Github Repo](https://github.com/AI4Bread/GouMang) • [🤔Reporting Issues](https://github.com/AI4Bread/GouMang/issues) • [📜Technical Report](https://github.com/AI4Bread)

</div>

<p align="center">
    👋 join us on <a href="https://github.com/AI4Bread/GouMang" target="_blank">Github</a>
</p>



## Introduction

XiXiLM（GouMang LLM） has open-sourced a 7 billion parameter base model and a chat model tailored for agricultural scenarios. The model has the following characteristics:

- **200K Context window**: Nearly perfect at finding needles in the haystack with 200K-long context, with leading performance on long-context tasks like LongBench and L-Eval. Try it with [LMDeploy](https://github.com/InternLM/lmdeploy) for 200K-context inference.

- **Outstanding comprehensive performance**: Significantly better than the last generation in all dimensions, especially in reasoning, math, code, chat experience, instruction following, and creative writing, with leading performance among open-source models in similar sizes. In some evaluations, InternLM2-Chat-20B may match or even surpass ChatGPT (GPT-3.5).

## XiXiLM-Qwen-14B


**Limitations:** Although we have made efforts to ensure the safety of the model during the training process and to 
encourage the model to generate text that complies with ethical and legal requirements, the model may still produce unexpected 
outputs due to its size and probabilistic generation paradigm. For example, the generated responses may contain biases, discrimination, 
or other harmful content. Please do not propagate such content. We are not responsible for any consequences resulting from the 
dissemination of harmful information.

### Import from Transformers

To load the XiXiLM model using Transformers, use the following code:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("AI4Bread/XiXi_Qwen_base_14b", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("AI4Bread/XiXi_Qwen_base_14b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
# Hello! How can I help you today?
response, history = model.chat(tokenizer, "马铃薯育种有什么注意事项？需要注意什么呢？", history=history)
print(response)
```

The responses can be streamed using `stream_chat`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "AI4Bread/XiXi_Qwen_base_14b"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = model.eval()
length = 0
for response, history in model.stream_chat(tokenizer, "Hello", history=[]):
    print(response[length:], flush=True, end="")
    length = len(response)
```


## Deployment

### LMDeploy

LMDeploy is a toolkit for compressing, deploying, and serving LLM, developed by the MMRazor and MMDeploy teams.

```bash
pip install lmdeploy
```

Or you can launch an OpenAI compatible server with the following command:

```bash
lmdeploy serve api_server internlm/internlm2-chat-7b --model-name internlm2-chat-7b --server-port 23333 
```

Then you can send a chat request to the server:

```bash
curl http://localhost:23333/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "internlm2-chat-7b",
    "messages": [
    {"role": "system", "content": "你是一个专业的农业专家"},
    {"role": "user", "content": "马铃薯种植的时候有哪些注意事项？"}
    ]
    }'
```

The output be like:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/658a3c4cbbb04840e3ce7e2c/NPdRr5Y5l5E0m0URCVZ1f.png)



Find more details in the [LMDeploy documentation](https://lmdeploy.readthedocs.io/en/latest/)

### vLLM

Launch OpenAI compatible server with `vLLM>=0.3.2`:

```bash
pip install vllm
```

```bash
python -m vllm.entrypoints.openai.api_server --model internlm/internlm2-chat-7b --served-model-name internlm2-chat-7b --trust-remote-code
```

Then you can send a chat request to the server:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "internlm2-chat-7b",
    "messages": [
    {"role": "system", "content": "You are a professional agriculture expert."},
    {"role": "user", "content": "Introduce potato farming to me."}
    ]
    }'
```

Find more details in the [vLLM documentation](https://docs.vllm.ai/en/latest/index.html)

### Convert  lmdeploy TurboMind

```bash
# Converting Model to TurboMind (FastTransformer Format)
lmdeploy convert internlm-chat-7b /path/to/XiXiLM
```

Here, we will use our pre-trained model file and execute the conversion in the user's root directory, as shown below.

```bash
lmdeploy convert internlm2-chat-7b /root/autodl-tmp/agri_intern/XiXiLM --tokenizer-path ./GouMang/tokenizer.json
```

After execution, a workspace folder will be generated in the current directory. 
This folder contains the necessary files for TurboMind and Triton "Model Inference." as shown below:


![image/png](https://cdn-uploads.huggingface.co/production/uploads/658a3c4cbbb04840e3ce7e2c/CqdwhshIL8xxjog_WD_St.png)


### Chat Locally

```bash
lmdeploy chat turbomind ./workspace
```

### TurboMind Inference + API Service

In the previous section, we tried starting the Client directly using the command line. Now, we will attempt to use lmdeploy for service deployment.

The "Model Inference/Service" currently offers two service deployment methods: TurboMind and TritonServer. In this case, the Server is either TurboMind or TritonServer, and the API Server can provide external API services. We recommend using TurboMind.

First, start the service with the following command:


```bash
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server-port 23333 \
	--instance_num 64 \
	--tp 1
```

In the above parameters, `server_name` and `server_port` indicate the service address and port, respectively. The `tp` parameter, as mentioned earlier, stands for Tensor Parallelism. The remaining parameter, instance_num, represents the number of instances and can be understood as the batch size. After execution, it will appear as shown below.

## Web Service Startup Method 1:

###  Starting the Service with Gradio

This section demonstrates using Gradio as a front-end demo.

> Since Gradio requires local access to display the interface,
> you also need to forward the data to your local machine via SSH. The command is as follows:
>
> ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p <your ssh port>

#### --TurboMind Service as the Backend

The API Server is started the same way as in the previous section. Here, we directly start Gradio as the front-end.

```bash
# Gradio+ApiServer. The Server must be started first, and Gradio acts as the Client
lmdeploy serve gradio http://0.0.0.0:23333 --server-port 6006
```

#### --Other way(Recommended!!!)

Of course, Gradio can also connect directly with TurboMind, as shown below

```bash
# Gradio+Turbomind(local)
lmdeploy serve gradio ./workspace
```

You can start Gradio directly. In this case, there is no API Server, and TurboMind communicates directly with Gradio.

## Web Service Startup Method 2:

### Starting the Service with Streamlit

```bash
pip install streamlit==1.24.0
```

Download the [GouMang](https://huggingface.co/AI4Bread/GouMang) project model (please Star if you like it)


Replace the model path in `web_demo.py` with the path where the downloaded parameters of `GouMang` are stored 

Run the `web_demo.py` file in the directory, and after entering the following command, [**check this tutorial 5.2 for local port configuration**](https://github.com/InternLM/tutorial/blob/main/helloworld/hello_world.md#52-%E9%85%8D%E7%BD%AE%E6%9C%AC%E5%9C%B0%E7%AB%AF%E5%8F%A3)，to map the port to your local machine. Enter `http://127.0.0.1:6006` in your local browser. 

```
streamlit run /root/personal_assistant/code/InternLM/web_demo.py --server.address 127.0.0.1 --server.port 6006
```

Note: The model will load only after you open the `http://127.0.0.1:6006` page in your browser. 
Once the model is loaded, you can start conversing with GouMang like this.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/658a3c4cbbb04840e3ce7e2c/VcuSpAKrRGY1HP1mwLGI6.png)


## Open Source License

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, please fill in the [申请表（中文）](https://wj.qq.com/s2/14897739/e871/). For other questions or collaborations, please contact <laiyifu@xjtu.edu.cn>.

## Citation



## 简介

XiXiLM ，即西西大模型（又名：句芒大模型），开源了面向农业问答的大模型。模型具有以下特点：

- 有效支持20万字超长上下文：模型在20万字长输入中几乎完美地实现长文“大海捞针”，而且在 LongBench 和 L-Eval 等长文任务中的表现也达到开源模型中的领先水平。 可以通过 [LMDeploy](https://github.com/InternLM/lmdeploy) 尝试20万字超长上下文推理。
- 综合性能全面提升：各能力维度相比上一代模型全面进步，在推理、数学、代码、对话体验、指令遵循和创意写作等方面的能力提升尤为显著，综合性能达到同量级开源模型的领先水平，在重点能力评测上 InternLM2-Chat-20B 能比肩甚至超越 ChatGPT （GPT-3.5）。

## XiXiLM-Qwen-14B


**局限性：** 尽管在训练过程中我们非常注重模型的安全性，尽力促使模型输出符合伦理和法律要求的文本，但受限于模型大小以及概率生成范式，模型可能会产生各种不符合预期的输出，例如回复内容包含偏见、歧视等有害内容，请勿传播这些内容。由于传播不良信息导致的任何后果，本项目不承担责任。

### 通过 Transformers 加载

通过以下的代码加载 InternLM2 7B Chat 模型

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("AI4Bread/XiXi_Qwen_base_14b", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("AI4Bread/XiXi_Qwen_base_14b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
# Hello! How can I help you today?
response, history = model.chat(tokenizer, "马铃薯育种有什么注意事项？需要注意什么呢？", history=history)
print(response)
```

如果想进行流式生成，则可以使用 `stream_chat` 接口：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "AI4Bread/XiXi_Qwen_base_14b"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = model.eval()
length = 0
for response, history in model.stream_chat(tokenizer, "马铃薯育种有什么注意事项？需要注意什么呢？", history=[]):
    print(response[length:], flush=True, end="")
    length = len(response)
```

## 部署

### LMDeploy

LMDeploy 由 MMDeploy 和 MMRazor 团队联合开发，是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。

```bash
pip install lmdeploy
```

你可以使用以下命令启动兼容 OpenAI API 的服务:

```bash
lmdeploy serve api_server internlm/internlm2-chat-7b --server-port 23333
```

然后你可以向服务端发起一个聊天请求:

```bash
curl http://localhost:23333/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "internlm2-chat-7b",
    "messages": [
    {"role": "system", "content": "你是一个专业的农业专家"},
    {"role": "user", "content": "马铃薯种植的时候有哪些注意事项？"}
    ]
    }'
```

更多信息请查看 [LMDeploy 文档](https://lmdeploy.readthedocs.io/en/latest/)

### vLLM

使用`vLLM>=0.3.2`启动兼容 OpenAI API 的服务:

```bash
pip install vllm
```

```bash
python -m vllm.entrypoints.openai.api_server --model internlm/internlm2-chat-7b --trust-remote-code
```

然后你可以向服务端发起一个聊天请求:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "internlm2-chat-7b",
    "messages": [
    {"role": "system", "content": "你是一个专业的农业专家."},
    {"role": "user", "content": "请给我介绍一下马铃薯育种."}
    ]
    }'
```

更多信息请查看 [vLLM 文档](https://docs.vllm.ai/en/latest/index.html)


## 网页服务启动方式1:

###  Gradio 方式启动服务

这一部分主要是将 Gradio 作为前端 Demo 演示。在上一节的基础上，我们不执行后面的 `api_client` 或 `triton_client`，而是执行 `gradio`。
请参考[LMDeploy](#lmdeploy)部分获取详细信息。

> 由于 Gradio 需要本地访问展示界面，因此也需要通过 ssh 将数据转发到本地。命令如下：
>
> ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p <你的 ssh 端口号>

#### --TurboMind 服务作为后端

直接启动作为前端的 Gradio。

```bash
# Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client
lmdeploy serve gradio http://0.0.0.0:23333 --server-port 6006
```

#### --其他方式(推荐!!!)

当然，Gradio 也可以直接和 TurboMind 连接，如下所示。

```bash
# Gradio+Turbomind(local)
lmdeploy serve gradio ./workspace
```

可以直接启动 Gradio，此时没有 API Server，TurboMind 直接与 Gradio 通信。

## 网页服务启动方式2:

### Streamlit 方式启动服务：

下载 [GouMang](https://huggingface.co/AI4Bread/GouMang) 项目模型（如果喜欢请给个 Star）

将 `web_demo.py` 中的模型路径替换为下载的 `GouMang` 参数存储路径

在目录中运行 `web_demo.py` 文件，并在输入以下命令后，[**查看本教程 5.2 以配置本地端口**](https://github.com/InternLM/tutorial/blob/main/helloworld/hello_world.md#52-%E9%85%8D%E7%BD%AE%E6%9C%AC%E5%9C%B0%E7%AB%AF%E5%8F%A3)，将端口映射到本地。在本地浏览器中输入 `http://127.0.0.1:6006`。

```
streamlit run /root/personal_assistant/code/InternLM/web_demo.py --server.address 127.0.0.1 --server.port 6006
```

注意：只有在浏览器中打开 `http://127.0.0.1:6006` 页面后，模型才会加载。
模型加载完成后，您就可以开始与 西西（句芒） 进行对话了。

## 开源许可证

本仓库的代码依照 Apache-2.0 协议开源。模型权重对学术研究完全开放，也可申请免费的商业使用授权（[申请表](https://wj.qq.com/s2/14897739/e871/)）。其他问题与合作请联系 <laiyifu@xjtu.edu.cn>。

## 引用
