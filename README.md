# HCI作业 3：Lab 2 基于 DeepSeek 的 RAG 信息检索系统

## 1. 环境配置
### 1.1 在命令行执行安装依赖
<img width="459" height="38" alt="image" src="https://github.com/user-attachments/assets/ead18e29-07ea-470b-9015-f19cf188ff95" />
requirements.txt 文件内容应包括：
<img width="503" height="567" alt="image" src="https://github.com/user-attachments/assets/6b4e0c6f-9445-461e-968e-14139f4b765d" />


### 1.2 设置API key
1. 访问 DeepSeek Platform 注册/登录账号。

2. 进入“API Keys”页面，创建一个新的 API Key 并复制保存。

3. 注意：DeepSeek API 需要先充值（最低1元）才能调用。本次实验消耗极少，远低于充值额度。
<img width="2558" height="814" alt="image" src="https://github.com/user-attachments/assets/fd3a7357-68df-47ed-823f-3bc75d841783" />

<img width="1404" height="1006" alt="image" src="https://github.com/user-attachments/assets/219e9896-c343-4f5b-aa68-05e01038fbef" />



### 1.3 embedding model下载
本项目使用 BAAI/bge-small-zh 作为嵌入模型。首次运行时，程序会自动从 Hugging Face Hub 下载。若网络不佳，可执行以下命令手动下载并缓存：

python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh')"

下载成功后，模型将缓存在本地 ~/.cache/huggingface/ 目录下。

## 2.知识库准备说明
### 2.1 使用的论文

本实验选取了三篇关于 DeepSeek 的中文学术论文作为知识库，以保证信息的权威性。

① 魏钰明, 贾开, 曾润喜, 等. DeepSeek突破效应下的人工智能创新发展与治理变革 [J]. 电子政务, 2025-02-26.

② 邓建鹏, 赵治松. DeepSeek的破局与变局：论生成式人工智能的监管方向 [J]. 新疆师范大学学报(哲学社会科学版), 2025-02-14.

③ 郭蕾蕾. 生成式人工智能驱动教育变革：机制、风险及应对——以DeepSeek为例 [J]. 重庆高教研究, 2025-03-10.

(扩展版中会增加一个 paper4.pdf 来演示多格式支持)

### 2.2 切分策略

为了保证语义完整并避免关键信息在切分边界处丢失，程序采用滑动窗口机制进行文本切分：

分片大小 (Chunk Size)：450 字符

步长 (Step)：400 字符

重叠大小 (Overlap)：50 字符 (由 chunk_size - step 计算得出)

这种策略确保了相邻文档块之间有50字符的重叠，维持了上下文的连贯性。

## 3. 运行指南

### 3.1 基础命令行版 (Task 3.1.2)

此版本实现了 RAG 的核心功能，并通过命令行进行交互。

步骤：

1. 打开 Deepseek_RAG_System.py，将代码开头的 MY_API_KEY 替换为你自己的密钥。

2. 确保 paper1.txt 等知识库文件在同一目录下。

3. 在终端中运行：


python Deepseek_RAG_System.py

4. 在 请输入您的问题: 提示符后输入问题，按回车获得回答。

5. 输入 quit 或 exit 退出程序。

<img width="436" height="185" alt="image" src="https://github.com/user-attachments/assets/07408499-1090-4939-95db-05c88b41e1a1" />


#### 3.1.1 场景一：基础知识问答

问题：DeepSeek 的核心创新点是什么？

结果：显示相关论文片段。

<img width="1706" height="600" alt="image" src="https://github.com/user-attachments/assets/b8d88c23-cb83-4128-afb6-419268ca35f4" />



#### 3.1.2 场景二：多文档交叉检索

问题：对比现有参考内容中对教育变革看法的异同。

结果：命中多个文件的片段。

<img width="1734" height="748" alt="image" src="https://github.com/user-attachments/assets/8226053d-207d-45f3-ad9b-89d7fee71fe7" />


#### 3.1.3 场景三：无法回答的情况（幻觉控制）

问题：今天的天气怎么样？

结果：系统虽检索到语义最接近的片段，但 LLM 准确判断出信息不匹配，输出预设的拒答语，有效防止了 AI 幻觉。

<img width="1734" height="520" alt="image" src="https://github.com/user-attachments/assets/9046923c-2aef-45a8-8c17-756833dff966" />

### 3.2 进阶网页对比版 (Task 3.1.3 & 3.1.4)

此版本使用 Gradio 构建了一个 Web 界面，可同时展示“仅 LLM”、“仅检索”、“RAG”三种模式的输出结果，并实现了缓存机制和PDF支持。

步骤：

1. 打开 htmlVersion.py，将 MY_API_KEY 替换为你自己的密钥。

2. 首次运行或HF网络不佳时：请注释掉代码开头的强制离线配置 (# os.environ['TRANSFORMERS_OFFLINE'] = '1')，开启代理后运行一次以完成模型下载。后续使用可恢复离线配置以加速启动。

3. 在终端中运行：

python htmlVersion.py

4. 等待终端输出 Running on local URL: http://127.0.0.1:7860 ，在浏览器中打开该地址即可访问。

5. 在输入框中输入问题，点击“开始对比测试”按钮，即可在三个面板中查看结果。

运行截图示例：

<img width="865" height="484" alt="image" src="https://github.com/user-attachments/assets/75ad05c9-5d30-42a8-b17e-ed55f3194369" />

<img width="865" height="472" alt="image" src="https://github.com/user-attachments/assets/3ad474b2-665a-46b1-99a6-2e75c245d63a" />


项目文件结构

2453601_牛奕洁_RAG作业/

├── Deepseek_RAG_System.py      # 基础命令行版主程序

├── htmlVersion.py              # 进阶Gradio网页版主程序

├── requirements.txt            # 项目依赖列表

├── README.md                   # 项目说明文档（本文件）

├── 实验报告.pdf                # 最终提交的实验报告

├── paper1.txt                  # 知识库文档1

├── paper2.txt                  # 知识库文档2

├── paper3.txt                  # 知识库文档3

└── paper4.pdf                  # 知识库文档4 (用于演示多格式支持)
