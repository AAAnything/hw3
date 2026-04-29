# HCI作业 3：Lab 2 基于 DeepSeek 的 RAG 信息检索系统

## 1. 环境配置
### 1.1 在命令行执行安装依赖
<img width="459" height="38" alt="image" src="https://github.com/user-attachments/assets/ead18e29-07ea-470b-9015-f19cf188ff95" />

### 1.2 设置API key
这次作业我采用了www.platform.deepseek.com平台的付费服务。
创建一个新的API key，复制保存。
<img width="2558" height="814" alt="image" src="https://github.com/user-attachments/assets/fd3a7357-68df-47ed-823f-3bc75d841783" />

充值1元来做作业。
<img width="1404" height="1006" alt="image" src="https://github.com/user-attachments/assets/219e9896-c343-4f5b-aa68-05e01038fbef" />

做作业用的钱其实远远不到1元，但是这个是最小充值金额。

### 1.3 embedding model下载
模型名称：BAAI/bge-small-zh (北京智源人工智能研究院开发的中文小模型)。
下载库：使用 sentence-transformers 库。
下载触发代码：
model = SentenceTransformer('BAAI/bge-small-zh')

下载机制：
自动检索：当程序运行到这一行时，会自动连接到 Hugging Face Hub 服务器。
本地缓存：模型文件会自动下载到你电脑的默认缓存文件夹（通常在 Windows 的 C:\Users\用户名\.cache\huggingface\hub）。
静默加载：一旦下载过一次，下次运行代码时会直接从本地硬盘读取，不再需要联网下载。

## 2.知识库准备说明
### 2.1 使用的论文
我选取了中国知网上引用次数最多的文章，以此保证论文权威性。
① 魏钰明, 贾开, 曾润喜, 等. DeepSeek突破效应下的人工智能创新发展与治理变革 [J]. 电子政务, 2025-02-26.
② 邓建鹏, 赵治松. DeepSeek的破局与变局：论生成式人工智能的监管方向 [J]. 新疆师范大学学报(哲学社会科学版), 2025-02-14.
③ 郭蕾蕾. 生成式人工智能驱动教育变革：机制、风险及应对——以DeepSeek为例 [J]. 重庆高教研究, 2025-03-10.

### 2.2 切分策略
程序采用 滑动窗口（Sliding Window） 机制进行文本切分，以保证上下文的连续性：
分片大小 (Chunk Size): 450 字符
重叠大小 (Overlap): 50 字符
步长: 400 字符

相关代码：
        # 策略：滑动窗口切分 (Chunk Size: 450, Overlap: 50)
        
        chunks = [text[i:i + 450] for i in range(0, len(text), 400)]
        
        for i, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()
            collection.add(
                ids=[f"{file_name}_{i}"],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[{"source": file_name}]
            )
    print(f"成功加载 {len(FILES)} 份文档，书架已就绪！")

## 3. 运行指南
### 3.1 基础命令行版 (Task 3.1.2)
首先需要确认已经执行了第一部分的环境配置。
将Deepseek_RAG_System.py中的MY_API_KEY改成自己的密钥。
在终端执行 Python 脚本：

Bash
python Deepseek_RAG_System.py
这一部分采用 CLI 命令行交互界面。
启动后，系统会自动完成文档向量化并存入内存数据库。
在 请输入您的问题: 提示符后输入提问内容。
输入 quit 或 exit 即可退出系统。
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
打开 htmlVersion.py，将第 9 行的 MY_API_KEY 替换为您自己的 DeepSeek API Key。
由于模型文件较大且访问 Hugging Face 可能受限，系统已配置离线加载模式。若您的环境中未下载模型，请先开启代理并注释掉代码顶部的 os.environ['TRANSFORMERS_OFFLINE'] = '1' 后运行一次。
启动命令：python htmlVersion.py
程序启动后，在浏览器访问 (http://127.0.0.1:7860) 即可进入 RAG 对比实验平台。
<img width="865" height="484" alt="image" src="https://github.com/user-attachments/assets/75ad05c9-5d30-42a8-b17e-ed55f3194369" />


#### 3.2.1  运行截图示例
<img width="865" height="472" alt="image" src="https://github.com/user-attachments/assets/3ad474b2-665a-46b1-99a6-2e75c245d63a" />


