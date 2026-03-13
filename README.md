# IdiomRAG



成语检索：搜索嵌入文档中与输入含义最相近的成语



\## 功能特性



\- 检索返回格式：

&#x09;【成语】，【含义】，【出处】

&#x09;或：

&#x09;没有找到相关成语



\### 环境要求

详见requirements.txt,需用到cuda加速，推荐在conda虚拟环境中进行



\# 克隆仓库

git clone https://github.com/ziyang2121/IdiomRAG.git

cd rag2-proj



\# 安装依赖

pip install -r requirements.txt



\#快速开始

python rag.py



\#目录结构

rag2-proj/

├── README.md

├── requirements.txt

├── rag.py        # 源代码

├── models/     # llm和embedding 

├──docs/           # 嵌入文档

├──config.yaml #参数配置

└── ...                  #其他工具脚本、日志

