# %% md
# # 检索器可以用于检索文档，协助大模型回答问题
# %%
# 1. 通过langchain设置chatapi
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

# 2. 加载要索引的数据。
# 使用WebBaseLoader加载网页数据
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# 3. 构建索引：
# 将其索引到向量存储中。
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
# 构建我们的索引
# FAISS：轻量化的本地存储
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
# 分片
documents = text_splitter.split_documents(docs)
# 向量化
vector = FAISS.from_documents(documents, embeddings)

# 4. 创建文本检索链（prompt+llm），来接收文档和问题
# %% md
# 现在，我们在向量存储中索引了这些数据，我们将创建一个检索链。该链将接收一个输入问题，查找相关文档，然后将这些文档连同原始问题一起传递给LLM，并要求它回答原始问题。
# # 首先，让我们设置链来接收一个问题和检索到的文档，并生成一个回答。
# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
# 使用提示样本
prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:  
 <context>  
{context}  
</context>  
 Question: {input}""")
# 构建文本链
document_chain = create_stuff_documents_chain(llm, prompt)


# 5.1. 直接基于文本回答
# %% md
# 如果我们想的话，可以直接传递文档来运行它：
# %%
from langchain_core.documents import Document
document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})
# 5.2. 基于检索器来动态匹配最相关的文档，并传递给指定问题
# %% md
# 然而，我们希望文档首先来自我们刚刚设置的检索器。
# 这样，我们就可以使用检索器动态选择最相关的文档并将其传递给给定的问题。
# %%
from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
