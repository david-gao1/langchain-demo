#%% md
# # 检索器可以用于检索文档，协助大模型回答问题
#%%
from langchain_openai import ChatOpenAI
# 通过langchain设置chatapi 
llm = ChatOpenAI()

# 不设置环境变量的方式 ,非常不建议这么做
#from langchain_openai import ChatOpenAI
 
#llm = ChatOpenAI(api_key="...")
#%%
# 加载要索引的数据。为此，我们将使用WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()
#%%
# 将其索引到向量存储中。
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
#%%
# 构建我们的索引
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
#%% md
# 现在，我们在向量存储中索引了这些数据，我们将创建一个检索链。该链将接收一个输入问题，查找相关文档，然后将这些文档连同原始问题一起传递给LLM，并要求它回答原始问题。
# 
# 首先，让我们设置链来接收一个问题和检索到的文档，并生成一个回答。
#%%
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
 
<context>
{context}
</context>
 
Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
#%% md
# 如果我们想的话，可以直接传递文档来运行它：
#%%
from langchain_core.documents import Document
 
document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})
#%% md
# 然而，我们希望文档首先来自我们刚刚设置的检索器。
# 这样，我们就可以使用检索器动态选择最相关的文档并将其传递给给定的问题。
#%%
from langchain.chains import create_retrieval_chain
 
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
#%% md
# 现在，我们可以调用此链。这将返回一个词典-LLM的响应在answer键中。
#%%
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
 
# LangSmith提供了几个功能，可以帮助进行测试:...
#%%
