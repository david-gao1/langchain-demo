#%% md
# 到目前为止，我们创建的链只能回答单个问题。人们正在构建的LLM应用程序的主要类型之一是聊天机器人。那么我们如何将这个链转变为可以回答后续问题的链呢？
#%%
# # 检索器可以用于检索文档，协助大模型回答问题
from langchain_openai import ChatOpenAI

# 通过langchain设置chatapi 
llm = ChatOpenAI()

# 不设置环境变量的方式 ,非常不建议这么做
#from langchain_openai import ChatOpenAI

#llm = ChatOpenAI(api_key="...")
# 加载要索引的数据。为此，我们将使用WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()
# 将其索引到向量存储中。
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
# 构建我们的索引
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
# 现在，我们在向量存储中索引了这些数据，我们将创建一个检索链。该链将接收一个输入问题，查找相关文档，然后将这些文档连同原始问题一起传递给LLM，并要求它回答原始问题。
# 
# 首先，让我们设置链来接收一个问题和检索到的文档，并生成一个回答。
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
 
<context>
{context}
</context>
 
Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
# 如果我们想的话，可以直接传递文档来运行它：
from langchain_core.documents import Document

document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})
# 然而，我们希望文档首先来自我们刚刚设置的检索器。
# 这样，我们就可以使用检索器动态选择最相关的文档并将其传递给给定的问题。
from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# 现在，我们可以调用此链。这将返回一个词典-LLM的响应在answer键中。
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])




# 对话检索
# 1. prompt：使用提示模版
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

# 首先，我们需要一个可以传递给LLM来生成此搜索查询的提示
prompt = ChatPromptTemplate.from_messages([
    # chat_history 来保存对话
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "根据上面的对话，生成一个搜索查询来获取与对话相关的信息")
])
# 收集历史对话，并能够检索
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
#%%
from langchain_core.messages import HumanMessage, AIMessage

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
#%%
#  在这些示例中，每个步骤都是预先确定的。 最后我们将创建一个代理人 - 在这种情况下，LLM决定要采取的步骤。

# 添加搜索工具

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "搜索与LangSmith相关的信息。有关LangSmith的任何问题，您必须使用此工具！",
)
tools = [retriever_tool]
#%%
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# 加载线上的模版
prompt = hub.pull("hwchase17/openai-functions-agent")

# 您需要设置OPENAI_API_KEY环境变量，或者将其作为参数`api_key`传递。
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# 创建基于openai的agent（利用到了function calling）
# 检索对话的工具，
# 官方的prompt
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 调用代理人并查看它的响应
agent_executor.invoke({"input": "langsmith如何帮助测试？"})

#%%
agent_executor.invoke({"input": "旧金山的天气如何？"})
#%%
chat_history = [HumanMessage(content="LangSmith可以帮助测试我的LLM应用程序吗？"), AIMessage(content="可以！")]
agent_executor.invoke({
    "chat_history": chat_history,
    "input": "告诉我如何进行"
})
#%%
