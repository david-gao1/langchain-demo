from langchain_openai import ChatOpenAI

# 通过langchain设置chatapi
llm = ChatOpenAI()

# 不设置环境变量的方式 
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(api_key="...")
# 调用大模型
llm.invoke("how can langsmith help with testing?")

# 使用提示模板来指导其回答。 
# 提示模板将原始用户输入转换为更好的输入以供LLM使用。
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

# # 将promt和llm组成一个调用链
# chain = prompt | llm
# # 通过链来调用大模型，
# # 它仍然不会知道答案，但是它应该以更适合技术作家的方式回答。
# chain.invoke({"input": "how can langsmith help with testing?"})

# ChatModel的输出（因此，也是这个链的输出）是一个消息。
# 然而，使用字符串更方便。让我们添加一个简单的输出解析器将聊天消息转换为字符串。

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

# 将其添加到之前的链中：
chain = prompt | llm | output_parser

# 答案现在将是一个字符串（而不是ChatMessage）。
Output = chain.invoke({"input": "how can langsmith help with testing?"})
print(Output)
