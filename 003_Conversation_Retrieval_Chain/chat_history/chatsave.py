#%%
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. 创建chat链
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个bi分析专家，用于校验和完善用户是否询问有关bi相关问题，"
     "用户的问题中要包含：指标集、指标、时间范围"
     "如不是则引导用户，或者基于历史对话补全用户问题"
     "如果是则直接返回用户输入的问题，不需做任何解释"),
    MessagesPlaceholder(variable_name="chat_history"),  # 接收历史对话
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# 2. 对话存储管理
### 有状态地管理聊天历史记录 ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
#%%
print(with_message_history.invoke(
    {"input": "360浏览器近一周的uv"},
    config={"configurable": {"session_id": "1"}},  # 一个对话（多个聊天）中只维护这一个session_id
))

#%%
print(with_message_history.invoke(
    {"input": "pv是多少"},
    config={"configurable": {"session_id": "1"}},
))

#%%
with_message_history.invoke(
    {"input": "uv是多少"},
    config={"configurable": {"session_id": "1"}},
)
#%%
with_message_history.invoke(
    {"input": "近一周"},
    config={"configurable": {"session_id": "1"}},
)