import os
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

import tcvectordb
from tcvectordb.model.document import Document, Filter, SearchParams




MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 设置页面标题、图标和布局
st.set_page_config(
    page_title="我的AI知识库",
    page_icon=":robot:",
    layout="wide"
)

def searchTvdb(txt):
    conn_params = {
        'url':'http://lb-p7oj0itn-j5gawvui3dz4lcn2.clb.ap-beijing.tencentclb.com:50000',
        'key':'lgrdYzlMsmc1GhVtcmilXhZevJBfFlId919EOvaE',
        'username':'root',
        'timeout':20
        }

    vdb_client = tcvectordb.VectorDBClient(
            url=conn_params['url'],
            username=conn_params['username'],
            key=conn_params['key'],
            timeout=conn_params['timeout'],
        )
    db_list =  vdb_client.list_databases()

    db = vdb_client.database('crawlingdb')
    coll = db.collection('tencent_knowledge')
    embeddingItems = [txt]
    search_by_text_res = coll.searchByText(embeddingItems=embeddingItems,limit=3, params=SearchParams(ef=100))
    # print_object(search_by_text_res.get('documents'))
    # print(search_by_text_res.get('documents'))
    return search_by_text_res.get('documents')

def listToString(doc_lists):
    str =""
    for i, docs in enumerate(doc_lists):
        for doc in docs:
          str=str+doc["text"]
    return str

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if 'cuda' in DEVICE:  # AMD, NVIDIA GPU can use Half Precision
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
    else:  # CPU, Intel GPU and other GPU can use Float16 Precision Only
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE).eval()
    # 多显卡支持,使用下面两行代替上面一行,将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)
    return tokenizer, model

# 加载Chatglm3的model和tokenizer
tokenizer, model = get_model()

# 初始化历史记录和past key values
if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

# 

def on_mode_change():
    mode = st.session_state.dialogue_mode
    text = f"已切换到 {mode} 模式。"
    st.session_state.history = []
    st.session_state.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
       
    st.toast(text)

dialogue_mode = st.sidebar.selectbox("请选择对话模式",
                                ["腾讯云知识库对话",
                                "正常LLM对话(支持历史)",
                                ],
                                on_change=on_mode_change,
                                key="dialogue_mode",
                                )

# 设置max_length、top_p和temperature
max_length = st.sidebar.slider("max_length", 0, 32768, 8000, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.8, step=0.01)

 

# 清理会话历史
buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

# 渲染聊天历史记录
for i, message in enumerate(st.session_state.history):
    print(message)
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

# 输入框和输出框
with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

# 获取用户输入
prompt_text = st.chat_input("请输入您的问题")

# 如果用户输入了内容,则生成回复
if prompt_text:
    # model = model.eval()
    # template = "对引号中内容进行分词：\""+prompt_text+"\""
    # response, history = model.chat(tokenizer, template,top_p=top_p,temperature=temperature, history=[])
    # print(response)
    
    mode = st.session_state.dialogue_mode
    template_data=""
    if mode =="腾讯云知识库对话":
        result = searchTvdb(prompt_text)
        str = listToString(result)
        # print(str)
        template_data = "请按照\""+prompt_text+"\"进行总结,内容是："+str
        template_data = template_data[:20000]
    else:
        template_data =prompt_text
    
    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    past_key_values = st.session_state.past_key_values
    history = []
    for response, history, past_key_values in model.stream_chat(
        tokenizer,
        template_data,
        history,
        past_key_values=past_key_values, 
        max_length=max_length, 
        top_p=top_p,
        temperature=temperature,
        return_past_key_values=True,
    ):
        # print(response)
        message_placeholder.markdown(response)
    
    endString = ""
   
    # 更新历史记录和past key values
    if mode != "腾讯云知识库对话":
        st.session_state.history = history
        st.session_state.past_key_values = past_key_values
    else:
      for i,doc in enumerate(result[0]):
        # print(doc)
        endString = endString+"\n\n"+doc["title"]+"     "+doc["id"]
      response=response+"\n\n参考链接：\n\n\n"+endString
    message_placeholder.markdown(response)