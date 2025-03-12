import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
from io import BytesIO

# 设置模型和标准化器的路径
MODEL_URL = 'https://raw.githubusercontent.com/chu623524/323457/main/RF.pkl'
SCALER_URL = 'https://raw.githubusercontent.com/chu623524/323457/main/scaler.pkl'

# 下载模型和标准化器
@st.cache_resource
def load_model():
    # 下载模型
    model_response = requests.get(MODEL_URL)
    model_data = BytesIO(model_response.content)
    model = joblib.load(model_data)

    # 下载标准化器
    scaler_response = requests.get(SCALER_URL)
    scaler_data = BytesIO(scaler_response.content)
    scaler = joblib.load(scaler_data)

    return model, scaler

model, scaler = load_model()

# 定义类别变量选项
文化程度_options = {0: "小学及以下", 1: "初中", 2: "高中/中专", 3: "大专及以上"}
饮酒状态_options = {0: "不饮酒", 1: "偶尔", 2: "经常"}
家庭月收入_options = {0: "0-2000", 1: "2001-5000", 2: "5001-8000", 3: "8000+"}
创伤时恐惧程度_options = {0: "无", 1: "轻度", 2: "中度", 3: "重度"}
吸烟状态_options = {0: "不吸烟", 1: "偶尔", 2: "经常"}

# 设置Web界面
st.title("PTSD 预测系统")
st.write("创伤后3个月PTSD预测")

# 获取用户输入的特征
ASDS = st.number_input("ASDS (分)", value=50.0, help="单位: 分，最高95分")
文化程度 = st.selectbox("文化程度", options=list(文化程度_options.keys()), format_func=lambda x: 文化程度_options[x])
饮酒状态 = st.selectbox("饮酒状态", options=list(饮酒状态_options.keys()), format_func=lambda x: 饮酒状态_options[x])
舒张压 = st.number_input("舒张压 (mmHg)", value=80.0, help="单位: mmHg")
家庭月收入 = st.selectbox("家庭月收入", options=list(家庭月收入_options.keys()), format_func=lambda x: 家庭月收入_options[x])
中性粒细胞绝对值 = st.number_input("中性粒细胞绝对值 (10^9/L)", value=50.0, help="单位: 10^9/L，范围 0-100")
氯 = st.number_input("氯 (mmol/L)", value=50.0, help="单位: mmol/L，范围 0-150")
吸烟状态 = st.selectbox("吸烟状态", options=list(吸烟状态_options.keys()), format_func=lambda x: 吸烟状态_options[x])
焦虑评分 = st.number_input("焦虑评分", value=50.0, help="单位: 分，范围 0-100")
AST_ALT = st.number_input("AST/ALT", value=0.5, help="范围 0 - 100")
A_G = st.number_input("A/G", value=0.5, help="范围 0 - 3.0")
血红蛋白 = st.number_input("血红蛋白 (g/dL)", value=12.0, help="单位: g/dL")
心理负担 = st.number_input("心理负担", value=50.0, help="单位: 分")
单核细胞绝对值 = st.number_input("单核细胞绝对值 (10^9/L)", value=50.0, help="单位: 10^9/L")
脉搏 = st.number_input("脉搏 (bpm)", value=70.0, help="单位: bpm")
创伤时恐惧程度 = st.selectbox("创伤时恐惧程度", options=list(创伤时恐惧程度_options.keys()), format_func=lambda x: 创伤时恐惧程度_options[x])

# 创建一个字典来存储所有输入的特征
input_data = {
    'ASDS': ASDS,
    '文化程度': 文化程度,
    '饮酒状态': 饮酒状态,
    '舒张压': 舒张压,
    '家庭月收入': 家庭月收入,
    '中性粒细胞绝对值': 中性粒细胞绝对值,
    '氯': 氯,
    '吸烟状态': 吸烟状态,
    '焦虑评分': 焦虑评分,
    'AST/ALT': AST_ALT,
    'A/G': A_G,
    '血红蛋白': 血红蛋白,
    '心理负担': 心理负担,
    '单核细胞绝对值': 单核细胞绝对值,
    '脉搏': 脉搏,
    '创伤时恐惧程度': 创伤时恐惧程度
}

# 预测按钮
if st.button("预测"):
    # 将输入数据转换为 NumPy 数组
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    # 标准化
    input_scaled = scaler.transform(input_array)

    # 进行预测
    prediction_prob = model.predict_proba(input_scaled)[0, 1]  # PTSD 的概率
    if prediction_prob > 0.5:
        prediction = f"根据我们的模型，你患PTSD的风险很高。声明: 该预测仅供参考，我们建议您结合专业医生的意见进行判断。"
    else:
        prediction = f"根据我们的模型，你患PTSD的风险较低。声明: 该预测仅供参考，我们建议您结合专业医生的意见进行判断。"
    
    # 输出结果
    st.write(f"**PTSD 概率:** {prediction_prob:.4f}")
    st.write(f"**预测结果:** {prediction}")
