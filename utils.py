from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from pydantic import SecretStr
import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from langchain_experimental.agents import create_pandas_dataframe_agent

base_url = 'https://api.openai-hk.com/v1'
api_key = 'hk-z8yz1o1000056196f1a2032989e330e608278c706fad5a66'
client = OpenAI(base_url=base_url, api_key=api_key)


def get_response(*, messages):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
    )
    return response.choices[0].message.content


model = ChatOpenAI(
    model='gpt-4o-mini',
    base_url='https://api.openai-hk.com/v1/',
    # api_key=SecretStr('hk-z8yz1o1000056196f1a2032989e330e608278c706fad5a66'),
    api_key=api_key,
    temperature=0.7
)


zp_embeddings = OpenAIEmbeddings(
    model='embedding-3',
    base_url='https://open.bigmodel.cn/api/paas/v4',
    api_key=SecretStr('4630e11e632e4274a7287471d0a32d81.BPL4ctRqpJmQJo7d'),
)


PROMPT_PREFIX = """
你是一位数据分析助手，你的回应内容取决于用户的请求内容，请按照下面的步骤处理用户请求：
一个正确的过程为(一定要注意json格式):
例1：
Thought: 用户询问李白的去世时间。我已经知道李白于762年去世，因此我将准备一个纯文字回答。
Action: python_repl_ast
Action Input: {"answer": "李白于762年去世。"}


1. 思考阶段 (Thought) ：先分析用户请求类型（文字回答/表格/图表），并验证数据类型是否匹配。
2. 行动阶段 (Action) ：根据分析结果选择以下严格对应的格式，一定要符合是json格式这个前提。
   • 纯文字回答: 

     {"answer": "不超过50个字符的明确答案"}

   • 表格数据：  

     {"table":{"columns":["列名1", "列名2", ...], "data":[["第一行值1", "值2", ...], ["第二行值1", "值2", ...]]}}

   • 柱状图 (Bar Chart): 

     {"bar": {
        "x": ["类别1", "类别2", ...],
        "y": [值1, 值2, ...],
        "title": "图表标题",
        "x_label": "X轴标签",
        "y_label": "Y轴标签"
     }}

   • 折线图 (Line Chart): 

     {"line": {
        "x": ["时间1", "时间2", ...],
        "y": [值1, 值2, ...],
        "title": "图表标题",
        "x_label": "X轴标签",
        "y_label": "Y轴标签"
     }}

   • 饼图 (Pie Chart): 

     {"pie": {
        "labels": ["类别1", "类别2", ...],
        "values": [值1, 值2, ...],
        "title": "图表标题"
     }}

   • 散点图 (Scatter Plot): 

     {"scatter": {
        "x": [x1, x2, ...],
        "y": [y1, y2, ...],
        "title": "图表标题",
        "x_label": "X轴标签",
        "y_label": "Y轴标签"
     }}

   • 热力图 (Heatmap): 

     {"heatmap": {
        "data": [[值11, 值12, ...], [值21, 值22, ...], ...],
        "x_labels": ["列1", "列2", ...],
        "y_labels": ["行1", "行2", ...],
        "title": "图表标题"
     }}

   • 箱线图 (Box Plot): 

     {"boxplot": {
        "data": [值1, 值2, ...] 或 
        "groups": {
          "组1": [值1, 值2, ...],
          "组2": [值3, 值4, ...],
          ...
        },
        "title": "图表标题",
        "x_label": "X轴标签",
        "y_label": "Y轴标签"
     }}

3. 格式校验要求
   • 字符串值必须使用英文双引号
   • 数值类型不得添加引号
   • 确保数组闭合无遗漏
   • 图表标题、轴标签等为可选字段

注意：响应数据的"output"中不要有换行符、制表符以及其他格式符号。
"""


# 添加对话历史上下文功能
def build_conversation_context(history):
    """构建对话上下文"""
    context = "\n\n### 对话历史:\n"

    # 添加最近的对话历史（最多10条消息）
    recent_history = history[-10:]  # 取最近的10条消息

    for entry in recent_history:
        role = entry["role"]
        content = entry["content"]

        # 简化系统消息
        if role == "system" and len(content) > 100:
            content = content[:100] + "..."

        context += f"{role}: {content}\n"

    return context + "\n### 当前请求:\n"


def dataframe_agent(df, question, openai_model, history=None):
    """
    创建智能体，提问与回答 - 添加对话历史支持
    :param df: 数据集
    :param question: 用户问题
    :param openai_model: OpenAI模型实例
    :param history: 对话历史
    :return: 响应结果
    """
    # 构建完整的提示词，包含对话历史
    full_prompt = PROMPT_PREFIX

    # 如果提供了对话历史，添加到提示词中
    if history:
        full_prompt += build_conversation_context(history)
    else:
        full_prompt += "\n\n### 当前请求:\n"

    full_prompt += question

    agent = create_pandas_dataframe_agent(
        llm=openai_model,
        df=df,
        verbose=True,
        max_iterations=8,
        allow_dangerous_code=True,
        agent_executor_kwargs={
            'handle_parsing_errors': True
        }
    )

    try:
        res = agent.invoke({
            'input': full_prompt
        })
        print("Agent Response:", res['output'])
        return json.loads(res['output'])
    except json.JSONDecodeError as e:
        return {"answer": f"处理请求时发生错误: {str(e)}"}


def generate_chart_with_plotly(data_source, chart_type):
    """使用plotly.express生成交互式图表"""
    try:
        if chart_type == 'bar':
            # 处理柱状图数据
            if 'groups' in data_source:
                # 分组柱状图
                df = pd.DataFrame({
                    'x': np.tile(data_source['x'], len(data_source['groups'])),
                    'y': np.concatenate(data_source['y']),
                    'group': np.repeat(data_source['groups'], len(data_source['x']))
                })
                fig = px.bar(
                    df,
                    x='x',
                    y='y',
                    color='group',
                    barmode='group',
                    title=data_source.get('title', '柱状图'),
                    labels={'x': data_source.get('x_label', '类别'),
                            'y': data_source.get('y_label', '数值')}
                )
            else:
                # 普通柱状图
                fig = px.bar(
                    x=data_source['x'],
                    y=data_source['y'],
                    title=data_source.get('title', '柱状图'),
                    labels={'x': data_source.get('x_label', '类别'),
                            'y': data_source.get('y_label', '数值')}
                )

        elif chart_type == 'line':
            # 处理折线图数据
            if 'groups' in data_source:
                # 多线折线图
                df = pd.DataFrame({
                    'x': np.tile(data_source['x'], len(data_source['groups'])),
                    'y': np.concatenate(data_source['y']),
                    'group': np.repeat(data_source['groups'], len(data_source['x']))
                })
                fig = px.line(
                    df,
                    x='x',
                    y='y',
                    color='group',
                    title=data_source.get('title', '折线图'),
                    markers=True,
                    labels={'x': data_source.get('x_label', 'X轴'),
                            'y': data_source.get('y_label', 'Y轴')}
                )
            else:
                # 单线折线图
                fig = px.line(
                    x=data_source['x'],
                    y=data_source['y'],
                    title=data_source.get('title', '折线图'),
                    markers=True,
                    labels={'x': data_source.get('x_label', 'X轴'),
                            'y': data_source.get('y_label', 'Y轴')}
                )

        elif chart_type == 'pie':
            # 处理饼图数据
            fig = px.pie(
                names=data_source['labels'],
                values=data_source['values'],
                title=data_source.get('title', '饼图'),
                hole=0.3
            )

        elif chart_type == 'scatter':
            # 处理散点图数据
            if 'groups' in data_source:
                # 分组散点图
                df = pd.DataFrame({
                    'x': data_source['x'],
                    'y': data_source['y'],
                    'group': data_source['groups']
                })
                fig = px.scatter(
                    df,
                    x='x',
                    y='y',
                    color='group',
                    title=data_source.get('title', '散点图'),
                    labels={'x': data_source.get('x_label', 'X轴'),
                            'y': data_source.get('y_label', 'Y轴')}
                )
            else:
                # 普通散点图
                fig = px.scatter(
                    x=data_source['x'],
                    y=data_source['y'],
                    title=data_source.get('title', '散点图'),
                    labels={'x': data_source.get('x_label', 'X轴'),
                            'y': data_source.get('y_label', 'Y轴')}
                )

        elif chart_type == 'heatmap':
            # 处理热力图数据
            fig = go.Figure(data=go.Heatmap(
                z=data_source['data'],
                x=data_source.get('x_labels', []),
                y=data_source.get('y_labels', []),
                colorscale='Viridis'
            ))
            fig.update_layout(
                title=data_source.get('title', '热力图'),
                xaxis_title=data_source.get('x_label', 'X轴'),
                yaxis_title=data_source.get('y_label', 'Y轴')
            )

        elif chart_type == 'boxplot':
            # 处理箱线图数据
            if 'groups' in data_source:
                # 分组箱线图
                df = pd.DataFrame({
                    'value': np.concatenate([vals for vals in data_source['groups'].values()]),
                    'group': np.repeat(list(data_source['groups'].keys()),
                                       [len(v) for v in data_source['groups'].values()])
                })
                fig = px.box(
                    df,
                    x='group',
                    y='value',
                    title=data_source.get('title', '箱线图'),
                    labels={'group': data_source.get('x_label', '分组'),
                            'value': data_source.get('y_label', '数值')}
                )
            else:
                # 普通箱线图
                fig = px.box(
                    y=data_source['data'],
                    title=data_source.get('title', '箱线图'),
                    labels={'y': data_source.get('y_label', '数值')}
                )

        else:
            st.error(f"不支持的图表类型: {chart_type}")
            return

        # 设置统一的中文字体
        fig.update_layout(
            font_family="Microsoft YaHei",
            title_font_size=20,
            title_x=0.5
        )

        # 显示图表
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"生成{chart_type}图表时出错: {str(e)}")
        st.json(data_source)
