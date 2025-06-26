import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from utils import dataframe_agent, generate_chart_with_plotly
import time

# 设置页面配置
st.set_page_config(
    page_title="智能数据分析助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* 主聊天容器 - 添加滚动功能 */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
    max-height: 65vh;
    overflow-y: auto;
}

/* 消息行容器 */
.message-row {
    display: flex;
    margin-bottom: 20px;
}

/* 用户消息行 - 靠右 */
.user-row {
    justify-content: flex-end;
}

/* 助手消息行 - 靠左 */
.assistant-row {
    justify-content: flex-start;
}

/* 消息气泡 */
.message-bubble {
    max-width: 85%;
    padding: 18px 25px;
    border-radius: 20px;
    position: relative;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

/* 用户气泡样式 */
.user-bubble {
    background: linear-gradient(135deg, #4776E6, #3054c4);
    color: white;
    border-bottom-right-radius: 4px;
}

/* 助手气泡样式 */
.assistant-bubble {
    background: white;
    color: #333;
    border-bottom-left-radius: 4px;
    border: 1px solid #e5e7eb;
}

/* 消息头部（角色和时间） */
.message-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-size: 0.9rem;
}

.user-bubble .message-header {
    color: #d0deff;
}

.assistant-bubble .message-header {
    color: #777;
}

.message-role {
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
}

.user-bubble .message-role i {
    color: #a7c3ff;
}

.assistant-bubble .message-role i {
    color: #4776E6;
}

.message-time {
    opacity: 0.8;
    font-size: 0.8rem;
}

/* 消息内容样式 */
.message-content {
    font-size: 1.05rem;
    line-height: 1.6;
}

/* 表格样式 */
.message-table {
    margin-top: 15px;
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}

.assistant-bubble .message-table {
    background-color: #f9fafc;
}

.message-table th {
    background-color: #f0f4ff;
    font-weight: 600;
    text-align: left;
    padding: 12px 15px;
}

.assistant-bubble .message-table th {
    background-color: #f5f7fb;
}

.message-table td {
    padding: 12px 15px;
    border-bottom: 1px solid #e5e7eb;
}

/* 图表容器 - 集成到气泡内 */
.message-chart {
    margin-top: 16px;
    border-radius: 8px;
    padding: 0;
    background-color: white;
    border: none;
}

/* 隐藏Streamlit默认的聊天消息样式 */
div[data-testid="stChatMessageContent"] {
    padding: 0;
    background: none;
    border: none;
    box-shadow: none;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .message-bubble {
        max-width: 92%;
    }

    .chat-container {
        max-height: 55vh;
    }
}

/* 添加动画效果 */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.message-row {
    animation: fadeIn 0.3s ease-out;
}

/* 滚动条美化 */
.chat-container::-webkit-scrollbar {
    width: 8px;
}

.chat-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

.chat-container::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 10px;
}

.chat-container::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
}

/* 输入区域固定 */
.chat-input-container {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 16px;
    box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.05);
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'data' not in st.session_state:
    st.session_state.data = None

if 'openai_model' not in st.session_state:
    st.session_state.openai_model = None

# 添加对话记忆状态
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
    st.session_state.conversation_history.append(
        {"role": "assistant", "content": "您好！我是您的数据分析助手，请上传数据文件或直接提问。"}
    )

# 应用标题和介绍
st.title("🤖 智能数据分析助手")
st.markdown("""
欢迎使用AI数据分析助手！上传您的数据文件，然后使用自然语言提问进行数据分析。
系统将使用AI模型理解您的查询并自动生成可视化图表和数据分析结果。
""")

# 侧边栏设置
with st.sidebar:
    st.subheader("API设置")
    api_key = st.text_input("输入OpenAI API密钥:", type="password", help="请提供OpenAI API密钥以使用AI功能")

    if api_key:
        try:
            st.session_state.openai_model = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=SecretStr(api_key),
                base_url="https://api.openai-hk.com/v1",
            )
            st.success("API密钥已设置!")
        except Exception as e:
            st.error(f"API密钥设置失败: {str(e)}")
    else:
        st.warning("请输入OpenAI API密钥以使用AI功能")

    st.divider()

    # 文件上传区域
    st.subheader("数据上传")
    uploaded_file = st.file_uploader(
        "上传CSV或Excel文件",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        help="支持CSV和Excel格式，文件大小限制200MB"
    )

    # 文件上传处理
    if uploaded_file is not None:
        try:
            # 根据文件类型读取数据
            if uploaded_file.name.endswith('.csv'):
                st.session_state.data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                # 读取Excel文件的工作表列表
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                # 添加工作表选择器
                selected_sheet = st.selectbox(
                    "选择工作表:",
                    sheet_names,
                    index=0,
                    key='sheet_selector'
                )
                # 读取选定的工作表
                st.session_state.data = pd.read_excel(
                    uploaded_file,
                    sheet_name=selected_sheet
                )
            st.success("数据加载成功！")

            # 更新对话历史
            st.session_state.conversation_history.append({
                "role": "system",
                "content": f"用户已上传数据文件：{uploaded_file.name}，"
                           f"数据包含{st.session_state.data.shape[0]}行{st.session_state.data.shape[1]}列"
            })

        except Exception as e:
            st.error(f"加载文件时出错: {str(e)}")
    else:
        st.info("请上传数据文件")

    # 显示数据基本信息
    if st.session_state.data is not None:
        with st.expander("数据概览", expanded=True):
            st.write(f"行数: {st.session_state.data.shape[0]}")
            st.write(f"列数: {st.session_state.data.shape[1]}")
            st.write("数据摘要:")
            st.dataframe(st.session_state.data.describe())

        # 显示前5行
        with st.expander("查看数据"):
            st.dataframe(st.session_state.data.head(5))

        # 显示列信息
        with st.expander("列信息"):
            for col in st.session_state.data.columns:
                dtype = st.session_state.data[col].dtype
                unique_count = st.session_state.data[col].nunique()
                st.write(f"- {col}: {dtype}, {unique_count}个唯一值")

        # 数据预处理选项
        st.divider()
        st.subheader("数据预处理")

        # 缺失值处理
        if st.session_state.data.isnull().sum().sum() > 0:
            missing_cols = st.session_state.data.columns[st.session_state.data.isnull().any()].tolist()
            st.warning(f"检测到缺失值: {', '.join(missing_cols)}")

            missing_option = st.selectbox(
                "选择处理缺失值方法:",
                ["不处理", "删除含有缺失值的行", "用平均值（众数）填充数值（分类）"],
                key='missing_option'
            )

            if st.button("应用缺失值处理"):
                if missing_option == "删除含有缺失值的行":
                    original_rows = st.session_state.data.shape[0]
                    st.session_state.data = st.session_state.data.dropna()
                    new_rows = st.session_state.data.shape[0]
                    st.success(f"已删除含有缺失值的行! 剩余行数: {new_rows} (删除了{original_rows - new_rows}行)")

                    # 更新对话历史
                    st.session_state.conversation_history.append({
                        "role": "system",
                        "content": f"用户删除了含有缺失值的行，从{original_rows}行减少到{new_rows}行"
                    })

                elif missing_option == "用平均值（众数）填充数值（分类）":
                    for col in st.session_state.data.select_dtypes(include=['int', 'float']):
                        st.session_state.data[col] = st.session_state.data[col].fillna(
                            st.session_state.data[col].mean())
                    for col in st.session_state.data.select_dtypes(include=['object']):
                        st.session_state.data[col] = st.session_state.data[col].fillna(
                            st.session_state.data[col].mode()[0])
                    st.success(f"数据缺失值填充完毕！")

                    # 更新对话历史
                    st.session_state.conversation_history.append({
                        "role": "system",
                        "content": "用户使用平均值填充数值列，众数填充分类列的缺失值"
                    })
        else:
            st.success("数据中没有缺失值")

    # 添加对话历史管理功能
    st.divider()
    # 添加清除对话历史按钮
    if st.button("清除对话历史", help="清除所有对话历史记录"):
        st.session_state.conversation_history = []
        st.session_state.messages = [{"role": "assistant", "content": "对话历史已清除，请问您有什么数据分析需求？"}]
        st.success("对话历史已清除！")

# 主聊天界面 - 使用可滚动容器
with st.container():
    # 使用自定义容器
    chat_container = st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        row_class = "user-row" if role == "user" else "assistant-row"
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"

        # 使用自定义消息样式
        message_html = f"""
        <div class="message-row {row_class}">
            <div class="message-bubble {bubble_class}">
                <div class="message-header">
                    <span class="message-role">
                        {'<i class="fas fa-user"></i> 用户' if role == "user" else '<i class="fas fa-robot"></i> 助手'}
                    </span>
                    <span class="message-time">刚刚</span>
                </div>
                <div class="message-content">
        """

        # 显示文本回答
        if "text" in message:
            message_html += f'<div>{message["text"]}</div>'
        elif "content" in message:
            message_html += f'<div>{message["content"]}</div>'

        # 显示表格数据
        if "table" in message:
            table_df = pd.DataFrame(
                data=message["table"]["data"],
                columns=message["table"]["columns"]
            )

            # 创建表格HTML
            table_html = '<div class="chart-container">'
            table_html += '<table class="message-table"><thead><tr>'
            for col in table_df.columns:
                table_html += f'<th>{col}</th>'
            table_html += '</tr></thead><tbody>'

            for _, row in table_df.iterrows():
                table_html += '<tr>'
                for value in row:
                    table_html += f'<td>{value}</td>'
                table_html += '</tr>'

            table_html += '</tbody></table></div>'
            message_html += table_html

        # 添加图表占位符
        if "chart" in message:
            message_html += f'<div id="chart-{i}" class="message-chart"></div>'

        message_html += """
                </div>
            </div>
        </div>
        """

        st.markdown(message_html, unsafe_allow_html=True)

        # 渲染图表到占位符
        if "chart" in message:
            chart_data = message["chart"]["data"]
            chart_type = message["chart"]["type"]

            # 在气泡内部显示图表
            with st.container():
                # 创建图表容器
                chart_container = st.empty()

                # 在容器中渲染图表
                with chart_container:
                    # 添加小延迟确保DOM已加载
                    time.sleep(0.1)
                    generate_chart_with_plotly(
                        data_source=chart_data,
                        chart_type=chart_type
                    )

                # 使用CSS将图表移动到气泡内部
                st.markdown(
                    f"""
                    <script>
                    // 将图表移动到气泡内的占位符
                    setTimeout(function() {{
                        const chartContainer = document.querySelector('#chart-{i}');
                        const chartElement = document.querySelector('[data-testid="stPlotlyChart"]:last-child');
                        if (chartContainer && chartElement) {{
                            // 复制图表元素到气泡中
                            const clonedChart = chartElement.cloneNode(true);
                            chartContainer.innerHTML = '';
                            chartContainer.appendChild(clonedChart);

                            // 移除原始图表
                            chartElement.remove();

                            // 调整图表大小
                            clonedChart.style.width = '100%';
                            clonedChart.style.height = 'auto';
                        }}
                    }}, 300);
                    </script>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

# 添加固定输入区域
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
user_query = st.chat_input("输入您的问题...", key="chat_input")
st.markdown('</div>', unsafe_allow_html=True)

# 处理用户输入
if user_query:
    # 添加到消息历史
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.conversation_history.append({"role": "user", "content": user_query})

    # 显示用户消息（使用自定义样式）
    user_message_html = f"""
    <div class="message-row user-row">
        <div class="message-bubble user-bubble">
            <div class="message-header">
                <span class="message-role">
                    <i class="fas fa-user"></i> 用户
                </span>
                <span class="message-time">刚刚</span>
            </div>
            <div class="message-content">{user_query}</div>
        </div>
    </div>
    """
    st.markdown(user_message_html, unsafe_allow_html=True)

    # 自动滚动到底部
    st.markdown(
        """
        <script>
        setTimeout(function() {
            const container = document.querySelector('.chat-container');
            container.scrollTop = container.scrollHeight;
        }, 100);
        </script>
        """,
        unsafe_allow_html=True
    )

    # 处理用户查询
    if st.session_state.data is None:
        st.warning("请先上传数据文件")
    elif st.session_state.openai_model is None:
        st.warning("请先设置有效的OpenAI API密钥")
    else:
        with st.spinner("正在思考中..."):
            try:
                # 调用代理处理查询
                res = dataframe_agent(
                    df=st.session_state.data,
                    question=user_query,
                    openai_model=st.session_state.openai_model,
                    history=st.session_state.conversation_history
                )

                # 构建助手消息
                assistant_message = {"role": "assistant"}
                message_index = len(st.session_state.messages)  # 用于图表占位符ID

                # 添加文本回答
                if "answer" in res:
                    assistant_message["text"] = res["answer"]
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": res["answer"]
                    })

                # 添加表格数据
                if "table" in res:
                    assistant_message["table"] = {
                        "columns": res["table"]["columns"],
                        "data": res["table"]["data"]
                    }
                    st.session_state.conversation_history.append({
                        "role": "system",
                        "content": f"生成了包含{len(res['table']['data'])}行数据的表格"
                    })

                # 添加图表数据
                chart_types = ['bar', 'line', 'pie', 'scatter', 'heatmap', 'boxplot']
                for chart_type in chart_types:
                    if chart_type in res:
                        assistant_message["chart"] = {
                            "type": chart_type,
                            "data": res[chart_type]
                        }
                        st.session_state.conversation_history.append({
                            "role": "system",
                            "content": f"生成了{chart_type}图表: {res[chart_type].get('title', '')}"
                        })
                        break  # 只显示第一个图表

                # 保存助手消息到历史
                st.session_state.messages.append(assistant_message)

                # 显示助手消息（使用自定义样式）
                assistant_message_html = f"""
                <div class="message-row assistant-row">
                    <div class="message-bubble assistant-bubble">
                        <div class="message-header">
                            <span class="message-role">
                                <i class="fas fa-robot"></i> 助手
                            </span>
                            <span class="message-time">刚刚</span>
                        </div>
                        <div class="message-content">
                """

                # 显示文本回答
                if "text" in assistant_message:
                    assistant_message_html += f'<div>{assistant_message["text"]}</div>'

                # 显示表格数据
                if "table" in assistant_message:
                    table_df = pd.DataFrame(
                        data=assistant_message["table"]["data"],
                        columns=assistant_message["table"]["columns"]
                    )

                    # 创建表格HTML
                    table_html = '<div class="chart-container">'
                    table_html += '<table class="message-table"><thead><tr>'
                    for col in table_df.columns:
                        table_html += f'<th>{col}</th>'
                    table_html += '</tr></thead><tbody>'

                    for _, row in table_df.iterrows():
                        table_html += '<tr>'
                        for value in row:
                            table_html += f'<td>{value}</td>'
                        table_html += '</tr>'

                    table_html += '</tbody></table></div>'
                    assistant_message_html += table_html

                # 添加图表占位符
                if "chart" in assistant_message:
                    assistant_message_html += f'<div id="chart-{message_index}" class="message-chart"></div>'

                assistant_message_html += """
                        </div>
                    </div>
                </div>
                """

                st.markdown(assistant_message_html, unsafe_allow_html=True)

                # 自动滚动到底部
                st.markdown(
                    """
                    <script>
                    setTimeout(function() {
                        const container = document.querySelector('.chat-container');
                        container.scrollTop = container.scrollHeight;
                    }, 100);
                    </script>
                    """,
                    unsafe_allow_html=True
                )

                # 渲染图表到占位符（如果存在）
                if "chart" in assistant_message:
                    chart_data = assistant_message["chart"]["data"]
                    chart_type = assistant_message["chart"]["type"]

                    # 在气泡内部显示图表
                    with st.container():
                        # 创建图表容器
                        chart_container = st.empty()

                        # 在容器中渲染图表
                        with chart_container:
                            # 添加小延迟确保DOM已加载
                            time.sleep(0.1)
                            generate_chart_with_plotly(
                                data_source=chart_data,
                                chart_type=chart_type
                            )

                        # 使用CSS将图表移动到气泡内部
                        st.markdown(
                            f"""
                            <script>
                            setTimeout(function() {{
                                const chartContainer = document.querySelector('#chart-{message_index}');
                                const chartElement = document.querySelector('[data-testid="stPlotlyChart"]:last-child');
                                if (chartContainer && chartElement) {{
                                    // 复制图表元素到气泡中
                                    const clonedChart = chartElement.cloneNode(true);
                                    chartContainer.innerHTML = '';
                                    chartContainer.appendChild(clonedChart);

                                    // 移除原始图表
                                    chartElement.remove();

                                    // 调整图表大小
                                    clonedChart.style.width = '100%';
                                    clonedChart.style.height = 'auto';

                                    // 再次滚动到底部确保可见
                                    const container = document.querySelector('.chat-container');
                                    container.scrollTop = container.scrollHeight;
                                }}
                            }}, 300);
                            </script>
                            """,
                            unsafe_allow_html=True
                        )

            except Exception as e:
                st.error(f"处理查询时出错: {str(e)}")
                st.session_state.conversation_history.append({
                    "role": "system",
                    "content": f"处理查询时出错: {str(e)}"
                })