import pandas as pd
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ------------------------- 1. 页面基础配置 -------------------------
# 设置页面标题、图标、布局
st.set_page_config(
    page_title="企鹅分类器",  # 浏览器标签显示的标题
    page_icon=":penguin:",    # 页面图标（Emoji 形式）
    layout="wide"             # 宽屏布局
)

# ------------------------- 2. 侧边栏实现多页面切换 -------------------------
with st.sidebar:
    # 侧边栏显示图标
    st.image('images/rigth_logo.png', width=100)  
    st.title("请选择页面")  # 侧边栏标题
    # 页面选择下拉框，支持“简介页面”和“预测分类页面”
    page = st.selectbox(
        "请选择页面", 
        ["简介页面", "预测分类页面"],
        label_visibility='collapsed'  # 隐藏默认标签，让界面更简洁
    )

# ------------------------- 3. 简介页面内容 -------------------------
if page == "简介页面":
    st.title("企鹅分类器:penguin:")  # 主内容区标题
    st.header("数据集介绍")
    # 用 Markdown 详细说明数据集背景
    st.markdown("""
    帕尔默群岛企鹅数据集是用于数据探索和数据可视化的一个出色的数据集，也可以作为机器学习入门练习。  
    该数据集是由 Gorman 等收集，并发布在一个名为 palmerpenguins 的 R 语言包，以对南极企鹅种类进行分类和研究。  
    该数据集记录了 344 行观测数据，包含 3 个不同物种的企鹅：阿德利企鹅、巴布亚企鹅和帽带企鹅的各种信息。
    """)
    st.header("三种企鹅的卡通图像")
    # 展示企鹅卡通图
    st.image('images/penguins.png')  


# ------------------------- 4. 预测分类页面内容 -------------------------
elif page == "预测分类页面":
    st.header("预测企鹅分类")  # 主内容区标题
    # 用 Markdown 说明应用功能
    st.markdown("""
    这个 Web 应用是基于帕尔默群岛企鹅数据集构建的模型。只需输入 6 个信息，就可以预测企鹅的物种，使用下面的表单开始预测吧！
    """)

    # ------------------------- 4.1 表单与用户输入 -------------------------
    # 3:1:2 的列布局，分别放表单、预留列、结果展示
    col_form, col, col_logo = st.columns([3, 1, 2])
    with col_form:
        # 用 Streamlit 表单收集用户输入，点击提交后再执行预测
        with st.form('user_inputs'):
            # 岛屿选择
            island = st.selectbox(
                '企鹅栖息的岛屿', 
                options=['托尔森岛', '比斯科群岛', '德里姆岛']
            )
            # 性别选择
            sex = st.selectbox(
                '性别', 
                options=['雄性', '雌性']
            )
            # 喙的长度输入
            bill_length = st.number_input(
                '喙的长度（毫米）', 
                min_value=0.0
            )
            # 喙的深度输入
            bill_depth = st.number_input(
                '喙的深度（毫米）', 
                min_value=0.0
            )
            # 翅膀长度输入
            flipper_length = st.number_input(
                '翅膀的长度（毫米）', 
                min_value=0.0
            )
            # 身体质量输入
            body_mass = st.number_input(
                '身体质量（克）', 
                min_value=0.0
            )
            # 表单提交按钮
            submitted = st.form_submit_button('预测分类')

    # ------------------------- 4.2 数据预处理（独热编码转换） -------------------------
    # 初始化岛屿相关独热编码变量（对应 3 个岛屿）
    island_biscoe, island_dream, island_torgerson = 0, 0, 0
    # 根据选择的岛屿设置对应编码
    if island == '比斯科群岛':
        island_biscoe = 1
    elif island == '德里姆岛':
        island_dream = 1
    elif island == '托尔森岛':
        island_torgerson = 1

    # 初始化性别相关独热编码变量（对应雌雄）
    sex_female, sex_male = 0, 0
    # 根据选择的性别设置对应编码
    if sex == '雌性':
        sex_female = 1
    elif sex == '雄性':
        sex_male = 1

    # 拼接预处理后的数据（与模型训练时的特征顺序对应）
    format_data = [
        bill_length, bill_depth, flipper_length, body_mass,
        island_dream, island_torgerson, island_biscoe, 
        sex_male, sex_female
    ]

    # ------------------------- 4.3 加载模型与预测 -------------------------
    # 加载训练好的随机森林模型
    with open('rfc_model.pkl', 'rb') as f:
        rfc_model = pickle.load(f)

    # 加载类别映射（编码 -> 企鹅种类）
    with open('output_uniques.pkl', 'rb') as f:
        output_uniques_map = pickle.load(f)

    # ------------------------- 4.4 预测结果展示 -------------------------
    if submitted:
        # 将预处理后的数据转为 DataFrame（匹配模型特征要求）
        format_data_df = pd.DataFrame(
            data=[format_data], 
            columns=rfc_model.feature_names_in_
        )
        # 使用模型预测
        predict_result_code = rfc_model.predict(format_data_df)
        # 映射编码到具体物种名称
        predict_result_species = output_uniques_map[predict_result_code[0]]
        
        # 输出预测结果
        st.write(
            f'根据您输入的数据，预测该企鹅的物种名称是：**{predict_result_species}**'
        )

    # ------------------------- 4.5 右侧图片展示（根据提交状态/预测结果切换） -------------------------
    with col_logo:
        if not submitted:
            # 未提交时显示默认图标
            st.image('images/rigth_logo.png', width=300)
        else:
            # 提交后显示对应物种的图片
            st.image(
                f'images/{predict_result_species}.png', 
                width=300
            )
