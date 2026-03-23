import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置页面配置
st.set_page_config(
    page_title="蓄电池健康状态(SOH)预测系统",
    page_icon="🔋",
    layout="wide"
)

# 标题和描述
st.title("🔋 蓄电池健康状态（SOH）预测系统")
st.markdown("""
### 基于机器学习的电池健康状态预测
本系统使用多种机器学习算法预测蓄电池的健康状态（SOH），支持自定义数据集或使用示例数据。
""")

# 侧边栏 - 模型选择
st.sidebar.header("⚙️ 模型配置")
model_choice = st.sidebar.selectbox(
    "选择机器学习模型",
    ["随机森林 (Random Forest)", "梯度提升 (Gradient Boosting)", "支持向量机 (SVR)", "线性回归 (Linear Regression)"]
)

# 侧边栏 - 数据分割比例
test_size = st.sidebar.slider("测试集比例", 0.1, 0.3, 0.15)
val_size = st.sidebar.slider("验证集比例", 0.1, 0.3, 0.15)

# 数据加载部分
st.header("📊 数据加载")

# 生成示例数据
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    # 模拟真实电池数据
    cycles = np.sort(np.random.randint(1, 1000, n_samples))
    
    # SOH随循环次数衰减（真实物理规律）
    base_soh = 100 - (cycles / 1000) * 30  # 从100%衰减到70%
    soh_noise = np.random.normal(0, 2, n_samples)
    soh = np.clip(base_soh + soh_noise, 0, 100)
    
    # 电压随SOH降低
    voltage = 3.7 + (soh / 100) * 0.3 + np.random.normal(0, 0.05, n_samples)
    
    # 电流（随机波动）
    current = 1.0 + np.random.normal(0, 0.2, n_samples)
    
    # 温度（与循环次数相关）
    temperature = 25 + (cycles / 1000) * 15 + np.random.normal(0, 2, n_samples)
    
    # 容量（与SOH直接相关）
    capacity = (soh / 100) * 3.0 + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame({
        'Cycle': cycles,
        'Voltage': voltage,
        'Current': current,
        'Temperature': temperature,
        'Capacity': capacity,
        'SOH': soh
    })
    
    return df

# 文件上传
uploaded_file = st.file_uploader("上传CSV数据集（格式：Cycle, Voltage, Current, Temperature, Capacity, SOH）", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ 数据加载成功！")
else:
    st.info("📝 未上传文件，使用示例数据进行演示")
    n_samples = st.slider("生成示例数据样本数", 100, 2000, 1000)
    df = generate_sample_data(n_samples)

# 数据预览
st.subheader("📋 数据预览")
st.dataframe(df.head(10))

# 数据统计
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("数据集大小", f"{len(df)} 条")
with col2:
    st.metric("特征数量", f"{len(df.columns) - 1} 个")
with col3:
    st.metric("SOH范围", f"{df['SOH'].min():.1f}% - {df['SOH'].max():.1f}%")

# 数据分布可视化
st.subheader("📈 数据分布分析")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('各特征分布图', fontsize=16)

columns = ['Cycle', 'Voltage', 'Current', 'Temperature', 'Capacity', 'SOH']
for idx, col in enumerate(columns):
    row = idx // 3
    col_idx = idx % 3
    axes[row, col_idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('频数')
    axes[row, col_idx].set_title(f'{col} 分布')
    axes[row, col_idx].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# 相关性热力图
st.subheader("🔗 特征相关性分析")
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = df.corr()
cax = ax.matshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(cax)
ax.set_xticks(range(len(df.columns)))
ax.set_yticks(range(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=45)
ax.set_yticklabels(df.columns)

# 添加数值标注
for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black")

plt.title('特征相关性热力图')
st.pyplot(fig)

# 数据预处理
st.header("🔧 数据预处理")

# 特征工程
st.subheader("🎯 特征工程")

# 提取高级特征
df_engineered = df.copy()

# 容量衰减率（相对于初始容量）
df_engineered['Capacity_Decay_Rate'] = (df_engineered['Capacity'].max() - df_engineered['Capacity']) / df_engineered['Capacity'].max() * 100

# 电压变化率
df_engineered['Voltage_Change_Rate'] = df_engineered['Voltage'].pct_change().fillna(0)

# 循环效率（假设新电池效率高）
df_engineered['Cycle_Efficiency'] = 100 - (df_engineered['Cycle'] / df_engineered['Cycle'].max()) * 20

# 温度归一化
df_engineered['Temp_Normalized'] = (df_engineered['Temperature'] - df_engineered['Temperature'].mean()) / df_engineered['Temperature'].std()

st.write("新增特征：")
st.write("- **容量衰减率**：相对于最大容量的衰减百分比")
st.write("- **电压变化率**：电压相对于前一时刻的变化")
st.write("- **循环效率**：基于循环次数的效率指标")
st.write("- **温度归一化**：温度的标准化值")

st.dataframe(df_engineered.head())

# 特征选择
feature_columns = ['Cycle', 'Voltage', 'Current', 'Temperature', 'Capacity', 
                   'Capacity_Decay_Rate', 'Voltage_Change_Rate', 'Cycle_Efficiency', 'Temp_Normalized']

X = df_engineered[feature_columns]
y = df_engineered['SOH']

# 数据标准化
st.subheader("📏 数据标准化")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

st.write("✅ 使用StandardScaler进行Z-score标准化")

# 数据集划分
st.subheader("✂️ 数据集划分")

# 先划分训练集和临时集（验证集+测试集）
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=(test_size + val_size), random_state=42
)

# 再将临时集划分为验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=42
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("训练集", f"{len(X_train)} 条 ({len(X_train)/len(X)*100:.1f}%)")
with col2:
    st.metric("验证集", f"{len(X_val)} 条 ({len(X_val)/len(X)*100:.1f}%)")
with col3:
    st.metric("测试集", f"{len(X_test)} 条 ({len(X_test)/len(X)*100:.1f}%)")

# 模型训练
st.header("🤖 模型训练")

# 根据选择创建模型
if model_choice == "随机森林 (Random Forest)":
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
elif model_choice == "梯度提升 (Gradient Boosting)":
    model = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=5)
elif model_choice == "支持向量机 (SVR)":
    model = SVR(kernel='rbf', C=10, gamma='scale')
else:  # 线性回归
    model = LinearRegression()

st.write(f"🎯 当前模型：**{model_choice}**")

# 训练模型
with st.spinner("模型训练中..."):
    model.fit(X_train, y_train)
    st.success("✅ 模型训练完成！")

# 在验证集上预测
y_val_pred = model.predict(X_val)

# 在测试集上预测
y_test_pred = model.predict(X_test)

# 模型评估
st.header("📊 模型评估")

# 计算评估指标
mse_train = mean_squared_error(y_train, model.predict(X_train))
mae_train = mean_absolute_error(y_train, model.predict(X_train))
r2_train = r2_score(y_train, model.predict(X_train))

mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# 显示评估指标
st.subheader("📈 评估指标对比")

metrics_df = pd.DataFrame({
    '数据集': ['训练集', '验证集', '测试集'],
    '均方误差 (MSE)': [mse_train, mse_val, mse_test],
    '平均绝对误差 (MAE)': [mae_train, mae_val, mae_test],
    '决定系数 (R²)': [r2_train, r2_val, r2_test]
})

st.dataframe(metrics_df.style.format({
    '均方误差 (MSE)': '{:.4f}',
    '平均绝对误差 (MAE)': '{:.4f}',
    '决定系数 (R²)': '{:.4f}'
}))

# 可视化预测结果
st.subheader("🎨 预测结果可视化")

# 测试集预测 vs 真实值
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'{model_choice} 模型预测结果', fontsize=16)

# 1. 散点图：预测值 vs 真实值
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('真实 SOH (%)')
axes[0, 0].set_ylabel('预测 SOH (%)')
axes[0, 0].set_title('预测值 vs 真实值')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].text(0.05, 0.95, f'R² = {r2_test:.4f}', transform=axes[0, 0].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. 残差图
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, s=20)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('预测 SOH (%)')
axes[0, 1].set_ylabel('残差 (%)')
axes[0, 1].set_title('残差分布')
axes[0, 1].grid(True, alpha=0.3)

# 3. 按样本索引的预测对比
sample_indices = range(len(y_test))
axes[1, 0].plot(sample_indices[:100], y_test.values[:100], 'b-', label='真实值', alpha=0.7)
axes[1, 0].plot(sample_indices[:100], y_test_pred[:100], 'r--', label='预测值', alpha=0.7)
axes[1, 0].set_xlabel('样本索引（前100个）')
axes[1, 0].set_ylabel('SOH (%)')
axes[1, 0].set_title('预测值 vs 真实值（前100个样本）')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 误差分布直方图
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('残差 (%)')
axes[1, 1].set_ylabel('频数')
axes[1, 1].set_title('残差分布直方图')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# 特征重要性（如果模型支持）
if hasattr(model, 'feature_importances_'):
    st.subheader("🎯 特征重要性分析")
    
    importance_df = pd.DataFrame({
        '特征': feature_columns,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['特征'], importance_df['重要性'], color='steelblue', edgecolor='black')
    ax.set_xlabel('重要性得分')
    ax.set_ylabel('特征名称')
    ax.set_title('特征重要性排序')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 在柱状图上添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.dataframe(importance_df.style.format({'重要性': '{:.4f}'}))

# 模型对比（所有模型）
st.header("🔬 多模型对比")

st.subheader("所有模型性能对比")

models_comparison = []
model_names = ["随机森林", "梯度提升", "支持向量机", "线性回归"]

for i, model_name in enumerate(model_names):
    if model_name == "随机森林":
        model_compare = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    elif model_name == "梯度提升":
        model_compare = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=5)
    elif model_name == "支持向量机":
        model_compare = SVR(kernel='rbf', C=10, gamma='scale')
    else:
        model_compare = LinearRegression()
    
    # 训练和预测
    model_compare.fit(X_train, y_train)
    y_pred_compare = model_compare.predict(X_test)
    
    # 计算指标
    mse = mean_squared_error(y_test, y_pred_compare)
    mae = mean_absolute_error(y_test, y_pred_compare)
    r2 = r2_score(y_test, y_pred_compare)
    
    models_comparison.append({
        '模型': model_name,
        '均方误差 (MSE)': mse,
        '平均绝对误差 (MAE)': mae,
        '决定系数 (R²)': r2
    })

comparison_df = pd.DataFrame(models_comparison)
st.dataframe(comparison_df.style.format({
    '均方误差 (MSE)': '{:.4f}',
    '平均绝对误差 (MAE)': '{:.4f}',
    '决定系数 (R²)': '{:.4f}'
}).background_gradient(cmap='RdYlGn'))

# 模型性能对比图表
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MSE对比
axes[0].bar(comparison_df['模型'], comparison_df['均方误差 (MSE)'], color='coral', edgecolor='black')
axes[0].set_ylabel('MSE')
axes[0].set_title('各模型 MSE 对比')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# MAE对比
axes[1].bar(comparison_df['模型'], comparison_df['平均绝对误差 (MAE)'], color='lightgreen', edgecolor='black')
axes[1].set_ylabel('MAE')
axes[1].set_title('各模型 MAE 对比')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

# R²对比
axes[2].bar(comparison_df['模型'], comparison_df['决定系数 (R²)'], color='skyblue', edgecolor='black')
axes[2].set_ylabel('R²')
axes[2].set_title('各模型 R² 对比')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='R²=0.9')
axes[2].legend()

plt.tight_layout()
st.pyplot(fig)

# 单点预测功能
st.header("🎮 实时预测")

st.subheader("输入电池参数进行SOH预测")

# 创建输入表单
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cycle_input = st.number_input("循环次数 (Cycle)", min_value=1, max_value=5000, value=500)
        voltage_input = st.number_input("电压 (V)", min_value=2.5, max_value=4.5, value=3.7, step=0.01)
        current_input = st.number_input("电流 (A)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    
    with col2:
        temperature_input = st.number_input("温度 (℃)", min_value=-20, max_value=80, value=25)
        capacity_input = st.number_input("容量 (Ah)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
    
    # 计算衍生特征
    max_capacity = df['Capacity'].max()
    capacity_decay_rate = (max_capacity - capacity_input) / max_capacity * 100
    cycle_efficiency = 100 - (cycle_input / 1000) * 20
    
    temp_mean = df['Temperature'].mean()
    temp_std = df['Temperature'].std()
    temp_normalized = (temperature_input - temp_mean) / temp_std
    
    voltage_change_rate = 0  # 无法从单点计算
    
    with col3:
        st.write("**衍生特征（自动计算）**")
        st.write(f"容量衰减率: {capacity_decay_rate:.2f}%")
        st.write(f"循环效率: {cycle_efficiency:.2f}%")
        st.write(f"温度归一化: {temp_normalized:.2f}")
    
    submitted = st.form_submit_button("🔮 预测 SOH")

if submitted:
    # 准备输入数据
    input_data = pd.DataFrame({
        'Cycle': [cycle_input],
        'Voltage': [voltage_input],
        'Current': [current_input],
        'Temperature': [temperature_input],
        'Capacity': [capacity_input],
        'Capacity_Decay_Rate': [capacity_decay_rate],
        'Voltage_Change_Rate': [voltage_change_rate],
        'Cycle_Efficiency': [cycle_efficiency],
        'Temp_Normalized': [temp_normalized]
    })
    
    # 标准化
    input_scaled = scaler.transform(input_data)
    
    # 预测
    prediction = model.predict(input_scaled)[0]
    
    # 显示结果
    st.success(f"🎯 预测结果：**SOH = {prediction:.2f}%**")
    
    # 健康状态评估
    if prediction >= 90:
        health_status = "🟢 优秀"
        advice = "电池状态良好，继续保持正常使用和维护。"
    elif prediction >= 70:
        health_status = "🟡 良好"
        advice = "电池状态尚可，建议关注充放电习惯，避免深度放电。"
    elif prediction >= 50:
        health_status = "🟠 一般"
        advice = "电池性能有所下降，建议检查使用环境，考虑适当维护。"
    else:
        health_status = "🔴 需要更换"
        advice = "电池性能严重衰减，建议尽快更换电池以确保安全和性能。"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("健康状态", health_status)
    with col2:
        st.metric("使用建议", advice)
    
    # SOH 进度条
    st.subheader("📊 SOH 可视化")
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # 创建进度条
    colors = ['#ff4444', '#ffbb33', '#00C851', '#007E33']
    thresholds = [50, 70, 90]
    
    # 背景条
    ax.barh(0, 100, height=0.5, color='#e0e0e0')
    
    # 根据SOH值设置颜色
    if prediction < thresholds[0]:
        color = colors[0]
    elif prediction < thresholds[1]:
        color = colors[1]
    elif prediction < thresholds[2]:
        color = colors[2]
    else:
        color = colors[3]
    
    # 进度条
    ax.barh(0, prediction, height=0.5, color=color)
    
    # 添加阈值线
    for threshold in thresholds:
        ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
        ax.text(threshold, 0.6, f'{threshold}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1)
    ax.set_yticks([])
    ax.set_xlabel('SOH (%)')
    ax.set_title(f'电池健康状态: {prediction:.2f}%')
    
    # 移除边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    st.pyplot(fig)

# 页脚
st.markdown("---")
st.markdown("""
### 💡 系统说明

**技术栈：**
- 机器学习框架：scikit-learn
- 数据处理：pandas, numpy
- 可视化：matplotlib
- Web应用：Streamlit

**支持的模型：**
1. **随机森林**：集成学习方法，对非线性关系建模能力强
2. **梯度提升**：逐步提升弱分类器，预测精度高
3. **支持向量机**：适合小样本数据，泛化能力强
4. **线性回归**：简单高效，可解释性强

**评估指标：**
- **MSE（均方误差）**：衡量预测值与真实值偏差的平方和
- **MAE（平均绝对误差）**：衡量预测值与真实值的平均绝对偏差
- **R²（决定系数）**：模型解释方差的比例，越接近1越好

**特征工程：**
- 容量衰减率：反映电池容量随时间衰减的程度
- 电压变化率：捕捉电压波动特征
- 循环效率：基于循环次数的效率指标
- 温度归一化：消除温度量纲影响

---
**开发者**：AI实践项目 | **日期**：2026年3月
""")
