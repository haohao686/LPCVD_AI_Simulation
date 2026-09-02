import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 页面全局配置
st.set_page_config(layout="wide", page_title="LPCVD 反应腔数字孪生", page_icon="⚙️")

st.title("🚀 LPCVD 反应腔数字孪生 (AI Surrogate Model)")
st.markdown("通过拖动左侧滑块改变工艺参数，AI 模型将实时预测沉积速率及腔体内部物理场分布。")

# ==========================================
# 1. 侧边栏：工艺参数输入
# ==========================================
st.sidebar.header("🎛️ 工艺参数设置")
# 这里的 min/max range 可以根据你的 DOE 实际范围进行调整
T1 = st.sidebar.slider("加热区温度 T1 (K)", min_value=600.0, max_value=1200.0, value=800.0, step=10.0)
m_inlet = st.sidebar.slider("入口质量流率 (sccm)", min_value=10.0, max_value=150.0, value=50.0, step=1.0)
p0 = st.sidebar.slider("出口压力 p0 (Pa)", min_value=10.0, max_value=500.0, value=100.0, step=10.0)

# ==========================================
# 2. 缓存加载所有模型和坐标数据 (避免每次拖动滑块重新加载)
# ==========================================
@st.cache_resource
def load_models():
    # 标量与预处理
    scaler_X = joblib.load('scaler_X.pkl')
    rf_rate = joblib.load('rf_rate.pkl')
    
    # 沉积速率 (1D)
    pca_dep = joblib.load('pca_dep.pkl')
    mlp_dep = joblib.load('mlp_dep.pkl')
    coords_dep = np.load('coords_dep.npy')
    
    # 温度场 (2D)
    pca_temp = joblib.load('pca_temp.pkl')
    mlp_temp = joblib.load('mlp_temp.pkl')
    coords_temp = np.load('coords_temp.npy')
    
    # 流速场 (2D)
    pca_vel = joblib.load('pca_vel.pkl')
    mlp_vel = joblib.load('mlp_vel.pkl')
    coords_vel = np.load('coords_vel.npy')
    
    return (scaler_X, rf_rate, 
            pca_dep, mlp_dep, coords_dep, 
            pca_temp, mlp_temp, coords_temp, 
            pca_vel, mlp_vel, coords_vel)

try:
    models = load_models()
    (scaler_X, rf_rate, 
     pca_dep, mlp_dep, coords_dep, 
     pca_temp, mlp_temp, coords_temp, 
     pca_vel, mlp_vel, coords_vel) = models
    
    # ==========================================
    # 3. 实时推理 (Inference)
    # ==========================================
    # 构造输入并标准化
    X_input = np.array([[T1, m_inlet, p0]])
    X_scaled = scaler_X.transform(X_input)
    
    # 预测标量 (平均沉积速率)
    avg_rate_pred = rf_rate.predict(X_scaled)[0]
    st.success(f"**⚡ 预测平均沉积速率:**  `{avg_rate_pred:.4f}` nm/min")
    st.markdown("---")
    
    # 预测高维场 (MLP 预测降维特征 -> PCA 逆变换还原为物理场)
    dep_pred = pca_dep.inverse_transform(mlp_dep.predict(X_scaled))[0]
    temp_pred = pca_temp.inverse_transform(mlp_temp.predict(X_scaled))[0]
    vel_pred = pca_vel.inverse_transform(mlp_vel.predict(X_scaled))[0]
    
    # ==========================================
    # 4. 可视化渲染 (Visualization)
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 晶圆表面沉积速率分布 (1D)")
        fig_dep, ax_dep = plt.subplots(figsize=(6, 4))
        # coords_dep 的第一列是 X 坐标
        ax_dep.plot(coords_dep[:, 0], dep_pred, color='dodgerblue', linewidth=2)
        ax_dep.set_xlabel("Wafer Position (m)")
        ax_dep.set_ylabel("Deposition Rate")
        ax_dep.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_dep)
        
    with col2:
        st.subheader("🔥 反应腔温度场分布 (2D)")
        fig_temp, ax_temp = plt.subplots(figsize=(6, 4))
        # COMSOL 的网格是非结构化散点，必须使用 tricontourf 来插值渲染
        tc_temp = ax_temp.tricontourf(coords_temp[:, 0], coords_temp[:, 1], temp_pred, levels=30, cmap='inferno')
        fig_temp.colorbar(tc_temp, ax=ax_temp, label='Temperature (K)')
        ax_temp.set_xlabel("X Coordinate")
        ax_temp.set_ylabel("Y Coordinate")
        st.pyplot(fig_temp)
        
    st.markdown("---")
    st.subheader("💨 反应腔气体流速场分布 (2D)")
    fig_vel, ax_vel = plt.subplots(figsize=(12, 4)) # 流速场通常在长管里，设宽一点
    tc_vel = ax_vel.tricontourf(coords_vel[:, 0], coords_vel[:, 1], vel_pred, levels=30, cmap='viridis')
    fig_vel.colorbar(tc_vel, ax=ax_vel, label='Velocity Magnitude (m/s)')
    ax_vel.set_xlabel("X Coordinate")
    ax_vel.set_ylabel("Y Coordinate")
    st.pyplot(fig_vel)

except Exception as e:
    st.error(f"加载模型或渲染时出错: {e}")
    st.warning("请确保所有 .pkl 和 .npy 文件与 app.py 放在同一个文件夹中！")