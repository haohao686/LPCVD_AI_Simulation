import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 页面全局配置
st.set_page_config(layout="wide", page_title="LPCVD 反应腔数字孪生", page_icon="⚙️")

st.title("🚀 LPCVD 反应腔数字孪生 (AI Surrogate Model)")
st.markdown("通过拖动左侧滑块改变工艺参数，AI 模型将实时预测沉积速率及腔体内部物理场分布。")

# ==========================================
# 1. 侧边栏：工艺参数输入 (请务必确保范围与你的 DOE 数据一致！)
# ==========================================
st.sidebar.header("🎛️ 工艺参数设置")

# 【重要】请将以下 min_value 和 max_value 修改为你 CSV 文件中的真实范围！
T1 = st.sidebar.slider("加热区温度 T1 (K)", 
                       min_value=700.0, max_value=900.0, value=800.0, step=1.0)

# 使用科学计数法显示 kg/s，范围设置为极小值示例
m_inlet = st.sidebar.slider("入口质量流率 (kg/s)", 
                            min_value=1.3e-5, max_value=1.973e-5, value=5.0e-5, step=1.0e-7, format="%.2e")

p0 = st.sidebar.slider("出口压力 p0 (Pa)", 
                       min_value=10.0, max_value=100.0, value=100.0, step=1.0)

# ==========================================
# 2. 缓存加载所有模型和坐标数据
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
    X_input = np.array([[T1, m_inlet, p0]])
    X_scaled = scaler_X.transform(X_input)
    
    avg_rate_pred = rf_rate.predict(X_scaled)[0]
    st.success(f"**⚡ 预测平均沉积速率:**  `{avg_rate_pred:.4f}` nm/min")
    st.markdown("---")
    
    dep_pred = pca_dep.inverse_transform(mlp_dep.predict(X_scaled))[0]
    temp_pred = pca_temp.inverse_transform(mlp_temp.predict(X_scaled))[0]
    vel_pred = pca_vel.inverse_transform(mlp_vel.predict(X_scaled))[0]
    
    # ==========================================
    # 4. 可视化渲染 (垂直全宽排版)
    # ==========================================
    
    # --- 1D 沉积速率曲线 ---
    st.subheader("📊 晶圆表面沉积速率分布 (1D)")
    fig_dep, ax_dep = plt.subplots(figsize=(10, 3))
    ax_dep.plot(coords_dep[:, 0], dep_pred, color='dodgerblue', linewidth=2)
    ax_dep.set_xlabel("Wafer Position (mm)")
    ax_dep.set_ylabel("Deposition Rate (nm/s)")
    ax_dep.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_dep)
    plt.close(fig_dep) # 释放内存
    
    st.markdown("---")
    
    # --- 2D 温度场 ---
    st.subheader("🔥 反应腔温度场分布 (2D)")
    fig_temp, ax_temp = plt.subplots(figsize=(12, 3))
    tc_temp = ax_temp.tricontourf(coords_temp[:, 0], coords_temp[:, 1], temp_pred, levels=40, cmap='inferno')
    fig_temp.colorbar(tc_temp, ax=ax_temp, label='Temperature (K)')
    ax_temp.set_xlabel("X Coordinate (mm)")
    ax_temp.set_ylabel("Y Coordinate (mm)")
    ax_temp.axis('equal') # 保持物理比例
    st.pyplot(fig_temp)
    plt.close(fig_temp)
    
    st.markdown("---")
    
    # --- 2D 流速场 ---
    st.subheader("💨 反应腔气体流速场分布 (2D)")
    fig_vel, ax_vel = plt.subplots(figsize=(12, 3))
    tc_vel = ax_vel.tricontourf(coords_vel[:, 0], coords_vel[:, 1], vel_pred, levels=40, cmap='viridis')
    fig_vel.colorbar(tc_vel, ax=ax_vel, label='Velocity Magnitude (m/s)')
    ax_vel.set_xlabel("X Coordinate (mm)")
    ax_vel.set_ylabel("Y Coordinate (mm)")
    ax_vel.axis('equal') # 保持物理比例
    st.pyplot(fig_vel)
    plt.close(fig_vel)

except Exception as e:
    st.error(f"发生错误: {e}")