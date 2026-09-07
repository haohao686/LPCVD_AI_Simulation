import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ==========================================
# 0. 页面全局配置
# ==========================================
st.set_page_config(layout="wide", page_title="LPCVD 反应腔数字孪生", page_icon="⚙️")
st.title("🚀 LPCVD 反应腔数字孪生 (CNN Surrogate Model)")
st.markdown("通过拖动左侧滑块改变工艺参数，AI 模型将实时预测沉积速率及腔体内部二维物理场分布。")

# ==========================================
# 1. 核心网络结构 (必须与训练时完全一致)
# ==========================================
class CNNGenerator(nn.Module):
    def __init__(self):
        super(CNNGenerator, self).__init__()
        self.fc = nn.Linear(3, 256 * 4 * 4) 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4) 
        x = self.decoder(x)
        return x.squeeze(1) 

# ==========================================
# 2. 侧边栏：工艺参数输入 (已修正温度单位为摄氏度)
# ==========================================
st.sidebar.header("🎛️ 工艺参数设置")
T1 = st.sidebar.slider("加热区温度 T1 (℃)", min_value=700.0, max_value=900.0, value=800.0, step=1.0)
m_inlet = st.sidebar.slider("入口质量流率 (kg/s)", min_value=1.3e-5, max_value=1.973e-5, value=1.5e-5, step=1.0e-7, format="%.2e")
p0 = st.sidebar.slider("出口压力 p0 (Pa)", min_value=10.0, max_value=100.0, value=50.0, step=1.0)

# ==========================================
# 3. 缓存加载所有模型和数据
# ==========================================
@st.cache_resource
def load_models():
    # 标量与输入预处理
    scaler_X = joblib.load('scaler_X.pkl')
    rf_rate = joblib.load('rf_rate.pkl')
    
    # 沉积速率 (1D: PCA + MLP)
    pca_dep = joblib.load('pca_dep.pkl')
    mlp_dep = joblib.load('mlp_dep.pkl')
    coords_dep = np.load('coords_dep.npy')
    
    # --- 加载 CNN 模型 (温度场) ---
    cnn_temp = CNNGenerator()
    cnn_temp.load_state_dict(torch.load('cnn_temp.pth'))
    cnn_temp.eval() # 设置为评估模式
    scaler_Y_temp = joblib.load('scaler_Y_temp.pkl')
    grid_x_temp = np.load('grid_x_temp.npy')
    grid_y_temp = np.load('grid_y_temp.npy')
    
    # --- 加载 CNN 模型 (流速场) ---
    cnn_vel = CNNGenerator()
    cnn_vel.load_state_dict(torch.load('cnn_vel.pth'))
    cnn_vel.eval() # 设置为评估模式
    scaler_Y_vel = joblib.load('scaler_Y_vel.pkl')
    grid_x_vel = np.load('grid_x_vel.npy')
    grid_y_vel = np.load('grid_y_vel.npy')
    
    return (scaler_X, rf_rate, 
            pca_dep, mlp_dep, coords_dep,
            cnn_temp, scaler_Y_temp, grid_x_temp, grid_y_temp,
            cnn_vel, scaler_Y_vel, grid_x_vel, grid_y_vel)

try:
    models = load_models()
    (scaler_X, rf_rate, 
     pca_dep, mlp_dep, coords_dep,
     cnn_temp, scaler_Y_temp, grid_x_temp, grid_y_temp,
     cnn_vel, scaler_Y_vel, grid_x_vel, grid_y_vel) = models
    
    # ==========================================
    # 4. 实时推理 (Inference)
    # ==========================================
    # 准备输入数据
    X_input = np.array([[T1, m_inlet, p0]])
    X_scaled = scaler_X.transform(X_input)
    X_tensor = torch.FloatTensor(X_scaled)
    
    # 预测平均沉积速率
    avg_rate_pred = rf_rate.predict(X_scaled)[0]
    st.success(f"**⚡ 预测平均沉积速率:**  `{avg_rate_pred:.4e}` m/s")
    st.markdown("---")
    
    # 预测 1D 沉积速率曲线
    dep_pred = pca_dep.inverse_transform(mlp_dep.predict(X_scaled))[0]
    
    # 预测 2D 温度场 (CNN)
    with torch.no_grad():
        temp_pred_tensor = cnn_temp(X_tensor)
        temp_pred_flat = temp_pred_tensor.numpy().flatten().reshape(1, -1)
        temp_pred_unscaled = scaler_Y_temp.inverse_transform(temp_pred_flat)
        temp_pred_2d = temp_pred_unscaled.reshape(64, 64)
        
    # 预测 2D 流速场 (CNN)
    with torch.no_grad():
        vel_pred_tensor = cnn_vel(X_tensor)
        vel_pred_flat = vel_pred_tensor.numpy().flatten().reshape(1, -1)
        vel_pred_unscaled = scaler_Y_vel.inverse_transform(vel_pred_flat)
        vel_pred_2d = vel_pred_unscaled.reshape(64, 64)
        
    # ==========================================
    # 5. 可视化渲染 (垂直全宽排版)
    # ==========================================
    
    # --- 1D 沉积速率曲线 ---
    st.subheader("📊 晶圆表面沉积速率分布 (1D)")
    fig_dep, ax_dep = plt.subplots(figsize=(10, 3))
    ax_dep.plot(coords_dep[:, 0], dep_pred, color='dodgerblue', linewidth=2)
    ax_dep.set_xlabel("Wafer Position (mm)")
    ax_dep.set_ylabel("Deposition Rate (m/s)")
    ax_dep.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_dep)
    plt.close(fig_dep)
    
    st.markdown("---")
    
    # --- 2D 温度场 (从 64x64 画布渲染) ---
    st.subheader("🔥 反应腔温度场分布 (2D)")
    fig_temp, ax_temp = plt.subplots(figsize=(12, 3))
    # 使用 pcolormesh 或 contourf 绘制规则网格数据
    tc_temp = ax_temp.contourf(grid_x_temp, grid_y_temp, temp_pred_2d, levels=40, cmap='inferno')
    fig_temp.colorbar(tc_temp, ax=ax_temp, label='Temperature (K)')
    ax_temp.set_xlabel("X Coordinate (mm)")
    ax_temp.set_ylabel("Y Coordinate (mm)")
    ax_temp.axis('equal') 
    st.pyplot(fig_temp)
    plt.close(fig_temp)
    
    st.markdown("---")
    
    # --- 2D 流速场 (从 64x64 画布渲染) ---
    st.subheader("💨 反应腔气体流速场分布 (2D)")
    fig_vel, ax_vel = plt.subplots(figsize=(12, 3))
    tc_vel = ax_vel.contourf(grid_x_vel, grid_y_vel, vel_pred_2d, levels=40, cmap='viridis')
    fig_vel.colorbar(tc_vel, ax=ax_vel, label='Velocity Magnitude (m/s)')
    ax_vel.set_xlabel("X Coordinate (mm)")
    ax_vel.set_ylabel("Y Coordinate (mm)")
    ax_vel.axis('equal') 
    st.pyplot(fig_vel)
    plt.close(fig_vel)

except Exception as e:
    st.error(f"发生错误: 请确保您已运行过 train_cnn.py 并且相关的 .pkl, .pth, .npy 文件都在当前目录下。详细错误: {e}")