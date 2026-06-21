import numpy as np
from scipy.optimize import brentq
 
# =========================
# 实验参数（请填入实测值）
# =========================
l   = 0.15      # 摆长 (m)
m   = 0.0377    # 单个摆球质量 (kg)
d0  = 0.0141   # 两悬点间距 (m)
g   = 9.8       # 重力加速度 (m/s²)
 
mu0 = 4 * np.pi * 1e-7
 
# =========================
# 输入：实测两球平衡间距
# =========================
d_measured = float(input("请输入两球平衡时的中心间距 (cm)：")) / 100  # 转换为 m
 
# =========================
# 从间距反算平衡角 θ*
# =========================
# 对称平衡：θ1* = θ*, θ2* = -θ*
# 两球间距：
#   dx = d0 + l*sin(-θ*) - l*sin(θ*) = d0 - 2l*sinθ*
#   dy = l*cos(θ*) - l*cos(-θ*) = 0
# 所以 d = d0 - 2l*sinθ*  →  θ* = arcsin((d0 - d) / (2l))
 
sin_theta = (d0 - d_measured) / (2 * l)
 
if abs(sin_theta) > 1:
    print(f"\n错误：间距 {d_measured*100:.2f} cm 超出几何范围。")
    print(f"  间距范围应在 {(d0-2*l)*100:.2f} ~ {d0*100:.2f} cm 之间")
    exit()
 
theta_eq = np.arcsin(sin_theta)
print(f"\n对应平衡偏角 θ* = {np.rad2deg(theta_eq):.4f}°")
print(f"验证间距 = {(d0 - 2*l*np.sin(theta_eq))*100:.4f} cm  ✓")
 
# =========================
# 平衡条件：对 θ1 的力矩 = 0
# θ1 = θ*, θ2 = -θ*
# (g/l)*sin(θ*) + (1/ml²) * ∂Um/∂θ1 = 0
# =========================
def dUm_dtheta1(theta_star, M):
    """解析梯度（排斥构型）"""
    t1, t2 = theta_star, -theta_star
 
    dx = d0 + l*np.sin(t2) - l*np.sin(t1)
    dy = l*np.cos(t1) - l*np.cos(t2)
    r2 = dx**2 + dy**2
    r  = np.sqrt(r2)
    r2 = r * r
 
    A = d0*np.cos(t1) + l*np.sin(t2 - t1)
    B = d0*np.cos(t2) + l*np.sin(t2 - t1)
 
    dr2_dt1 = 2*dx*(-l*np.cos(t1)) + 2*dy*(-l*np.sin(t1))
    dA_dt1  = -d0*np.sin(t1) - l*np.cos(t2 - t1)
    dB_dt1  =                - l*np.cos(t2 - t1)
 
    C = mu0 * M**2 / (4 * np.pi)
    f = -np.cos(t1 - t2) + 3*A*B / r2
    df_dt1 = (np.sin(t1 - t2)
              + 3*(dA_dt1*B + A*dB_dt1)/r2
              - 3*A*B/r2**2 * dr2_dt1)
 
    return C * (-1.5 * r**(-5) * dr2_dt1 * f + r**(-3) * df_dt1)
 
 
def equilibrium_residual(M):
    """平衡条件残差：力矩之和应为 0"""
    tau_gravity = (g / l) * np.sin(theta_eq)          # 重力矩（使摆回正）
    tau_mag     = dUm_dtheta1(theta_eq, M) / (m * l**2)  # 磁力矩（使摆偏开）
    return tau_gravity + tau_mag                        # 平衡时 = 0
 
 
# =========================
# 用二分法求解 M
# =========================
# 粗略扫描确认有根
M_range = np.logspace(-3, 1, 500)
residuals = [equilibrium_residual(M) for M in M_range]
 
# 找符号变化区间
M_lo, M_hi = None, None
for i in range(len(residuals) - 1):
    if residuals[i] * residuals[i+1] < 0:
        M_lo, M_hi = M_range[i], M_range[i+1]
        break
 
if M_lo is None:
    print("\n未找到合理的 M 解，请检查输入参数或磁铁极性设置。")
    print("残差范围：", min(residuals), "~", max(residuals))
else:
    M_solution = brentq(equilibrium_residual, M_lo, M_hi, xtol=1e-8)
 
    print()
    print("=" * 45)
    print(f"  反算得到磁矩 M = {M_solution:.5f} A·m²")
    print("=" * 45)
 
    # 验证
    tau_g = (g/l) * np.sin(theta_eq)
    tau_m = dUm_dtheta1(theta_eq, M_solution) / (m * l**2)
    print(f"\n验证（力矩平衡）：")
    print(f"  重力矩   = {tau_g:.6f} rad/s²")
    print(f"  磁力矩   = {tau_m:.6f} rad/s²")
    print(f"  残差     = {abs(tau_g + tau_m):.2e}  ✓")
 
    # 磁力与重力的比值
    F_mag_approx = 3 * mu0 * M_solution**2 / (2 * np.pi * d_measured**4)
    F_grav = m * g
    print(f"\n磁力/重力 ≈ {F_mag_approx/F_grav*100:.2f}%")
    print(f"请将代码中的 M = {M_solution:.4f} 代入模拟")