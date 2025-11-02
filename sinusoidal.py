# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import quad
#
# # 定义不同的θ(t)函数
# theta_funcs = {
#     't':        lambda t: t,
#     # 't/10':        lambda t: t/10,
#     '$t^2$':    lambda t: t**2,
#     # '$t^3$':    lambda t: t**3,
#     # '$t^2+t$': lambda t: t**2+t,
#     # '$200^{-t}$': lambda t: 200**(-t),
#     '$500^{-t}$': lambda t: 500**(-t),
#     '$1000^{-t}$': lambda t: 1000**(-t),
#     '$10000^{-t}$': lambda t: 10000**(-t),
#     '$100000^{-t}$': lambda t: 100000**(-t)
# }
#
# ds = np.arange(-300, 301)
# plt.figure(figsize=(10, 6))
#
# for name, theta_t in theta_funcs.items():
#     def real_integral(d):
#         func = lambda t: np.cos(d * theta_t(t))
#         val, _ = quad(func, 0, 1, limit=100)
#         return val
#     vals = [real_integral(d) for d in ds]
#     plt.plot(ds, vals, label=name)
#
# plt.xlabel('|m - n|', fontsize=24)
# plt.ylabel('<$p_m$, $p_n$>', fontsize=24)
# plt.tick_params(axis='both', labelsize=18)  # 修改坐标轴数字的大小
#
# # 设置图例字体
# plt.legend(fontsize=18, prop={'size': 16})
#
# plt.grid(True)
#
# # 保存新的图像
# plt.savefig("pe.png",dpi=1000)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 使用白底，无网格样式
plt.style.use('default')
# 定义颜色列表 (Dark2 配色)
colors = plt.get_cmap('Dark2').colors

theta_funcs = {
    't':           lambda t: t,
    '$t^2$':       lambda t: t**2,
    '$t^3$':       lambda t: t**3,
    '$500^{-t}$':  lambda t: 500**(-t),
    '$1000^{-t}$': lambda t: 1000**(-t),
    '$10000^{-t}$': lambda t: 10000**(-t),
    '$100000^{-t}$': lambda t: 100000**(-t)
}

ds = np.arange(-300, 301)

fig, ax = plt.subplots(figsize=(6, 4))  # 典型论文比例
for idx, (label, theta) in enumerate(theta_funcs.items()):
    vals = [quad(lambda t: np.cos(d * theta(t)), 0, 1, limit=100)[0] for d in ds]
    ax.plot(ds, vals, label=label,
            color=colors[idx % len(colors)],
            linewidth=1.8)

# 字体和刻度设置
ax.set_xlabel(r'$|m-n|$', fontsize=12)
ax.set_ylabel(r'$\langle p_m, p_n \rangle$', fontsize=12)
ax.tick_params(axis='both', labelsize=10)

# 网格设置（可选细网格）
ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)

# 图例放到右侧
ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

ax.set_xlim(-300, 300)
plt.tight_layout()

# 输出为矢量格式 PDF/SVG
fig.savefig('pe_symmetric_high_res.pdf', format='pdf', dpi=300)
fig.savefig('pe_symmetric_high_res.svg', format='svg')