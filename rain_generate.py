import numpy as np

def rain_generate(r=0.4, P=2, T=180):
    A = 8.701  # 雨力参数
    C = 0.594  # 雨力变动参数
    b = 11.13  # 降雨历时修正参数
    n = 0.555  # 衰减系数

    # r = 0.4 # 雨峰系数
    # P = 2 # 降雨重现期
    # T = 180 # 降雨历时

    # 峰前、后降雨历时
    Ta = int(T * r)
    Tb = T - Ta
    ta = np.arange(Ta)
    tb = np.arange(Tb)

    a = A * (1 + C * np.log10(P))
    ia = a * ((1-n)*ta/r + b) / (ta/r + b)**(1+n) * 60
    ib = a * ((1-n)*tb/(1-r) + b) / (tb/(1-r) + b) ** (1+n) * 60

    ia = ia[::-1]

    rain = np.concatenate((ia, ib), axis=0)  # mm/hr

    return np.sum(rain/60)

print(rain_generate(P=100))