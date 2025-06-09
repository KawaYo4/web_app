import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("./data/data.csv")

# Food selection
food_names = df['name'].unique()
selected_food = st.selectbox("Select a food", food_names)

# Get data for selected food
data = df[df["name"] == selected_food].iloc[0]

# PFC values (grams)
protein = data["たんぱく質P"]
fat = data["脂質F"]
carb = data["炭水化物C"]

# Triangle vertices
P = np.array([0, 1])
F = np.array([-np.sqrt(3)/2, -0.5])
C = np.array([np.sqrt(3)/2, -0.5])

# Draw PFC triangle
def plot_pfc_triangle(protein, fat, carb):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    # Gradient fill
    resolution = 150
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            a = i / resolution
            b = j / resolution
            c = 1 - a - b
            if c < 0: continue

            x = a * P[0] + b * F[0] + c * C[0]
            y = a * P[1] + b * F[1] + c * C[1]

            # Brighter gradient: light pink (P), light blue (F), light yellow (C)
            color_p = np.array([1.0, 0.2, 1.0]) * a
            color_f = np.array([0.2, 1.0, 1.0]) * b
            color_c = np.array([1.0, 1.0, 0.2]) * c
            color = color_p + color_f + color_c
  
            ax.plot(x, y, marker='o', markersize=0.5, color=color, alpha=0.9)

    # Triangle outline
    triangle = np.array([P, C, F, P])
    ax.plot(triangle[:, 0], triangle[:, 1], color='black')

    # Labels
    ax.text(*(P + [0, 0.1]), 'P', ha='center', fontsize=14)
    ax.text(*(C + [0.1, -0.05]), 'C', ha='center', fontsize=14)
    ax.text(*(F + [-0.1, -0.05]), 'F', ha='center', fontsize=14)

    # Auxiliary lines and labels (every 10%)
    def interpolate(A, B, t):
        return (1 - t) * A + t * B

    for i in range(1, 10):
        t = i / 10

        # Lines parallel to each side with labels
        a = interpolate(F, P, t)
        b = interpolate(C, P, t)
        ax.plot([a[0], b[0]], [a[1], b[1]], linestyle='--', color='gray', linewidth=0.5)
        ax.text(*a, f'{int(t * 100)}%', ha='right', va='center', fontsize=8, color='purple')

        a = interpolate(C, F, t)
        b = interpolate(P, F, t)
        ax.plot([a[0], b[0]], [a[1], b[1]], linestyle='--', color='gray', linewidth=0.5)
        ax.text(*a, f'{int(t * 100)}%', ha='left', va='bottom', fontsize=8, color='blue')

        a = interpolate(P, C, t)
        b = interpolate(F, C, t)
        ax.plot([a[0], b[0]], [a[1], b[1]], linestyle='--', color='gray', linewidth=0.5)
        ax.text(*a, f'{int(t * 100)}%', ha='left', va='bottom', fontsize=8, color='orange')

    # Plot selected point
    total = protein + fat + carb
    a = protein / total
    b = fat / total
    c = carb / total
    x = a * P[0] + b * F[0] + c * C[0]
    y = a * P[1] + b * F[1] + c * C[1]
    ax.plot(x, y, marker='o', color='red', markersize=10)
    ax.text(x, y, f'  (P:{protein}%, F:{fat}%, C:{carb}%)', color='red', fontsize=12) 

    # Draw PFC side labels
    def midpoint(A, B):
        return (A + B) / 2

    # Draw PFC side labels (offset outward)
    offset = -0.2 # オフセット量

    # Protein: P–F の中点をベースに、外側へオフセット
    mid_pf = midpoint(P, F)
    normal_pf = np.array([P[1] - F[1], F[0] - P[0]])  # 外向き法線ベクトル
    normal_pf = normal_pf / np.linalg.norm(normal_pf)
    ax.text(*(mid_pf + offset * normal_pf), "Protein", ha='center', va='center', fontsize=12, color='black')

    # Fat: F–C の中点をベースに、外側へオフセット
    mid_fc = midpoint(F, C)
    normal_fc = np.array([F[1] - C[1], C[0] - F[0]])
    normal_fc = normal_fc / np.linalg.norm(normal_fc)
    ax.text(*(mid_fc - 0.05 * normal_fc), "Fat", ha='center', va='center', fontsize=12, color='black')

    # Carbohydrate: C–P の中点をベースに、外側へオフセット
    mid_cp = midpoint(C, P)
    normal_cp = np.array([C[1] - P[1], P[0] - C[0]])
    normal_cp = normal_cp / np.linalg.norm(normal_cp)
    ax.text(*(mid_cp + offset * normal_cp), "Carbohydrate", ha='center', va='center', fontsize=12, color='black')

    # 追加：指定された頂点の三角形を描く
#    p_val = 0.20
#    c_val = 0.65
#    f_val = 0.30

    # 軸に沿った座標計算（方向に注意）
#    point_p = (1 - p_val) * F + p_val * P  # たんぱく質軸（左辺、F→P 上方向）
#    point_c = (1 - c_val) * P + c_val * C  # 炭水化物軸（右辺、P→C 下方向）
#    point_f = (1 - f_val) * C + f_val * F  # 脂質軸（底辺、C→F 右→左）

#    triangle_pts = np.array([point_p, point_c, point_f, point_p])
#    ax.plot(triangle_pts[:, 0], triangle_pts[:, 1], color='red', linewidth=2)

    # 既存の選択食品点もプロット
    total = protein + fat + carb
    a = protein / total
    b = fat / total
    c = carb / total
    x = a * P[0] + b * F[0] + c * C[0]
    y = a * P[1] + b * F[1] + c * C[1]
    ax.plot(x, y, marker='o', color='red', markersize=10)
    ax.text(x, y, f'  (P:{protein}%, F:{fat}%, C:{carb}%)', color='black', fontsize=12)

    return fig


# Streamlit rendering
st.subheader("PFC Balance Triangle")
st.pyplot(plot_pfc_triangle(protein, fat, carb))