import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Load data
df = pd.read_csv("./data/data.csv")

# Food selection
food_names = df['name'].unique()
selected_food = st.selectbox("Select a food", food_names)

# Get data for selected food
data = df[df["name"] == selected_food].iloc[0]
# データ抽出
food = df[df["name"] == selected_food].iloc[0]

color = (0,0,0)
backColor = color

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
#    fig, ax = plt.subplots(figsize=(6, 6))
    global backColor

    # Gradient fill
    resolution = 300
    xs, ys, colors = [], [], []
    
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
            
            xs.append(x)
            ys.append(y)
            colors.append(color)    
#            ax.plot(x, y, marker='o', markersize=0.5, color=color, alpha=0.9)

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, c=colors, s=1, alpha=0.9, edgecolors='none')
    ax.set_aspect('equal')
    ax.axis('off')
    
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

    # 既存の選択食品点もプロット
    total = protein + fat + carb
    a = protein / total
    b = fat / total
    c = carb / total
    x = a * P[0] + b * F[0] + c * C[0]
    y = a * P[1] + b * F[1] + c * C[1]
    ax.plot(x, y, marker='o', color='red', markersize=10)
    ax.text(x, y, f'  (P:{protein}%, F:{fat}%, C:{carb}%)', color='black', fontsize=12)

    color_p = np.array([1.0, 0.2, 1.0]) * a
    color_f = np.array([0.2, 1.0, 1.0]) * b
    color_c = np.array([1.0, 1.0, 0.2]) * c
    backColor = color_p + color_f + color_c
    
    return fig


# Streamlit rendering
st.subheader("PFC Balance Triangle")
st.pyplot(plot_pfc_triangle(protein, fat, carb))

# --- フラワービジュアルの描画 ---
from matplotlib.colors import to_rgba

# RGBのfloat値（例: backColor = (0.8, 0.9, 0.5)）
rgba = to_rgba(backColor)
rgba_str = f'rgba({rgba[0]*255:.0f}, {rgba[1]*255:.0f}, {rgba[2]*255:.0f}, {rgba[3]:.2f})'

nutrient_categories = {
    "炭水化物": ("三大栄養素", "yellow"),         # 黄色
    "たんぱく質": ("三大栄養素", "red"),         # 赤
    "脂質": ("三大栄養素", "blue"),             # 青
    "ビタミンA": ("ビタミン", "magenta"),       # 華やか
    "ビタミンC": ("ビタミン", "violet"),        # 華やか
    "ビタミンD": ("ビタミン", "orange"),        # 華やか
    "カルシウム": ("ミネラル", "lightgreen"),   # 維持
    "鉄": ("ミネラル", "mediumseagreen"),       # 維持
    "カリウム": ("ミネラル", "cyan"),           # 華やか
    "食塩上限2.5g": ("その他", "orchid"),       # 華やか
    "食物繊維": ("その他", "deeppink")          # 華やか
}

petal_nutrients = list(nutrient_categories.keys())
petal_values = [min(max(food[n] / 100, 0), 2.0) for n in petal_nutrients]

num_petals = len(petal_nutrients)
angles = np.linspace(0, 2 * np.pi, num_petals, endpoint=False)

energy_ratio = min(max(food["エネルギー"] / 100, 0), 2.0)
energy_radius = 0.3 + energy_ratio * 0.2

fig = go.Figure()

# 花びら描画
for i, (nutrient, value) in enumerate(zip(petal_nutrients, petal_values)):
    angle = angles[i]
    color = nutrient_categories[nutrient][1]

    petal_center_x = energy_radius * np.cos(angle)
    petal_center_y = energy_radius * np.sin(angle)

    petal_length = value * 0.5
    petal_width = 0.1

    t = np.linspace(0, 2 * np.pi, 100)
    x_ellipse = petal_length * np.cos(t)
    y_ellipse = petal_width * np.sin(t)

    x_rot = x_ellipse * np.cos(angle) - y_ellipse * np.sin(angle)
    y_rot = x_ellipse * np.sin(angle) + y_ellipse * np.cos(angle)

    x_final = x_rot + petal_center_x
    y_final = y_rot + petal_center_y

    fig.add_trace(go.Scatter(
        x=x_final,
        y=y_final,
        fill='toself',
        mode='lines',
        line=dict(color=color),
        fillcolor=color,
        showlegend=False
    ))

    label_x = petal_center_x + (petal_length + 0.1) * np.cos(angle)
    label_y = petal_center_y + (petal_length + 0.1) * np.sin(angle)
    label_text = f"{nutrient}: {food[nutrient]:.1f}%"

    fig.add_trace(go.Scatter(
        x=[label_x],
        y=[label_y],
        mode='text',
        text=[label_text],
        textposition='middle center',
        textfont=dict(size=10),
        showlegend=False
    ))

# 中心エネルギー円
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = energy_radius * np.cos(theta)
circle_y = energy_radius * np.sin(theta)

fig.add_trace(go.Scatter(
    x=circle_x,
    y=circle_y,
    mode='lines',
    fill='toself',
#    fillcolor='lightyellow',
    fillcolor= rgba_str,
    line=dict(color='darkorange'),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=[0],
    y=[0],
    mode='text',
    text=[f"Energy<br>{food['エネルギー']:.1f}%<br>一食767kcalあたり"],
    textfont=dict(size=14, color='black'),
    textposition='middle center',
    showlegend=False
))

fig.update_layout(
    title=f"{selected_food} の栄養フラワービジュアル",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    width=650,
    height=650,
    margin=dict(l=20, r=20, t=60, b=20),
    plot_bgcolor='white'
)

# フラワービジュアル描画
# タイトル
st.title("🌸 フラワービジュアル")
st.plotly_chart(fig, use_container_width=False)
