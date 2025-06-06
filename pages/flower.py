import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# CSV読み込み
df = pd.read_csv("./data/data.csv")

# 食品選択
selected_food = st.selectbox("食品を選んでください", df["name"].unique())

# データ抽出
food = df[df["name"] == selected_food].iloc[0]

# --- PFCバランス三角形の描画 ---
def plot_pfc_triangle(carb, protein, fat):
    total = carb + protein + fat
    if total == 0:
        carb_r, protein_r, fat_r = 1/3, 1/3, 1/3
    else:
        carb_r = carb / total
        protein_r = protein / total
        fat_r = fat / total

    # 頂点（正三角形）
    vertices = np.array([
        [0.5, np.sqrt(3)/2],  # 炭水化物
        [0, 0],               # たんぱく質
        [1, 0]                # 脂質
    ])

    # PFCバランス点（重心を使った内分）
    point = carb_r * vertices[0] + protein_r * vertices[1] + fat_r * vertices[2]

    fig = go.Figure()

    # 三角形の枠
    fig.add_trace(go.Scatter(
        x=[vertices[0][0], vertices[1][0], vertices[2][0], vertices[0][0]],
        y=[vertices[0][1], vertices[1][1], vertices[2][1], vertices[0][1]],
        mode='lines',
        line=dict(color='gray'),
        showlegend=False
    ))

    # 頂点ラベル
    labels = ["炭水化物", "たんぱく質", "脂質"]
    for i in range(3):
        fig.add_trace(go.Scatter(
            x=[vertices[i][0]],
            y=[vertices[i][1]],
            mode='text',
            text=[labels[i]],
            textposition='top center',
            textfont=dict(size=12),
            showlegend=False
        ))

    # PFCバランス点
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers+text',
        marker=dict(size=12, color='red'),
        text=["バランス点"],
        textposition='bottom center',
        showlegend=False
    ))

    fig.update_layout(
        title=f"{selected_food} のPFCバランス",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=400,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white'
    )

    return fig

# PFC三角形プロット
pfc_fig = plot_pfc_triangle(food["炭水化物C"], food["たんぱく質P"], food["脂質F"])
# タイトル
st.title("PFCバランストライアングル")
st.plotly_chart(pfc_fig, use_container_width=False)

# --- フラワービジュアルの描画 ---
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
    fillcolor='lightyellow',
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
