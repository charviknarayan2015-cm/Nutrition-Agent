"""
Visualization module – builds Plotly charts for the Streamlit dashboard.
All charts use the 'plotly_white' template for a clean look.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Any

ACCENT_COLOR  = "#3b82d4"
SUCCESS_COLOR = "#22c55e"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR  = "#ef4444"
BG_COLOR      = "#f7f8fa"
CHART_TEMPLATE = "plotly_white"


# ── Macro Breakdown Donut ──────────────────────────────────────────────────────
def macro_donut_chart(protein_g: float, carbs_g: float, fat_g: float) -> go.Figure:
    """Donut chart showing macronutrient distribution in kcal."""
    protein_kcal = protein_g * 4
    carbs_kcal   = carbs_g   * 4
    fat_kcal     = fat_g     * 9
    total        = protein_kcal + carbs_kcal + fat_kcal or 1

    fig = go.Figure(go.Pie(
        labels=["Protein", "Carbohydrates", "Fat"],
        values=[protein_kcal, carbs_kcal, fat_kcal],
        hole=0.55,
        marker_colors=["#3b82d4", "#22c55e", "#f59e0b"],
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:.0f} kcal<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=BG_COLOR,
        showlegend=True,
        margin=dict(t=10, b=10, l=10, r=10),
        annotations=[dict(
            text=f"<b>{total:.0f}</b><br>kcal",
            x=0.5, y=0.5, font_size=14, showarrow=False,
        )],
        height=280,
    )
    return fig


# ── Daily Nutrient Progress Bars ───────────────────────────────────────────────
def nutrient_progress_chart(actual: dict[str, float], targets: dict[str, float]) -> go.Figure:
    """Horizontal bar chart showing % of daily targets achieved."""
    nutrients_display = {
        "calories":          ("Calories", "kcal"),
        "protein_g":         ("Protein", "g"),
        "carbohydrates_g":   ("Carbs", "g"),
        "fat_g":             ("Fat", "g"),
        "fiber_g":           ("Fiber", "g"),
        "sodium_mg":         ("Sodium", "mg"),
        "calcium_mg":        ("Calcium", "mg"),
        "iron_mg":           ("Iron", "mg"),
        "vitamin_c_mg":      ("Vitamin C", "mg"),
    }

    labels, pct_values, colors, hover = [], [], [], []

    for key, (label, unit) in nutrients_display.items():
        target = targets.get(key, 1)
        actual_val = actual.get(key, 0)
        pct = min((actual_val / target * 100), 150) if target > 0 else 0
        labels.append(f"{label}")
        pct_values.append(pct)
        hover.append(f"{actual_val:.1f}{unit} / {target}{unit}")
        if pct < 50:
            colors.append(DANGER_COLOR)
        elif pct < 80:
            colors.append(WARNING_COLOR)
        elif pct <= 110:
            colors.append(SUCCESS_COLOR)
        else:
            colors.append(DANGER_COLOR)

    fig = go.Figure(go.Bar(
        x=pct_values,
        y=labels,
        orientation="h",
        marker_color=colors,
        customdata=hover,
        hovertemplate="<b>%{y}</b><br>%{customdata}<br>%{x:.0f}% of target<extra></extra>",
        text=[f"{p:.0f}%" for p in pct_values],
        textposition="outside",
    ))

    # Reference line at 100%
    fig.add_vline(x=100, line_dash="dash", line_color="#6b7280", line_width=1.5,
                  annotation_text="100%", annotation_position="top")

    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=BG_COLOR,
        xaxis=dict(range=[0, 160], title="% of Daily Target"),
        yaxis=dict(autorange="reversed"),
        height=340,
        margin=dict(t=20, b=30, l=100, r=60),
    )
    return fig


# ── Weekly Calorie Trend ───────────────────────────────────────────────────────
def weekly_calorie_trend(weekly_data: list[dict]) -> go.Figure:
    """
    Line + bar combo showing daily calorie intake vs target.
    weekly_data: list of {'day': str, 'calories': float, 'target': float, 'protein': float}
    """
    days      = [d["day"]     for d in weekly_data]
    cals      = [d.get("calories", 0) for d in weekly_data]
    targets   = [d.get("target", 2000) for d in weekly_data]
    proteins  = [d.get("protein", 0)   for d in weekly_data]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=days, y=cals, name="Calories",
        marker_color=ACCENT_COLOR, opacity=0.75,
        hovertemplate="<b>%{x}</b><br>Calories: %{y:.0f} kcal<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=days, y=targets, name="Target",
        line=dict(color=DANGER_COLOR, dash="dash", width=2),
        mode="lines",
        hovertemplate="Target: %{y:.0f} kcal<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=days, y=proteins, name="Protein (g)",
        line=dict(color=SUCCESS_COLOR, width=2),
        mode="lines+markers",
        hovertemplate="Protein: %{y:.0f}g<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=BG_COLOR,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
        margin=dict(t=30, b=40, l=60, r=60),
    )
    fig.update_yaxes(title_text="Calories (kcal)", secondary_y=False)
    fig.update_yaxes(title_text="Protein (g)",     secondary_y=True, showgrid=False)
    return fig


# ── Nutrient Radar Chart ───────────────────────────────────────────────────────
def nutrient_radar_chart(actual: dict[str, float], targets: dict[str, float]) -> go.Figure:
    """Radar/spider chart comparing actual intake to targets."""
    keys = ["protein_g", "carbohydrates_g", "fat_g", "fiber_g", "vitamin_c_mg", "calcium_mg", "iron_mg"]
    labels_map = {
        "protein_g": "Protein", "carbohydrates_g": "Carbs",
        "fat_g": "Fat", "fiber_g": "Fiber",
        "vitamin_c_mg": "Vit C", "calcium_mg": "Calcium", "iron_mg": "Iron",
    }
    categories = [labels_map[k] for k in keys]
    actual_pct  = [min((actual.get(k, 0) / targets.get(k, 1)) * 100, 150) for k in keys]
    target_line = [100] * len(keys)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=actual_pct + [actual_pct[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Actual",
        fillcolor="rgba(59,130,212,0.25)",
        line=dict(color=ACCENT_COLOR, width=2),
    ))
    fig.add_trace(go.Scatterpolar(
        r=target_line + [target_line[0]],
        theta=categories + [categories[0]],
        mode="lines",
        name="Target (100%)",
        line=dict(color=DANGER_COLOR, dash="dot", width=1.5),
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=BG_COLOR,
        polar=dict(radialaxis=dict(visible=True, range=[0, 150])),
        showlegend=True,
        height=320,
        margin=dict(t=30, b=20, l=20, r=20),
    )
    return fig


# ── Food Comparison Bar Chart ──────────────────────────────────────────────────
def food_comparison_chart(food_a: dict, food_b: dict) -> go.Figure:
    """Side-by-side bar chart for two food items."""
    nutrients = ["calories", "protein_g", "carbohydrates_g", "fat_g", "fiber_g"]
    labels    = ["Calories\n(kcal)", "Protein\n(g)", "Carbs\n(g)", "Fat\n(g)", "Fiber\n(g)"]

    val_a = [food_a.get(n, 0) for n in nutrients]
    val_b = [food_b.get(n, 0) for n in nutrients]

    fig = go.Figure(data=[
        go.Bar(name=food_a.get("name", "Food A"), x=labels, y=val_a,
               marker_color=ACCENT_COLOR,
               hovertemplate="<b>" + food_a.get("name", "A") + "</b><br>%{y:.1f}<extra></extra>"),
        go.Bar(name=food_b.get("name", "Food B"), x=labels, y=val_b,
               marker_color=WARNING_COLOR,
               hovertemplate="<b>" + food_b.get("name", "B") + "</b><br>%{y:.1f}<extra></extra>"),
    ])
    fig.update_layout(
        barmode="group",
        template=CHART_TEMPLATE,
        paper_bgcolor=BG_COLOR,
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
    )
    return fig


# ── Calorie Breakdown Per Meal ─────────────────────────────────────────────────
def meal_calorie_breakdown(log_entries: list[dict]) -> go.Figure:
    """Pie chart breaking calorie intake by meal type for the day."""
    if not log_entries:
        return go.Figure()

    meal_totals: dict[str, float] = {}
    for entry in log_entries:
        mtype = entry.get("meal_type", "meal").title()
        meal_totals[mtype] = meal_totals.get(mtype, 0) + entry.get("calories", 0)

    colors = [ACCENT_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR, "#7c5cd8"]
    fig = go.Figure(go.Pie(
        labels=list(meal_totals.keys()),
        values=list(meal_totals.values()),
        marker_colors=colors[:len(meal_totals)],
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>%{value:.0f} kcal (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=BG_COLOR,
        height=260,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return fig


# ── Deficiency Alert Gauge ─────────────────────────────────────────────────────
def deficiency_gauge(nutrient_name: str, actual: float, target: float, unit: str) -> go.Figure:
    """Gauge chart showing how close a nutrient is to its daily target."""
    pct = min((actual / target * 100), 150) if target > 0 else 0

    color = SUCCESS_COLOR if 80 <= pct <= 110 else (WARNING_COLOR if 50 <= pct < 80 else DANGER_COLOR)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=actual,
        delta={"reference": target, "valueformat": ".1f"},
        title={"text": f"{nutrient_name}<br><span style='font-size:0.8em'>{actual:.1f} / {target}{unit}</span>"},
        gauge={
            "axis": {"range": [0, target * 1.5], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, target * 0.5], "color": "#fee2e2"},
                {"range": [target * 0.5, target * 0.8], "color": "#fef3c7"},
                {"range": [target * 0.8, target * 1.1], "color": "#dcfce7"},
                {"range": [target * 1.1, target * 1.5], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": DANGER_COLOR, "width": 3},
                "thickness": 0.8,
                "value": target,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        height=220,
        margin=dict(t=40, b=20, l=30, r=30),
    )
    return fig
