"""
AI-Powered Nutrition Agent – Streamlit Application
====================================================
Multi-agent system powered by IBM watsonx.ai (Llama-3 70B)

Agents:
  1. Nutrition Knowledge Agent  – RAG-based food & nutrient facts
  2. Diet Recommendation Agent  – Personalized meal plans
  3. Health Advisory Agent      – Preventive & disease-specific guidance
  4. Food Log & Feedback Agent  – Meal logging + instant analysis

Run:  streamlit run nutrition_agent/app.py
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NutriAgent – AI Nutrition Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f7f8fa; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #1e293b; }
    section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stMultiSelect label { color: #94a3b8 !important; }

    /* Cards */
    .agent-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .agent-tag {
        display: inline-block;
        background: #eff6ff;
        color: #3b82d4;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        border: 1px solid #bfdbfe;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1f2328; }
    .metric-card .label { font-size: 0.8rem; color: #57606a; margin-top: 2px; }
    .status-ok   { color: #22c55e; }
    .status-warn { color: #f59e0b; }
    .status-bad  { color: #ef4444; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82d4 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .app-header h1 { margin: 0; font-size: 1.8rem; }
    .app-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Answer blocks */
    .answer-block {
        background: #f8fafc;
        border-left: 4px solid #3b82d4;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.8rem;
        white-space: pre-wrap;
        font-size: 0.92rem;
        line-height: 1.7;
    }
    div[data-testid="stButton"] > button {
        background-color: #3b82d4;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.45rem 1.2rem;
    }
    div[data-testid="stButton"] > button:hover { background-color: #2563eb; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "meal_log":          [],
        "chat_history":      [],
        "user_profile":      {},
        "weekly_data":       [],
        "generated_plan":    None,
        "active_tab":        "🏠 Dashboard",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ── Import agents (lazy, after page config) ────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to IBM watsonx.ai…")
def _load_agents():
    from nutrition_agent.agents.nutrition_knowledge_agent import (
        answer_nutrition_question, get_food_nutrition_profile, compare_foods
    )
    from nutrition_agent.agents.diet_recommendation_agent import (
        generate_meal_plan, suggest_meal_swap
    )
    from nutrition_agent.agents.health_advisory_agent import (
        get_health_advice, get_preventive_tips, assess_diet_risk
    )
    from nutrition_agent.agents.food_log_agent import (
        log_meal, analyze_daily_log, get_meal_alternatives,
        reset_log, get_log
    )
    from nutrition_agent.visualization import (
        macro_donut_chart, nutrient_progress_chart, weekly_calorie_trend,
        nutrient_radar_chart, food_comparison_chart, meal_calorie_breakdown,
        deficiency_gauge,
    )
    return {
        "answer_nutrition_question":  answer_nutrition_question,
        "get_food_nutrition_profile": get_food_nutrition_profile,
        "compare_foods":              compare_foods,
        "generate_meal_plan":         generate_meal_plan,
        "suggest_meal_swap":          suggest_meal_swap,
        "get_health_advice":          get_health_advice,
        "get_preventive_tips":        get_preventive_tips,
        "assess_diet_risk":           assess_diet_risk,
        "log_meal":                   log_meal,
        "analyze_daily_log":          analyze_daily_log,
        "get_meal_alternatives":      get_meal_alternatives,
        "reset_log":                  reset_log,
        "get_log":                    get_log,
        # charts
        "macro_donut_chart":          macro_donut_chart,
        "nutrient_progress_chart":    nutrient_progress_chart,
        "weekly_calorie_trend":       weekly_calorie_trend,
        "nutrient_radar_chart":       nutrient_radar_chart,
        "food_comparison_chart":      food_comparison_chart,
        "meal_calorie_breakdown":     meal_calorie_breakdown,
        "deficiency_gauge":           deficiency_gauge,
    }


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🥗 NutriAgent — AI-Powered Nutrition Assistant</h1>
  <p>Personalized dietary guidance powered by IBM watsonx.ai (Llama-3 70B) · Multi-Agent RAG System</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar – User Profile ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Your Profile")

    age          = st.slider("Age",              10, 90, 30)
    gender       = st.selectbox("Gender",        ["Male", "Female", "Other"])
    weight_kg    = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, step=0.5)
    height_cm    = st.number_input("Height (cm)", 120.0, 220.0, 170.0, step=0.5)
    activity     = st.selectbox("Activity Level", ["sedentary", "light", "moderate", "active", "very_active"])
    goal         = st.selectbox("Fitness Goal",   ["weight loss", "muscle gain", "maintenance", "heart health", "diabetes management"])
    cuisine      = st.selectbox("Cuisine Preference", [
        "No specific preference", "Indian", "Mediterranean", "East Asian",
        "American", "Middle Eastern", "Latin American", "African", "European"
    ])
    conditions   = st.multiselect("Health Conditions", [
        "diabetes", "hypertension", "heart disease", "obesity",
        "kidney disease", "celiac disease", "lactose intolerance",
        "thyroid disorder", "PCOS", "IBS"
    ])
    allergies    = st.multiselect("Allergies / Intolerances", [
        "peanuts", "tree nuts", "gluten", "dairy", "eggs",
        "shellfish", "soy", "fish"
    ])
    preferences  = st.multiselect("Dietary Preferences", [
        "vegetarian", "vegan", "pescatarian", "keto", "paleo",
        "low-carb", "low-sodium", "high-protein", "halal", "kosher"
    ])

    st.session_state["user_profile"] = {
        "age": age, "gender": gender.lower(),
        "weight_kg": weight_kg, "height_cm": height_cm,
        "activity_level": activity, "goal": goal,
        "cuisine": cuisine, "conditions": conditions,
        "allergies": allergies, "preferences": preferences,
    }

    st.markdown("---")
    st.markdown("**Powered by**")
    st.markdown("🤖 IBM watsonx.ai")
    st.markdown("🦙 Llama-3 70B Instruct")
    st.markdown("📚 RAG + USDA Data")


# ── Navigation Tabs ────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Dashboard",
    "🔍 Nutrition Knowledge",
    "🍽️ Meal Plan",
    "📋 Food Log",
    "🏥 Health Advisory",
    "📊 Analytics",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 – Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 📈 Today's Nutrition Overview")

    agents = _load_agents()
    log    = agents["get_log"]()
    up     = st.session_state["user_profile"]

    # ── Quick stats row ────────────────────────────────────────────────────────
    total_cals = sum(e.get("calories", 0)                       for e in log)
    total_prot = sum(e.get("totals", {}).get("protein_g", 0)    for e in log)
    total_carb = sum(e.get("totals", {}).get("carbohydrates_g", 0) for e in log)
    total_fat  = sum(e.get("totals", {}).get("fat_g", 0)        for e in log)
    total_fib  = sum(e.get("totals", {}).get("fiber_g", 0)      for e in log)

    c1, c2, c3, c4, c5 = st.columns(5)
    def _card(col, value, label, unit="", color=""):
        col.markdown(f"""
        <div class="metric-card">
          <div class="value {color}">{value:.0f}<span style="font-size:0.9rem">{unit}</span></div>
          <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    _card(c1, total_cals, "Calories", " kcal")
    _card(c2, total_prot, "Protein",  "g",    "status-ok" if total_prot >= 30 else "status-warn")
    _card(c3, total_carb, "Carbs",    "g")
    _card(c4, total_fat,  "Fat",      "g")
    _card(c5, total_fib,  "Fiber",    "g",    "status-ok" if total_fib >= 15 else "status-warn")

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### 🍩 Macro Distribution")
        if total_prot + total_carb + total_fat > 0:
            fig = agents["macro_donut_chart"](total_prot, total_carb, total_fat)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Log your first meal to see macro distribution.")

    with col_r:
        st.markdown("#### 📊 Nutrient Progress")
        from nutrition_agent.config import DAILY_REFERENCE_VALUES
        actual = {
            "calories": total_cals, "protein_g": total_prot,
            "carbohydrates_g": total_carb, "fat_g": total_fat,
            "fiber_g": total_fib,
            "sodium_mg": sum(e.get("totals", {}).get("sodium_mg", 0) for e in log),
            "calcium_mg": sum(e.get("totals", {}).get("calcium_mg", 0) for e in log),
            "iron_mg": sum(e.get("totals", {}).get("iron_mg", 0) for e in log),
            "vitamin_c_mg": sum(e.get("totals", {}).get("vitamin_c_mg", 0) for e in log),
        }
        fig = agents["nutrient_progress_chart"](actual, DAILY_REFERENCE_VALUES)
        st.plotly_chart(fig, width="stretch")

    # ── Today's log summary ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📝 Today's Meal Log")
    if log:
        for entry in log:
            with st.expander(
                f"🍴 {entry['meal_type'].upper()} — {entry['description'][:60]}…  "
                f"({entry['calories']:.0f} kcal)",
                expanded=False
            ):
                st.markdown(f"**Logged at:** {entry['timestamp'][:16].replace('T', ' ')}")
                t = entry.get("totals", {})
                st.markdown(
                    f"**Nutrients:** Protein {t.get('protein_g',0):.1f}g | "
                    f"Carbs {t.get('carbohydrates_g',0):.1f}g | "
                    f"Fat {t.get('fat_g',0):.1f}g | "
                    f"Fiber {t.get('fiber_g',0):.1f}g"
                )
                st.markdown(f"**AI Feedback:**\n{entry.get('feedback','')}")
    else:
        st.info("📌 No meals logged yet. Head to the **📋 Food Log** tab to get started!")

    # ── Agent status ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 Active Agents")
    ac1, ac2, ac3, ac4 = st.columns(4)
    for col, name, icon in [
        (ac1, "Nutrition Knowledge Agent", "🔬"),
        (ac2, "Diet Recommendation Agent", "🍽️"),
        (ac3, "Health Advisory Agent",     "🏥"),
        (ac4, "Food Log & Feedback Agent", "📋"),
    ]:
        col.markdown(f"""
        <div class="agent-card" style="text-align:center;padding:0.8rem">
          <div style="font-size:1.8rem">{icon}</div>
          <div style="font-size:0.8rem;font-weight:600;color:#1f2328;margin-top:0.3rem">{name}</div>
          <div style="color:#22c55e;font-size:0.75rem;margin-top:0.2rem">● Active</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Nutrition Knowledge
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 🔬 Nutrition Knowledge Agent")
    st.markdown(
        "Ask any nutrition question, look up a food's nutrients, or compare two foods — "
        "all powered by RAG retrieval over curated nutrition science."
    )

    agents = _load_agents()

    knowledge_mode = st.radio(
        "Mode",
        ["❓ Ask a Question", "🥦 Look Up a Food", "⚖️ Compare Two Foods"],
        horizontal=True,
    )

    if knowledge_mode == "❓ Ask a Question":
        question = st.text_area(
            "Nutrition Question",
            placeholder="e.g. What are the best foods for iron deficiency? How much protein do I need for muscle gain?",
            height=100,
        )
        if st.button("🔍 Get Answer", key="ask_btn") and question.strip():
            with st.spinner("🤖 Nutrition Knowledge Agent is retrieving an answer…"):
                result = agents["answer_nutrition_question"](question)
            st.markdown('<div class="agent-tag">🔬 Nutrition Knowledge Agent</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-block">{result["answer"]}</div>', unsafe_allow_html=True)
            if result["sources"]:
                st.markdown(f"**📚 Sources:** {' · '.join(result['sources'])}")
            if result.get("food_nutrients"):
                st.markdown("**🥘 Database Match:**")
                st.json(result["food_nutrients"])

    elif knowledge_mode == "🥦 Look Up a Food":
        col_a, col_b = st.columns([3, 1])
        food_name = col_a.text_input("Food Name", placeholder="e.g. Salmon, Lentils, Blueberries")
        lookup_btn = col_b.button("🔍 Look Up", key="lookup_btn")

        if lookup_btn and food_name.strip():
            with st.spinner(f"Fetching nutrition profile for {food_name}…"):
                result = agents["get_food_nutrition_profile"](food_name)

            st.markdown('<div class="agent-tag">🔬 Nutrition Knowledge Agent</div>', unsafe_allow_html=True)

            if result["nutrients"]:
                n = result["nutrients"]
                st.markdown(f"#### {n['name']} — Nutritional Profile (per 100g serving)")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Calories",  f"{n.get('calories',0):.0f} kcal")
                m2.metric("Protein",   f"{n.get('protein_g',0):.1f}g")
                m3.metric("Carbs",     f"{n.get('carbohydrates_g',0):.1f}g")
                m4.metric("Fat",       f"{n.get('fat_g',0):.1f}g")
                m5.metric("Fiber",     f"{n.get('fiber_g',0):.1f}g")
                m6.metric("Sodium",    f"{n.get('sodium_mg',0):.0f}mg")

                st.markdown("**Micronutrients:**")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Vitamin C",  f"{n.get('vitamin_c_mg',0):.1f}mg")
                mc2.metric("Calcium",    f"{n.get('calcium_mg',0):.0f}mg")
                mc3.metric("Iron",       f"{n.get('iron_mg',0):.2f}mg")

            st.markdown(f'<div class="answer-block">{result["analysis"]}</div>', unsafe_allow_html=True)

    else:  # Compare Two Foods
        c1, c2 = st.columns(2)
        food_a  = c1.text_input("Food A", placeholder="e.g. Salmon")
        food_b  = c2.text_input("Food B", placeholder="e.g. Chicken Breast")
        if st.button("⚖️ Compare", key="compare_btn") and food_a.strip() and food_b.strip():
            with st.spinner(f"Comparing {food_a} vs {food_b}…"):
                result = agents["compare_foods"](food_a, food_b)

            st.markdown('<div class="agent-tag">🔬 Nutrition Knowledge Agent</div>', unsafe_allow_html=True)

            if result["food_a"]["nutrients"] and result["food_b"]["nutrients"]:
                fig = agents["food_comparison_chart"](
                    result["food_a"]["nutrients"],
                    result["food_b"]["nutrients"],
                )
                st.plotly_chart(fig, width="stretch")

            st.markdown(f'<div class="answer-block">{result["comparison"]}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Meal Plan
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 🍽️ Diet Recommendation Agent")
    st.markdown("Get a personalized **7-day meal plan** tailored to your profile, health conditions, and goals.")

    agents = _load_agents()
    up     = st.session_state["user_profile"]

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("#### Your profile (from sidebar) will be used:")
        if up:
            st.markdown(
                f"- **Age/Gender:** {up.get('age')} / {up.get('gender','').title()}\n"
                f"- **Goal:** {up.get('goal','').title()}\n"
                f"- **Activity:** {up.get('activity_level','').title()}\n"
                f"- **Conditions:** {', '.join(up.get('conditions', [])) or 'None'}\n"
                f"- **Allergies:** {', '.join(up.get('allergies', [])) or 'None'}\n"
                f"- **Preferences:** {', '.join(up.get('preferences', [])) or 'None'}\n"
                f"- **Cuisine:** {up.get('cuisine','')}"
            )

    with col_r:
        st.markdown("#### Calorie Preview")
        if up.get("weight_kg") and up.get("height_cm"):
            # Calculate preview
            from nutrition_agent.agents.diet_recommendation_agent import _calculate_bmr, _calculate_tdee, _adjust_for_goal
            bmr_val  = _calculate_bmr(up["age"], up["gender"], up["weight_kg"], up["height_cm"])
            tdee_val = _calculate_tdee(bmr_val, up.get("activity_level", "moderate"))
            target   = round(_adjust_for_goal(tdee_val, up.get("goal", "maintenance")))
            st.metric("BMR",         f"{bmr_val:.0f} kcal")
            st.metric("TDEE",        f"{tdee_val:.0f} kcal")
            st.metric("Daily Target", f"{target} kcal")

    if st.button("🚀 Generate My 7-Day Meal Plan", key="gen_plan_btn"):
        with st.spinner("🍽️ Diet Recommendation Agent is crafting your personalized meal plan…"):
            result = agents["generate_meal_plan"](up)
            st.session_state["generated_plan"] = result

    if st.session_state.get("generated_plan"):
        plan = st.session_state["generated_plan"]
        st.markdown("---")
        st.markdown('<div class="agent-tag">🍽️ Diet Recommendation Agent</div>', unsafe_allow_html=True)

        t = plan["targets"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Daily Calories", f"{t['calories']} kcal")
        m2.metric("Protein",        f"{t['protein_g']}g")
        m3.metric("Carbs",          f"{t['carbs_g']}g")
        m4.metric("Fat",            f"{t['fat_g']}g")

        st.markdown(f'<div class="answer-block">{plan["meal_plan"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔄 Meal Swap Suggestion")
    col_ms1, col_ms2 = st.columns(2)
    current_meal = col_ms1.text_input("Meal to replace", placeholder="e.g. Fried chicken and chips")
    swap_reason  = col_ms2.text_input("Reason for swap",  placeholder="e.g. Too high in fat, I'm diabetic")
    if st.button("💡 Suggest Alternatives", key="swap_btn") and current_meal:
        with st.spinner("Finding healthier alternatives…"):
            swap_result = agents["suggest_meal_swap"](current_meal, swap_reason, up)
        st.markdown('<div class="agent-tag">🍽️ Diet Recommendation Agent</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-block">{swap_result["alternatives"]}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – Food Log
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 📋 Food Log & Feedback Agent")
    st.markdown("Log what you eat and get **instant AI nutritional analysis** with personalized feedback.")

    agents = _load_agents()

    col_log, col_type = st.columns([3, 1])
    meal_input = col_log.text_area(
        "What did you eat?",
        placeholder="e.g. 2 boiled eggs, oatmeal with banana and almond milk, black coffee",
        height=80,
    )
    meal_type_sel = col_type.selectbox("Meal Type", ["breakfast", "lunch", "dinner", "snack", "other"])

    col_b1, col_b2 = st.columns([1, 5])
    log_btn    = col_b1.button("📝 Log Meal", key="log_btn")
    reset_btn  = col_b2.button("🗑️ Clear Log", key="reset_btn")

    if reset_btn:
        agents["reset_log"]()
        st.session_state["meal_log"] = []
        st.rerun()

    if log_btn and meal_input.strip():
        with st.spinner("🤖 Analyzing your meal…"):
            result = agents["log_meal"](
                meal_input.strip(),
                meal_type_sel,
                {"calories": 2000, "protein_g": 50, "carbohydrates_g": 275, "fat_g": 78},
            )
            st.session_state["meal_log"].append(result)

        st.markdown('<div class="agent-tag">📋 Food Log & Feedback Agent</div>', unsafe_allow_html=True)
        n = result["nutrients"]
        nm1, nm2, nm3, nm4 = st.columns(4)
        nm1.metric("Calories",  f"{result['calories']:.0f} kcal")
        nm2.metric("Protein",   f"{n.get('protein_g',0):.1f}g")
        nm3.metric("Carbs",     f"{n.get('carbohydrates_g',0):.1f}g")
        nm4.metric("Fat",       f"{n.get('fat_g',0):.1f}g")
        st.markdown(f'<div class="answer-block">{result["feedback"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Daily analysis
    log_entries = agents["get_log"]()
    if log_entries:
        st.markdown(f"#### 📊 Daily Analysis ({len(log_entries)} meals logged)")

        if st.button("🔍 Analyze My Day", key="analyze_day_btn"):
            with st.spinner("Analyzing your full day's nutrition…"):
                analysis = agents["analyze_daily_log"]()

            st.markdown('<div class="agent-tag">📋 Food Log & Feedback Agent</div>', unsafe_allow_html=True)

            # Risk flags
            if analysis.get("risk_flags"):
                st.markdown("**⚠️ Risk Flags:**")
                for flag in analysis["risk_flags"]:
                    st.warning(flag)

            # Totals table
            t = analysis.get("totals", {})
            st.markdown("**Daily Totals:**")
            totals_df = pd.DataFrame([{
                "Nutrient": k.replace("_", " ").title(),
                "Amount": f"{v:.1f}",
            } for k, v in t.items() if v > 0])
            if not totals_df.empty:
                st.dataframe(totals_df, width="stretch", hide_index=True)

            st.markdown(f'<div class="answer-block">{analysis["analysis"]}</div>', unsafe_allow_html=True)

        # Meal breakdown pie
        st.markdown("#### 🍕 Calorie Distribution by Meal")
        fig_pie = agents["meal_calorie_breakdown"](log_entries)
        if fig_pie.data:
            st.plotly_chart(fig_pie, width="stretch")

    st.markdown("---")
    st.markdown("#### 🔄 Get Meal Alternative")
    alt_meal   = st.text_input("Meal to replace", placeholder="e.g. Double cheeseburger with fries", key="alt_meal_input")
    alt_reason = st.text_input("Why?", placeholder="e.g. Too many calories, high sodium", key="alt_reason_input")
    if st.button("💡 Find Healthier Alternative", key="alt_btn") and alt_meal:
        with st.spinner("Searching for healthier alternatives…"):
            alt_result = agents["get_meal_alternatives"](alt_meal, alt_reason)
        st.markdown('<div class="agent-tag">📋 Food Log & Feedback Agent</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-block">{alt_result["alternatives"]}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – Health Advisory
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 🏥 Health Advisory Agent")
    st.markdown(
        "Get evidence-based dietary guidance for **chronic disease management** and **preventive health**."
    )

    agents = _load_agents()

    advisory_mode = st.radio(
        "Advisory Mode",
        ["🩺 Condition-Specific Guidance", "🛡️ Preventive Health Plan", "⚠️ Diet Risk Assessment"],
        horizontal=True,
    )

    if advisory_mode == "🩺 Condition-Specific Guidance":
        condition_options = [
            "diabetes", "hypertension", "heart disease", "obesity",
            "kidney disease", "celiac disease", "gut health", "osteoporosis",
            "anemia", "thyroid disease", "PCOS", "cancer prevention"
        ]
        selected_condition = st.selectbox("Select Health Condition", condition_options)
        medications        = st.text_input("Current Medications (optional)", placeholder="e.g. Metformin, Lisinopril")

        user_info = {
            **st.session_state["user_profile"],
            "medications": [m.strip() for m in medications.split(",")] if medications else [],
        }

        if st.button("🏥 Get Health Advisory", key="health_adv_btn"):
            with st.spinner(f"🏥 Health Advisory Agent is preparing guidance for {selected_condition}…"):
                result = agents["get_health_advice"](selected_condition, user_info)

            st.markdown('<div class="agent-tag">🏥 Health Advisory Agent</div>', unsafe_allow_html=True)

            if result.get("guideline"):
                g = result["guideline"]
                with st.expander("📋 Clinical Guidelines Summary", expanded=True):
                    col_g1, col_g2 = st.columns(2)
                    col_g1.markdown(f"**✅ Prioritize:** {g['foods_to_eat']}")
                    col_g2.markdown(f"**❌ Avoid:** {g['foods_to_avoid']}")
                    st.markdown(f"**🎯 Clinical Target:** {g['clinical_target']}")

            st.markdown(f'<div class="answer-block">{result["advice"]}</div>', unsafe_allow_html=True)

    elif advisory_mode == "🛡️ Preventive Health Plan":
        risk_options = [
            "cardiovascular disease", "type 2 diabetes", "obesity",
            "osteoporosis", "cognitive decline", "cancer", "hypertension",
            "metabolic syndrome", "vitamin deficiency", "poor gut health"
        ]
        risk_factors = st.multiselect("Select Risk Factors to Address", risk_options)
        up           = st.session_state["user_profile"]

        if st.button("🛡️ Generate Preventive Plan", key="prev_plan_btn") and risk_factors:
            with st.spinner("Generating your preventive nutrition plan…"):
                result = agents["get_preventive_tips"](risk_factors, up)

            st.markdown('<div class="agent-tag">🏥 Health Advisory Agent</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-block">{result["preventive_plan"]}</div>', unsafe_allow_html=True)

    else:  # Diet Risk Assessment
        st.markdown("This will analyze your **logged meals** for nutritional risk patterns.")
        log = agents["get_log"]()

        if not log:
            st.warning("⚠️ No meals logged yet. Go to the **📋 Food Log** tab to log your meals first.")
        else:
            if st.button("⚠️ Assess My Diet Risks", key="risk_btn"):
                # Format log entries for risk assessment
                formatted_log = [
                    {
                        "food":     e.get("description", "Unknown"),
                        "calories": e.get("calories", 0),
                        "nutrients": e.get("totals", {}),
                    }
                    for e in log
                ]
                with st.spinner("Assessing dietary risk patterns…"):
                    result = agents["assess_diet_risk"](formatted_log)

                st.markdown('<div class="agent-tag">🏥 Health Advisory Agent</div>', unsafe_allow_html=True)

                if result["risk_flags"]:
                    for flag in result["risk_flags"]:
                        st.error(f"🚨 {flag}")
                else:
                    st.success("✅ No critical nutritional risk flags detected.")

                t = result["totals"]
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Total Sodium",  f"{t.get('sodium_mg',0):.0f}mg", delta_color="inverse",
                              delta=f"Limit 2300mg")
                col_r2.metric("Total Sugar",   f"{t.get('sugar_g',0):.1f}g",   delta_color="inverse",
                              delta=f"Limit 50g")
                col_r3.metric("Total Fiber",   f"{t.get('fiber_g',0):.1f}g",   delta_color="normal",
                              delta=f"Target 28g")

                st.markdown(f'<div class="answer-block">{result["assessment"]}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – Analytics
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### 📊 Nutrition Analytics Dashboard")

    agents = _load_agents()
    log    = agents["get_log"]()
    from nutrition_agent.config import DAILY_REFERENCE_VALUES

    if not log:
        st.info(
            "📌 No data yet. Log your meals in the **📋 Food Log** tab to populate this dashboard.",
            icon="📊",
        )
    else:
        # ── Aggregate today's nutrients ────────────────────────────────────────
        actual: dict[str, float] = {k: 0.0 for k in DAILY_REFERENCE_VALUES}
        for entry in log:
            for k in actual:
                actual[k] += entry.get("totals", {}).get(k, 0)

        st.markdown("#### 🕸️ Nutrient Radar – Actual vs Target")
        fig_radar = agents["nutrient_radar_chart"](actual, DAILY_REFERENCE_VALUES)
        st.plotly_chart(fig_radar, width="stretch")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🍩 Macro Breakdown")
            fig_donut = agents["macro_donut_chart"](
                actual["protein_g"], actual["carbohydrates_g"], actual["fat_g"]
            )
            st.plotly_chart(fig_donut, width="stretch")

        with col_right:
            st.markdown("#### 🍕 Calories by Meal")
            fig_pie = agents["meal_calorie_breakdown"](log)
            if fig_pie.data:
                st.plotly_chart(fig_pie, width="stretch")

        st.markdown("---")
        st.markdown("#### 📊 Detailed Nutrient Progress")
        fig_bar = agents["nutrient_progress_chart"](actual, DAILY_REFERENCE_VALUES)
        st.plotly_chart(fig_bar, width="stretch")

        st.markdown("---")
        st.markdown("#### 🔴 Deficiency Gauges")

        gauge_nutrients = [
            ("Vitamin C",  "vitamin_c_mg", "mg"),
            ("Calcium",    "calcium_mg",   "mg"),
            ("Iron",       "iron_mg",      "mg"),
            ("Fiber",      "fiber_g",      "g"),
        ]
        g_cols = st.columns(4)
        for col, (name, key, unit) in zip(g_cols, gauge_nutrients):
            fig_g = agents["deficiency_gauge"](
                name, actual.get(key, 0), DAILY_REFERENCE_VALUES[key], unit
            )
            col.plotly_chart(fig_g, width="stretch")

        # ── Summary table ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📋 Detailed Daily Summary")
        drv_rows = []
        for key, target in DAILY_REFERENCE_VALUES.items():
            act_val = actual.get(key, 0)
            pct     = (act_val / target * 100) if target > 0 else 0
            status  = "✅" if 80 <= pct <= 120 else ("⚠️" if 50 <= pct < 80 else "🔴")
            drv_rows.append({
                "Nutrient": key.replace("_", " ").title(),
                "Actual":   f"{act_val:.1f}",
                "Target":   str(target),
                "% Target": f"{pct:.0f}%",
                "Status":   status,
            })
        df = pd.DataFrame(drv_rows)
        st.dataframe(df, width="stretch", hide_index=True)

        # ── Weekly simulation (demo) ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📈 Weekly Progress (Simulated Demo)")
        up   = st.session_state["user_profile"]
        from nutrition_agent.agents.diet_recommendation_agent import _calculate_bmr, _calculate_tdee, _adjust_for_goal
        if up.get("weight_kg"):
            bmr_d = _calculate_bmr(
                up.get("age", 30), up.get("gender", "male"),
                up.get("weight_kg", 70), up.get("height_cm", 170)
            )
            tdee_d = _calculate_tdee(bmr_d, up.get("activity_level", "moderate"))
            target_d = round(_adjust_for_goal(tdee_d, up.get("goal", "maintenance")))
        else:
            target_d = 2000

        import random
        random.seed(42)
        today_cals = actual.get("calories", 0) or round(target_d * random.uniform(0.8, 1.1))
        weekly_demo = [
            {"day": d, "calories": round(target_d * random.uniform(0.75, 1.15)),
             "target": target_d, "protein": round(50 * random.uniform(0.6, 1.2))}
            for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        ] + [{"day": "Today", "calories": today_cals, "target": target_d,
              "protein": round(actual.get("protein_g", 0))}]

        fig_weekly = agents["weekly_calorie_trend"](weekly_demo)
        st.plotly_chart(fig_weekly, width="stretch")
