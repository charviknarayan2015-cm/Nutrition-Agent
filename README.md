# 🥗 NutriAgent – AI-Powered Nutrition Assistant

A multi-agent nutrition guidance system powered by **IBM watsonx.ai (Llama-3 70B Instruct)**.

## 🏗️ Architecture

```
nutrition_agent/
├── app.py                          ← Streamlit UI (6 tabs, full dashboard)
├── config.py                       ← watsonx.ai credentials, DRVs, condition profiles
├── llm_client.py                   ← IBM watsonx.ai ModelInference singleton
├── rag_engine.py                   ← BM25 RAG retrieval over nutrition knowledge base
├── visualization.py                ← Plotly charts (donut, radar, gauge, bar, trend)
├── agents/
│   ├── nutrition_knowledge_agent.py ← Agent 1: RAG food/nutrient facts
│   ├── diet_recommendation_agent.py ← Agent 2: Personalized 7-day meal plans
│   ├── health_advisory_agent.py     ← Agent 3: Disease-specific & preventive guidance
│   └── food_log_agent.py            ← Agent 4: Meal logging + instant analysis
├── data/
│   └── food_knowledge_base.py       ← 20+ foods, 25+ knowledge docs (USDA/WHO)
├── tests/
│   └── test_agents.py               ← Smoke tests (no LLM required)
└── requirements.txt
```

## 🤖 Multi-Agent System

| Agent | Responsibility |
|---|---|
| **Nutrition Knowledge Agent** | RAG-powered food lookup, nutrient facts, food comparisons |
| **Diet Recommendation Agent** | Personalized 7-day meal plans, BMR/TDEE calculation, meal swaps |
| **Health Advisory Agent** | Disease-specific guidance (diabetes, hypertension, heart), preventive plans, risk assessment |
| **Food Log & Feedback Agent** | Text meal logging, instant nutritional analysis, daily summaries |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r nutrition_agent/requirements.txt
```

### 2. Configure Credentials

The credentials are pre-configured in `nutrition_agent/config.py`:
```python
WATSONX_API_KEY    = "oEemYECa1DXFEQtUq0EbMoPUHJ1OW6tqemLsddrI3RsQ"
WATSONX_PROJECT_ID = "40e09d61-b236-4796-8222-2f212105aeb0"
WATSONX_URL        = "https://us-south.ml.cloud.ibm.com"
```

Or set them as environment variables in `.env`:
```
WATSONX_API_KEY=oEemYECa1DXFEQtUq0EbMoPUHJ1OW6tqemLsddrI3RsQ
WATSONX_PROJECT_ID=40e09d61-b236-4796-8222-2f212105aeb0
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

### 3. Run Tests (no LLM required)

```bash
python nutrition_agent/tests/test_agents.py
```

### 4. Launch the App

```bash
streamlit run nutrition_agent/app.py
```

---

## 📱 Application Features

### 🏠 Dashboard
- Real-time daily calorie and macro summary (5 KPI cards)
- Macro donut chart + nutrient progress bars
- Today's meal log with AI feedback for each entry
- Multi-agent status overview

### 🔍 Nutrition Knowledge (Agent 1)
- **Ask a Question**: Free-text nutrition Q&A with RAG retrieval
- **Look Up a Food**: Full nutrient profile with 6 macro + 3 micro metrics
- **Compare Two Foods**: Side-by-side bar chart + AI narrative comparison

### 🍽️ Meal Plan (Agent 2)
- **7-Day Personalized Meal Plan**: Adapts to health conditions, allergies, cuisine preferences
- **Calorie Preview**: Live BMR/TDEE calculation from sidebar profile
- **Meal Swap Suggestions**: Replace any meal with 3 healthier alternatives

### 📋 Food Log (Agent 4)
- **Log Meals**: Free-text entry (e.g. "2 eggs, oatmeal, green tea")
- **Instant Analysis**: Calories, macros, feedback, and improvement tips
- **Daily Analysis**: Full-day nutritional grade (A/B/C/D), gap analysis
- **Calorie Breakdown Pie**: Distribution across breakfast/lunch/dinner/snacks

### 🏥 Health Advisory (Agent 3)
- **Condition-Specific Guidance**: Structured clinical guidelines for 12 conditions
- **Preventive Health Plan**: Multi-risk-factor prevention plan
- **Diet Risk Assessment**: Flags high sodium, excess sugar, low fiber, low protein

### 📊 Analytics
- **Nutrient Radar Chart**: 7-nutrient spider web vs daily targets
- **Deficiency Gauges**: Visual red/amber/green gauges for Vit C, Calcium, Iron, Fiber
- **Detailed Summary Table**: All nutrients with % of daily target and status emoji
- **Weekly Calorie Trend**: Bar + line combo chart with protein tracking

---

## 🧠 Technical Details

### RAG Engine
- **Index**: BM25 inverted index over 25 curated documents (USDA/WHO data)
- **Sources**: Macronutrients, vitamins/minerals, 10+ food profiles, 5 disease guidelines, 5 meal plan templates
- **Lookup**: Exact and partial-match food database (20 common foods)
- **No external vector DB required** – runs fully in-process

### LLM Integration
- **Model**: `meta-llama/llama-3-3-70b-instruct` via IBM watsonx.ai
- **Temperature**: 0.4 (balanced factual + creative)
- **Max tokens**: 1024 per response
- **Singleton pattern**: One model instance shared across all agents

### Personalization
- **BMR/TDEE**: Mifflin-St Jeor equation
- **Goal-adjusted calories**: -500 kcal (loss) / +300 kcal (gain)
- **Condition-aware macros**: Diabetic, weight-loss, muscle-gain splits
- **Health condition profiles**: Per-nutrient limits for 7 chronic conditions

---

## 📋 Supported Health Conditions

| Condition | Key Restrictions |
|---|---|
| Diabetes | Carbs ≤130g, Sugar ≤25g, Fiber ≥35g |
| Hypertension | Sodium ≤1500mg |
| Heart Disease | Fat ≤55g, Sodium ≤1500mg |
| Kidney Disease | Protein ≤40g, Potassium ≤2000mg |
| Obesity | Calories ≤1600, Fat ≤55g |
| Celiac Disease | Gluten-free |
| Lactose Intolerance | Dairy-free |
