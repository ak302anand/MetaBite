"""MetaBite: cycle-aware nutrition analysis and food exploration dashboard.

Run with: streamlit run analyzer.py
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


FOUNDATION_DATA_DIR = Path(__file__).resolve().parent / "FoodData_Central_foundation_food_csv_2025-12-18"
NUTRIENT_IDS = {1003: "Protein (g)", 1004: "Total Fat (g)", 1005: "Carbohydrates (g)", 1079: "Fiber (g)", 1089: "Iron (mg)", 1090: "Magnesium (mg)"}
ENERGY_NUTRIENT_PRIORITY = {1008: 0, 2047: 1, 2048: 2}
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

ACTIVITY_MULTIPLIERS = {
    "Sedentary": 1.2,
    "Lightly Active": 1.375,
    "Moderately Active": 1.55,
    "Very Active": 1.725,
}
GOAL_ADJUSTMENTS = {"Weight Loss": -500, "Maintenance": 0, "Weight Gain": 400}
DEFAULT_MACROS = {"Carbs": 0.40, "Protein": 0.30, "Fat": 0.30}
CYCLE_CONFIG = {
    "Follicular Phase (Days 1-13)": {
        "calorie_adjustment": 0,
        "macros": {"Carbs": 0.45, "Protein": 0.30, "Fat": 0.25},
        "note": "Higher insulin sensitivity: a carbohydrate-forward, balanced plan.",
    },
    "Ovulatory Phase (Days 14-16)": {
        "calorie_adjustment": 100,
        "macros": {"Carbs": 0.40, "Protein": 0.30, "Fat": 0.30},
        "note": "Includes 100 kcal to support peak energy demands.",
    },
    "Luteal Phase (Days 17-28)": {
        "calorie_adjustment": 200,
        "macros": {"Carbs": 0.35, "Protein": 0.35, "Fat": 0.30},
        "note": "Includes 200 kcal and higher protein/fat to support appetite and blood-sugar stability.",
    },
    "Menstrual Phase (Days 1-5)": {
        "calorie_adjustment": 0,
        "macros": {"Carbs": 0.40, "Protein": 0.30, "Fat": 0.30},
        "note": "Standard energy target; prioritize anti-inflammatory, iron-rich, and magnesium-rich foods.",
    },
}


def mock_foods() -> pd.DataFrame:
    """Return a small usable dataset whenever a source is unavailable."""
    return pd.DataFrame(
        [
            ("Chicken breast, roasted", 165, 31.0, 0.0, 3.6),
            ("Salmon, cooked", 208, 20.0, 0.0, 13.0),
            ("Greek yogurt, plain", 97, 9.0, 3.9, 5.0),
            ("Lentils, cooked", 116, 9.0, 20.0, 0.4),
            ("Oats, dry", 379, 13.2, 67.7, 6.5),
            ("Spinach, raw", 23, 2.9, 3.6, 0.4),
            ("Banana, raw", 89, 1.1, 22.8, 0.3),
            ("Almonds", 579, 21.2, 21.6, 49.9),
        ],
        columns=["Food Description", "Calories", "Protein (g)", "Carbohydrates (g)", "Total Fat (g)"],
    )


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find the first column matching a standard nutrition field."""
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return next((column for key, column in normalized.items() if any(c in key for c in candidates)), None)


def standardize_food_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common USDA/local headers into a stable per-100g table schema."""
    aliases = {
        "Food Description": ["food description", "description", "food_name", "food name"],
        "Calories": ["calories", "energy", "energy (kcal)", "kcal"],
        "Protein (g)": ["protein (g)", "protein", "protein_g"],
        "Carbohydrates (g)": ["carbohydrates (g)", "carbohydrate", "carbohydrates", "carbs", "carbohydrate_g"],
        "Total Fat (g)": ["total fat (g)", "total fat", "fat", "fat_g"],
        "Fiber (g)": ["fiber (g)", "fiber", "dietary fiber"],
        "Iron (mg)": ["iron (mg)", "iron", "iron_mg"],
        "Magnesium (mg)": ["magnesium (mg)", "magnesium", "magnesium_mg"],
    }
    result = pd.DataFrame()
    for target, candidates in aliases.items():
        source = _find_column(list(frame.columns), candidates)
        result[target] = frame[source] if source else ("Unknown" if target == "Food Description" else 0)
    for column in result.columns[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).round(1)
    return result.dropna(subset=["Food Description"]).drop_duplicates().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_local_foods(file_path: str) -> pd.DataFrame:
    """Load the local USDA CSV with clear errors for absent or invalid files."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return standardize_food_data(pd.read_csv(path))



@st.cache_data(show_spinner=False)
def load_foundation_foods(data_directory: str) -> pd.DataFrame:
    """Build one food-per-row dashboard dataset from USDA's normalized export."""
    directory = Path(data_directory)
    required = ("foundation_food.csv", "food.csv", "food_nutrient.csv")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Foundation dataset is incomplete at {directory}. Missing: {', '.join(missing)}"
        )

    foundation = pd.read_csv(directory / "foundation_food.csv", usecols=["fdc_id"])
    foods = pd.read_csv(directory / "food.csv", usecols=["fdc_id", "description"])
    nutrients = pd.read_csv(
        directory / "food_nutrient.csv",
        usecols=["fdc_id", "nutrient_id", "amount"],
        dtype={"fdc_id": "int64", "nutrient_id": "int64"},
    )
    foundation_foods = foundation.merge(foods, on="fdc_id", how="inner")
    nutrient_ids = [*NUTRIENT_IDS, *ENERGY_NUTRIENT_PRIORITY]
    nutrients = nutrients[nutrients["fdc_id"].isin(foundation_foods["fdc_id"])]
    nutrients = nutrients[nutrients["nutrient_id"].isin(nutrient_ids)].copy()
    nutrients["amount"] = pd.to_numeric(nutrients["amount"], errors="coerce")

    macro_nutrients = nutrients[nutrients["nutrient_id"].isin(NUTRIENT_IDS)].copy()
    macro_nutrients["metric"] = macro_nutrients["nutrient_id"].map(NUTRIENT_IDS)
    macro_wide = macro_nutrients.pivot_table(
        index="fdc_id", columns="metric", values="amount", aggfunc="median"
    ).reset_index()

    energy = nutrients[nutrients["nutrient_id"].isin(ENERGY_NUTRIENT_PRIORITY)].copy()
    energy["priority"] = energy["nutrient_id"].map(ENERGY_NUTRIENT_PRIORITY)
    energy = energy.sort_values(["fdc_id", "priority"]).drop_duplicates("fdc_id")
    energy = energy[["fdc_id", "amount"]].rename(columns={"amount": "Calories"})

    combined = foundation_foods.merge(macro_wide, on="fdc_id", how="left")
    combined = combined.merge(energy, on="fdc_id", how="left")
    return standardize_food_data(combined)

@st.cache_data(ttl=3600, show_spinner=False)
def load_usda_foods(api_key: str) -> pd.DataFrame:
    """Fetch a representative USDA FoodData Central search result."""
    if not api_key.strip():
        raise ValueError("Enter a USDA API key to use the USDA API source.")
    response = requests.get(
        USDA_SEARCH_URL,
        params={"api_key": api_key, "query": "whole foods", "pageSize": 100},
        timeout=15,
    )
    response.raise_for_status()
    foods = response.json().get("foods", [])
    if not foods:
        raise ValueError("The USDA API returned no foods.")

    records: list[dict[str, Any]] = []
    for food in foods:
        nutrients = {str(item.get("nutrientName", "")).lower(): item.get("value", 0) for item in food.get("foodNutrients", [])}
        records.append({
            "Food Description": food.get("description", "Unknown"),
            "Calories": nutrients.get("energy", nutrients.get("energy (atwater general factors)", 0)),
            "Protein (g)": nutrients.get("protein", 0),
            "Carbohydrates (g)": nutrients.get("carbohydrate, by difference", 0),
            "Total Fat (g)": nutrients.get("total lipid (fat)", 0),
        })
    return standardize_food_data(pd.DataFrame(records))


def calculate_plan(age: int, gender: str, weight: float, height: float, activity: str, goal: str, phase: str) -> dict[str, Any]:
    """Calculate BMR, adjusted calorie target, and gram-level macro targets."""
    sex_constant = 5 if gender == "Male" else -161
    bmr = (10 * weight) + (6.25 * height) - (5 * age) + sex_constant
    baseline_tdee = bmr * ACTIVITY_MULTIPLIERS[activity]
    cycle = CYCLE_CONFIG.get(phase, {"calorie_adjustment": 0, "macros": DEFAULT_MACROS, "note": "Standard macro distribution."})
    adjusted_tdee = baseline_tdee + cycle["calorie_adjustment"]
    target_calories = max(0, adjusted_tdee + GOAL_ADJUSTMENTS[goal])
    macros = cycle["macros"]
    grams = {
        "Carbs": target_calories * macros["Carbs"] / 4,
        "Protein": target_calories * macros["Protein"] / 4,
        "Fat": target_calories * macros["Fat"] / 9,
    }
    return {"bmr": bmr, "baseline_tdee": baseline_tdee, "tdee": adjusted_tdee, "target": target_calories, "macros": macros, "grams": grams, "cycle": cycle}


def meal_schedule(wake: time, sleep: time, calories: float) -> pd.DataFrame:
    """Build meal windows across the waking interval, handling overnight schedules."""
    anchor = datetime.combine(datetime.today().date(), wake)
    bedtime = datetime.combine(datetime.today().date(), sleep)
    if bedtime <= anchor:
        bedtime += timedelta(days=1)
    waking_hours = bedtime - anchor
    meal_times = [anchor + timedelta(hours=1), anchor + waking_hours * 0.48, bedtime - timedelta(hours=3)]
    labels = ["Breakfast", "Lunch", "Dinner"]
    shares = [0.40, 0.35, 0.25]
    guidance = ["High protein to blunt the morning cortisol spike", "Balanced macros, dense fiber", "Light, low-glycemic focus for recovery"]
    return pd.DataFrame({
        "Meal": labels,
        "Calories": [calories * share for share in shares],
        "Share": shares,
        "Window": [item.strftime("%I:%M %p").lstrip("0") for item in meal_times],
        "Focus": guidance,
    })


def main() -> None:
    st.set_page_config(page_title="MetaBite", page_icon="⏱️", layout="wide")
    st.title("MetaBite")
    st.caption("Circadian nutrition targets & cycle-aware adjustments backed by USDA")

    with st.sidebar:
        st.header("Workspace & data")
        data_source = st.radio("Data Source", ["USDA Foundation Foods", "USDA API"])
        api_key = st.text_input("USDA API Key", type="password") if data_source == "USDA API" else ""
        st.divider()
        st.header("User profile")
        age = st.number_input("Age", min_value=13, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"])
        weight = st.number_input("Weight (kg)", min_value=25.0, max_value=400.0, value=65.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
        activity = st.selectbox("Activity Level", list(ACTIVITY_MULTIPLIERS))
        goal = st.selectbox("Primary Goal", list(GOAL_ADJUSTMENTS))
        phase = "Not Applicable / Regular Calculation"
        if gender == "Female":
            phase = st.selectbox("Cycle Phase", [*CYCLE_CONFIG, "Not Applicable / Regular Calculation"])
        st.divider()
        st.header("Circadian schedule")
        wake = st.time_input("Wake-up Time", value=time(7, 0))
        sleep = st.time_input("Sleep Time", value=time(23, 0))

    plan = calculate_plan(int(age), gender, float(weight), float(height), activity, goal, phase)
    schedule = meal_schedule(wake, sleep, plan["target"])

    st.subheader("Daily targets")
    kpis = st.columns(4)
    kpis[0].metric("Target Daily Calories", f"{plan['target']:,.0f} kcal")
    kpis[1].metric("BMR", f"{plan['bmr']:,.0f} kcal")
    kpis[2].metric("Adjusted TDEE", f"{plan['tdee']:,.0f} kcal")
    kpis[3].metric("Protein", f"{plan['grams']['Protein']:,.0f} g")
    macro_kpis = st.columns(3)
    macro_kpis[0].metric("Carbohydrates", f"{plan['grams']['Carbs']:,.0f} g")
    macro_kpis[1].metric("Fat", f"{plan['grams']['Fat']:,.0f} g")
    macro_kpis[2].metric("Goal adjustment", f"{GOAL_ADJUSTMENTS[goal]:+,.0f} kcal")

    cycle_delta = plan["cycle"]["calorie_adjustment"]
    st.info(f"**Cycle Sync: {phase if gender == 'Female' else 'Regular calculation'}** — {plan['cycle']['note']} "
            f"Cycle energy adjustment: **{cycle_delta:+.0f} kcal**. Macro split: "
            f"**{plan['macros']['Carbs']:.0%} carbs / {plan['macros']['Protein']:.0%} protein / {plan['macros']['Fat']:.0%} fat**.")

    left, right = st.columns(2)
    macro_frame = pd.DataFrame({
        "Macro": ["Carbs", "Protein", "Fat"],
        "Grams": [plan["grams"]["Carbs"], plan["grams"]["Protein"], plan["grams"]["Fat"]],
        "Calories": [plan["target"] * plan["macros"]["Carbs"], plan["target"] * plan["macros"]["Protein"], plan["target"] * plan["macros"]["Fat"]],
    })
    with left:
        chart = px.pie(macro_frame, values="Calories", names="Macro", hole=0.58, title="Macro targets")
        chart.update_traces(texttemplate="%{label}<br>%{value:.0f} kcal", hovertemplate="%{label}: %{value:.0f} kcal<br>%{customdata[0]:.0f} g<extra></extra>", customdata=macro_frame[["Grams"]])
        st.plotly_chart(chart, use_container_width=True)
    with right:
        chart = px.bar(schedule, x="Meal", y="Calories", color="Meal", text_auto=".0f", title="Circadian calorie split", labels={"Calories": "Calories (kcal)"})
        chart.update_layout(showlegend=False)
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("Meal timing")
    st.dataframe(schedule.assign(Share=schedule["Share"].map("{:.0%}".format), Calories=schedule["Calories"].round(0)), use_container_width=True, hide_index=True)

    st.subheader("Food explorer")
    try:
        foods = load_foundation_foods(str(FOUNDATION_DATA_DIR)) if data_source == "USDA Foundation Foods" else load_usda_foods(api_key)
        source_label = data_source
    except (FileNotFoundError, ValueError, pd.errors.ParserError, requests.RequestException) as error:
        foods = mock_foods()
        source_label = "built-in mock data"
        st.warning(f"Could not load {data_source} ({error}). Showing built-in mock data instead.")
    st.caption(f"Showing {len(foods):,} foods from {source_label}. Nutrition values are per 100 g where supplied by the source.")
    query = st.text_input("Search candidate food items", placeholder="e.g., salmon, spinach, oats")
    filtered = foods[foods["Food Description"].str.contains(query, case=False, na=False)] if query else foods
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=360)


if __name__ == "__main__":
    main()
