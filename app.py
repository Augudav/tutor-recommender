"""
Tutor Recommendation System - Demo UI
Hybrid ranking with LightGBM + SHAP explanations.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Must be first Streamlit command
st.set_page_config(
    page_title="Tutor Recommender",
    page_icon="🎓",
    layout="wide"
)

from model import TutorRecommender, MODEL_PATH, FEATURE_NAMES
from sample_data import generate_all_data, HK_DISTRICTS, SUBJECTS, LEVELS

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tutors' not in st.session_state:
    st.session_state.tutors = None
if 'cases' not in st.session_state:
    st.session_state.cases = None


@st.cache_resource
def load_or_train_model():
    """Load existing model or train new one."""
    model = TutorRecommender()

    tutors_path = Path(__file__).parent / "tutors.csv"
    cases_path = Path(__file__).parent / "cases.csv"

    # Check if ALL required files exist
    if MODEL_PATH.exists() and tutors_path.exists() and cases_path.exists():
        model.load()
        tutors = pd.read_csv(tutors_path)
        cases = pd.read_csv(cases_path)
    else:
        # Generate fresh data and train
        tutors, cases, results = generate_all_data()
        model.train(tutors, cases, results)
        model.save()
        tutors.to_csv(tutors_path, index=False)
        cases.to_csv(cases_path, index=False)
        results.to_csv(Path(__file__).parent / "results.csv", index=False)

    return model, tutors, cases


def render_shap_explanation(shap_details: list):
    """Render SHAP values as a visual explanation."""
    st.markdown("**Feature Impacts (SHAP)**")

    for item in shap_details[:6]:
        impact = item['impact']
        feature = item['feature'].replace('_', ' ').title()
        value = item['value']

        # Color based on positive/negative impact
        if impact > 0:
            color = "green"
            symbol = "+"
        else:
            color = "red"
            symbol = ""

        bar_width = min(abs(impact) * 200, 100)

        st.markdown(
            f"<div style='margin:2px 0;'>"
            f"<span style='width:150px;display:inline-block;'>{feature}</span>"
            f"<span style='color:{color};width:60px;display:inline-block;'>{symbol}{impact:.3f}</span>"
            f"<span style='background:{color};width:{bar_width}px;height:12px;display:inline-block;border-radius:3px;'></span>"
            f"<span style='color:gray;margin-left:10px;'>({value:.2f})</span>"
            f"</div>",
            unsafe_allow_html=True
        )


def main():
    st.title("🎓 Tutor Recommendation System")
    st.markdown("*Hybrid ML ranking with explainable predictions*")

    # Load model
    with st.spinner("Loading model..."):
        model, tutors, cases = load_or_train_model()

    st.success(f"Model loaded. {len(tutors)} tutors available.")

    # Sidebar: Case input
    st.sidebar.header("📋 New Case Details")

    district = st.sidebar.selectbox("District", list(HK_DISTRICTS.keys()))
    subject = st.sidebar.selectbox("Subject", SUBJECTS)
    level = st.sidebar.selectbox("Level", LEVELS)

    budget = st.sidebar.slider("Budget (HKD/hour)", 100, 500, 200, step=25)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        gender_pref = st.selectbox("Gender Pref", [None, "M", "F"])
    with col2:
        online_ok = st.checkbox("Online OK", value=True)

    sen_required = st.sidebar.checkbox("SEN Experience Required")

    description = st.sidebar.text_area(
        "Case Description",
        f"Looking for {subject} tutor for {level} student in {district}."
    )

    # Build case dict
    lat, lon = HK_DISTRICTS[district]
    case = {
        'district': district,
        'latitude': lat,
        'longitude': lon,
        'subject': subject,
        'level': level,
        'budget': budget,
        'gender_preference': gender_pref,
        'online_ok': online_ok,
        'sen_requirement': sen_required,
        'description': description,
    }

    # Main content
    if st.sidebar.button("🔍 Find Best Tutors", type="primary"):
        with st.spinner("Ranking tutors..."):
            results = model.rank_tutors(case, tutors, top_n=10)

        if not results:
            st.warning("No tutors passed the filters. Try relaxing requirements.")
            return

        st.header(f"Top {len(results)} Recommended Tutors")

        for i, result in enumerate(results):
            prob = result['probability']
            prob_color = "green" if prob > 0.6 else "orange" if prob > 0.4 else "red"

            with st.expander(
                f"**#{i+1} {result['name']}** | "
                f"Success: :{prob_color}[{prob:.0%}] | "
                f"{'⚠️ ' + ', '.join(result['risk_tags']) if result['risk_tags'] else '✅ No risks'}"
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Tutor ID:** {result['tutor_id']}")

                    tutor_data = result['tutor_data']
                    st.markdown(f"**Subject:** {tutor_data.get('primary_subject', 'N/A')}")
                    st.markdown(f"**Experience:** {tutor_data.get('experience_years', 0)} years")
                    st.markdown(f"**Expected Rate:** HKD {tutor_data.get('expected_rate', 'N/A')}/hr")
                    st.markdown(f"**District:** {tutor_data.get('district', 'N/A')}")

                    if tutor_data.get('bio'):
                        st.markdown(f"**Bio:** {tutor_data['bio'][:200]}...")

                    # Why this tutor
                    st.markdown("---")
                    st.markdown("**Why this tutor?**")
                    for reason in result['why']:
                        st.markdown(f"- {reason}")

                with col2:
                    st.metric("Success Probability", f"{prob:.0%}")

                    # Risk tags
                    if result['risk_tags']:
                        st.markdown("**Risk Factors:**")
                        for tag in result['risk_tags']:
                            st.markdown(f"- ⚠️ {tag}")

                # SHAP details
                st.markdown("---")
                render_shap_explanation(result['shap_details'])

                # Budget simulation
                st.markdown("---")
                st.markdown("**💰 Budget Impact Simulation**")

                tutor_row = tutors[tutors['tutor_id'] == result['tutor_id']].iloc[0]
                budget_sim = model.simulate_budget(case, tutor_row)

                sim_df = pd.DataFrame(budget_sim)
                st.dataframe(
                    sim_df.style.format({
                        'probability': '{:.1%}',
                        'budget': 'HKD {}',
                    }),
                    hide_index=True,
                    use_container_width=True
                )

    # Show model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Model Info")
    st.sidebar.markdown(f"- **Algorithm:** LightGBM")
    st.sidebar.markdown(f"- **Features:** {len(FEATURE_NAMES)}")
    st.sidebar.markdown(f"- **Explainability:** SHAP")
    st.sidebar.markdown(f"- **Tutors:** {len(tutors)}")


if __name__ == "__main__":
    main()
