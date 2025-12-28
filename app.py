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

    budget = st.sidebar.slider("Budget (HKD/hour)", 100, 500, 250, step=25)

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

        # Summary stats
        avg_prob = sum(r['probability'] for r in results) / len(results)
        within_budget = sum(1 for r in results if r['features']['price_gap'] <= 0)

        st.header(f"Top {len(results)} Recommended Tutors")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Success Rate", f"{avg_prob:.0%}")
        with col2:
            st.metric("Within Budget", f"{within_budget}/{len(results)}")
        with col3:
            st.metric("Best Match", f"{results[0]['probability']:.0%}")

        st.markdown("---")

        # Find best value (highest prob within budget)
        within_budget_results = [r for r in results if r['features']['price_gap'] <= 0]
        best_value_id = within_budget_results[0]['tutor_id'] if within_budget_results else None

        for i, result in enumerate(results):
            prob = result['probability']
            prob_color = "green" if prob > 0.6 else "orange" if prob > 0.4 else "red"

            # Badges
            badges = []
            if i == 0:
                badges.append("🏆 Best Match")
            if result['tutor_id'] == best_value_id:
                badges.append("💰 Best Value")
            if result['features']['price_gap'] <= 0:
                badges.append("✅ Within Budget")
            badge_str = " | ".join(badges) if badges else ""

            with st.expander(
                f"**#{i+1} {result['name']}** | "
                f"Success: :{prob_color}[{prob:.0%}] | "
                f"{badge_str or ('⚠️ ' + ', '.join(result['risk_tags'][:2]) if result['risk_tags'] else '')}",
                expanded=(i == 0)  # First one open by default
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    tutor_data = result['tutor_data']

                    # Rate comparison
                    rate = tutor_data.get('expected_rate', 0)
                    rate_diff = rate - budget
                    rate_status = "✅ Within budget" if rate_diff <= 0 else f"⚠️ HKD {rate_diff} over budget"

                    st.markdown(f"**Subject:** {tutor_data.get('primary_subject', 'N/A')} | **Experience:** {tutor_data.get('experience_years', 0)} years")
                    st.markdown(f"**Rate:** HKD {rate}/hr ({rate_status})")
                    st.markdown(f"**District:** {tutor_data.get('district', 'N/A')} | **Distance:** {result['features']['distance_km']:.1f} km")

                    if tutor_data.get('bio'):
                        st.markdown(f"**Bio:** {tutor_data['bio']}")

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
