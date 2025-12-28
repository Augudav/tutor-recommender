"""
Tutor Recommendation Model with LightGBM + SHAP explanations.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

MODEL_PATH = Path(__file__).parent / "model.joblib"


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def engineer_features(case: dict, tutor: pd.Series) -> dict:
    """
    Create features for a case-tutor pair.
    No naive point systems - meaningful derived features.
    """
    # Distance feature
    distance = haversine_distance(
        case.get('latitude', 22.3),
        case.get('longitude', 114.2),
        tutor['latitude'],
        tutor['longitude']
    )

    # Price gap (tutor rate - case budget)
    price_gap = tutor['expected_rate'] - case.get('budget', 200)
    price_gap_pct = price_gap / max(case.get('budget', 200), 1)

    # Subject match
    subject_match = int(
        case.get('subject', '') == tutor['primary_subject'] or
        case.get('subject', '') in str(tutor.get('secondary_subjects', ''))
    )

    # Level match
    level_match = int(case.get('level', '') in str(tutor.get('levels', '')))

    # Gender compatibility
    gender_pref = case.get('gender_preference')
    gender_ok = int(gender_pref is None or gender_pref == tutor['gender'])

    # Online compatibility
    online_compat = int(
        case.get('online_ok', True) or
        tutor.get('online_ok', True) or
        distance < 5
    )

    # Tutor quality metrics
    experience = tutor.get('experience_years', 0)
    total_cases = tutor.get('total_cases', 0)
    successful_cases = tutor.get('successful_cases', 0)
    historical_rate = successful_cases / max(total_cases, 1)

    # Is tutor a student (might affect availability/credibility)
    is_student = int(tutor.get('is_student', False))

    # SEN match
    sen_required = case.get('sen_requirement', False)
    has_sen_exp = int('SEN' in str(tutor.get('niche_skills', '')))
    sen_match = int(not sen_required or has_sen_exp)

    # Niche skill relevance
    has_niche = int(bool(tutor.get('niche_skills', '')))

    return {
        'distance_km': distance,
        'price_gap': price_gap,
        'price_gap_pct': price_gap_pct,
        'subject_match': subject_match,
        'level_match': level_match,
        'gender_ok': gender_ok,
        'online_compat': online_compat,
        'experience_years': experience,
        'historical_success_rate': historical_rate,
        'total_cases': total_cases,
        'is_student': is_student,
        'sen_match': sen_match,
        'has_niche_skill': has_niche,
    }


FEATURE_NAMES = [
    'distance_km', 'price_gap', 'price_gap_pct', 'subject_match',
    'level_match', 'gender_ok', 'online_compat', 'experience_years',
    'historical_success_rate', 'total_cases', 'is_student',
    'sen_match', 'has_niche_skill'
]


class TutorRecommender:
    """
    Hybrid recommendation system:
    1. Hard rule filter (gatekeeper)
    2. LightGBM probability ranking with SHAP explanations
    """

    def __init__(self):
        self.model: Optional[lgb.Booster] = None
        self.explainer: Optional[shap.TreeExplainer] = None

    def train(self, tutors_df: pd.DataFrame, cases_df: pd.DataFrame, results_df: pd.DataFrame):
        """Train the recommendation model on historical data."""
        # Build training data
        X_list = []
        y_list = []

        # Merge results with case and tutor data
        for _, result in results_df.iterrows():
            case = cases_df[cases_df['case_id'] == result['case_id']]
            tutor = tutors_df[tutors_df['tutor_id'] == result['tutor_id']]

            if len(case) == 0 or len(tutor) == 0:
                continue

            case_dict = case.iloc[0].to_dict()
            tutor_row = tutor.iloc[0]

            features = engineer_features(case_dict, tutor_row)
            X_list.append([features[f] for f in FEATURE_NAMES])
            y_list.append(result['success'])

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"Training on {len(X)} samples...")

        # Train LightGBM
        train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
        )

        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        print("Model trained successfully!")
        return self

    def save(self, path: Path = MODEL_PATH):
        """Save model to disk."""
        joblib.dump({'model': self.model}, path)
        print(f"Model saved to {path}")

    def load(self, path: Path = MODEL_PATH):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.explainer = shap.TreeExplainer(self.model)
        return self

    def hard_filter(self, case: dict, tutor: pd.Series) -> Tuple[bool, Optional[str]]:
        """
        Stage 1: Hard rule gatekeeper.
        Returns (passes, rejection_reason).
        """
        # Gender requirement
        if case.get('gender_preference') and case['gender_preference'] != tutor['gender']:
            return False, "Gender mismatch"

        # Distance limit for student tutors
        if tutor.get('is_student', False):
            distance = haversine_distance(
                case.get('latitude', 22.3), case.get('longitude', 114.2),
                tutor['latitude'], tutor['longitude']
            )
            if distance > 8 and not case.get('online_ok', True):
                return False, "Student tutor too far for in-person"

        # SEN requirement
        if case.get('sen_requirement', False):
            if 'SEN' not in str(tutor.get('niche_skills', '')):
                return False, "SEN experience required"

        return True, None

    def predict(self, case: dict, tutor: pd.Series) -> dict:
        """
        Get success probability and SHAP explanation for a case-tutor pair.
        """
        features = engineer_features(case, tutor)
        X = np.array([[features[f] for f in FEATURE_NAMES]])

        # Predict probability
        prob = self.model.predict(X)[0]

        # Get SHAP values
        shap_values = self.explainer.shap_values(X)[0]

        # Build explanation
        feature_impacts = []
        for i, fname in enumerate(FEATURE_NAMES):
            impact = shap_values[i]
            value = features[fname]
            feature_impacts.append({
                'feature': fname,
                'value': value,
                'impact': impact,
            })

        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)

        return {
            'probability': float(prob),
            'features': features,
            'shap_values': feature_impacts,
        }

    def rank_tutors(
        self,
        case: dict,
        tutors_df: pd.DataFrame,
        top_n: int = 10
    ) -> List[dict]:
        """
        Rank all tutors for a given case.
        Returns top N with probabilities and explanations.
        """
        results = []

        for _, tutor in tutors_df.iterrows():
            # Stage 1: Hard filter
            passes, rejection = self.hard_filter(case, tutor)
            if not passes:
                continue

            # Stage 2: Predict
            pred = self.predict(case, tutor)

            # Generate risk tags
            risk_tags = []
            if pred['features']['distance_km'] > 8:
                risk_tags.append("Distance high")
            if pred['features']['price_gap'] > 50:
                risk_tags.append("Over budget")
            if pred['features']['historical_success_rate'] < 0.3 and pred['features']['total_cases'] > 5:
                risk_tags.append("Low historical rate")
            if not pred['features']['subject_match']:
                risk_tags.append("Subject mismatch")

            # Generate "Why" explanation from top SHAP factors
            why_factors = []
            for item in pred['shap_values'][:3]:
                if item['impact'] > 0.05:
                    why_factors.append(f"{item['feature'].replace('_', ' ').title()}: +")
                elif item['impact'] < -0.05:
                    why_factors.append(f"{item['feature'].replace('_', ' ').title()}: -")

            results.append({
                'tutor_id': tutor['tutor_id'],
                'name': tutor.get('name', tutor['tutor_id']),
                'probability': pred['probability'],
                'risk_tags': risk_tags,
                'why': why_factors[:3],
                'shap_details': pred['shap_values'],
                'features': pred['features'],
                'tutor_data': tutor.to_dict(),
            })

        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)

        return results[:top_n]

    def simulate_budget(
        self,
        case: dict,
        tutor: pd.Series,
        budget_adjustments: List[float] = [-0.2, 0, 0.2, 0.4]
    ) -> List[dict]:
        """
        Dynamic pricing simulation.
        Show how success probability changes with budget.
        """
        original_budget = case.get('budget', 200)
        results = []

        for adj in budget_adjustments:
            adjusted_case = case.copy()
            adjusted_case['budget'] = int(original_budget * (1 + adj))

            pred = self.predict(adjusted_case, tutor)
            results.append({
                'budget': adjusted_case['budget'],
                'adjustment': f"{adj:+.0%}",
                'probability': pred['probability'],
            })

        return results


def train_and_save():
    """Train model on sample data and save."""
    from sample_data import generate_all_data

    tutors, cases, results = generate_all_data()

    model = TutorRecommender()
    model.train(tutors, cases, results)
    model.save()

    return model, tutors, cases


if __name__ == "__main__":
    train_and_save()
