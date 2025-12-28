"""
Generate sample tutoring data for demo.
Mimics the Hong Kong tutoring agency dataset structure.
"""
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Hong Kong districts with approximate coordinates
HK_DISTRICTS = {
    "Central": (22.282, 114.158),
    "Wan Chai": (22.279, 114.171),
    "Causeway Bay": (22.280, 114.185),
    "North Point": (22.291, 114.200),
    "Quarry Bay": (22.288, 114.214),
    "Tai Koo": (22.286, 114.220),
    "Shau Kei Wan": (22.279, 114.228),
    "Tsim Sha Tsui": (22.298, 114.172),
    "Mong Kok": (22.319, 114.169),
    "Yau Ma Tei": (22.313, 114.170),
    "Sham Shui Po": (22.330, 114.162),
    "Kowloon Tong": (22.337, 114.176),
    "Sha Tin": (22.381, 114.188),
    "Tai Po": (22.451, 114.164),
    "Tuen Mun": (22.390, 113.973),
    "Yuen Long": (22.445, 114.035),
    "Tsuen Wan": (22.371, 114.114),
    "Kwun Tong": (22.311, 114.226),
}

SUBJECTS = [
    "Mathematics", "English", "Chinese", "Physics", "Chemistry",
    "Biology", "Economics", "History", "Geography", "ICT",
    "IELTS", "SAT", "IB Mathematics", "IB Economics", "A-Level Maths",
    "Piano", "Violin", "Cantonese", "Mandarin", "French"
]

LEVELS = ["Primary", "Secondary", "DSE", "IB", "A-Level", "University", "Adult"]

NICHE_SKILLS = [
    "Medical Interview Prep", "Aviation English", "Oxbridge Application",
    "SEN Experience", "Gifted Education", "Exam Technique", "Essay Writing",
    "Public Speaking", "Debate Coaching", "Competition Maths"
]

TUTOR_BIOS = [
    "Experienced tutor with 5 years teaching {subject}. Graduated from HKU with First Class Honours.",
    "Native English speaker, TEFL certified. Specializing in {subject} for international curriculum.",
    "Former {subject} teacher at prestigious local school. Expert in DSE preparation.",
    "PhD student at CUHK. Patient and methodical approach to teaching {subject}.",
    "Professional tutor focusing on {subject}. Many students achieved Level 5** in DSE.",
    "Cambridge graduate with experience in {skill}. Available for online and in-person sessions.",
    "Bilingual tutor (English/Cantonese). Strong track record with {subject} students.",
    "IB examiner for {subject}. Deep understanding of assessment criteria.",
]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def generate_tutors(n=500):
    """Generate sample tutor data."""
    tutors = []
    for i in range(n):
        district = random.choice(list(HK_DISTRICTS.keys()))
        lat, lon = HK_DISTRICTS[district]
        # Add some randomness to location
        lat += random.uniform(-0.01, 0.01)
        lon += random.uniform(-0.01, 0.01)

        primary_subject = random.choice(SUBJECTS)
        secondary_subjects = random.sample(SUBJECTS, k=random.randint(1, 3))

        experience_years = random.randint(1, 15)

        # Expected rate based on experience and subject
        # Lower base to ensure many tutors fit within typical budgets (150-300)
        base_rate = 120 + experience_years * 10
        if primary_subject in ["IELTS", "SAT", "IB Mathematics", "IB Economics"]:
            base_rate *= 1.2
        expected_rate = int(base_rate + random.randint(-40, 40))

        # Generate bio
        bio_template = random.choice(TUTOR_BIOS)
        niche = random.choice(NICHE_SKILLS) if random.random() > 0.7 else ""
        bio = bio_template.format(subject=primary_subject, skill=niche or "exam preparation")
        if niche:
            bio += f" Specialized in {niche}."

        tutors.append({
            "tutor_id": f"T{i+1:04d}",
            "name": f"Tutor_{i+1}",
            "district": district,
            "latitude": lat,
            "longitude": lon,
            "primary_subject": primary_subject,
            "secondary_subjects": ",".join(secondary_subjects),
            "levels": ",".join(random.sample(LEVELS, k=random.randint(1, 4))),
            "experience_years": experience_years,
            "expected_rate": expected_rate,
            "gender": random.choice(["M", "F"]),
            "is_student": random.random() < 0.3,
            "online_ok": random.random() > 0.2,
            "bio": bio,
            "niche_skills": niche,
            "total_cases": random.randint(0, 50),
            "successful_cases": 0,  # Will be computed from results
        })

    return pd.DataFrame(tutors)


def generate_cases(n=300):
    """Generate sample case (student request) data."""
    cases = []
    for i in range(n):
        district = random.choice(list(HK_DISTRICTS.keys()))
        lat, lon = HK_DISTRICTS[district]
        lat += random.uniform(-0.01, 0.01)
        lon += random.uniform(-0.01, 0.01)

        subject = random.choice(SUBJECTS)
        level = random.choice(LEVELS)

        # Budget based on level and subject
        base_budget = 150
        if level in ["IB", "A-Level", "University"]:
            base_budget = 250
        if subject in ["IELTS", "SAT"]:
            base_budget = 300
        budget = int(base_budget + random.randint(-50, 100))

        # Case description
        descriptions = [
            f"Looking for {subject} tutor for {level} student. Prefer experienced teacher.",
            f"Need help with {subject} ({level}). Student struggling with recent topics.",
            f"Seeking {subject} tutor, {level} level. Exam preparation focus.",
            f"Want {subject} lessons for {level}. Flexible schedule preferred.",
        ]

        cases.append({
            "case_id": f"C{i+1:04d}",
            "district": district,
            "latitude": lat,
            "longitude": lon,
            "subject": subject,
            "level": level,
            "budget": budget,
            "sessions_per_week": random.choice([1, 2, 3]),
            "gender_preference": random.choice([None, "M", "F"]),
            "online_ok": random.random() > 0.3,
            "description": random.choice(descriptions),
            "sen_requirement": random.random() < 0.1,
            "created_at": datetime.now() - timedelta(days=random.randint(0, 365)),
        })

    return pd.DataFrame(cases)


def generate_results(tutors_df, cases_df, n_pairs=2000):
    """Generate case-tutor match results with realistic success patterns."""
    results = []

    tutor_success_counts = {tid: 0 for tid in tutors_df['tutor_id']}

    for _ in range(n_pairs):
        case = cases_df.sample(1).iloc[0]
        tutor = tutors_df.sample(1).iloc[0]

        # Calculate features that affect success
        distance = haversine_distance(
            case['latitude'], case['longitude'],
            tutor['latitude'], tutor['longitude']
        )

        price_gap = tutor['expected_rate'] - case['budget']

        # Subject match
        subject_match = (
            case['subject'] == tutor['primary_subject'] or
            case['subject'] in tutor['secondary_subjects']
        )

        # Level match
        level_match = case['level'] in tutor['levels']

        # Gender match
        gender_ok = (
            case['gender_preference'] is None or
            case['gender_preference'] == tutor['gender']
        )

        # Online compatibility
        online_ok = case['online_ok'] or tutor['online_ok'] or distance < 5

        # Calculate success probability
        success_prob = 0.5

        # Distance effect (closer is better)
        if distance < 3:
            success_prob += 0.15
        elif distance < 5:
            success_prob += 0.05
        elif distance > 10:
            success_prob -= 0.2

        # Price effect
        if price_gap <= 0:
            success_prob += 0.1
        elif price_gap > 50:
            success_prob -= 0.15
        elif price_gap > 100:
            success_prob -= 0.3

        # Subject/level match
        if subject_match:
            success_prob += 0.2
        else:
            success_prob -= 0.3

        if level_match:
            success_prob += 0.1

        # Experience effect
        success_prob += min(tutor['experience_years'] * 0.02, 0.15)

        # Gender/online compatibility
        if not gender_ok:
            success_prob -= 0.4
        if not online_ok and distance > 10:
            success_prob -= 0.3

        # Add some randomness
        success_prob += random.uniform(-0.1, 0.1)
        success_prob = max(0.05, min(0.95, success_prob))

        # Determine outcome
        success = random.random() < success_prob

        if success:
            tutor_success_counts[tutor['tutor_id']] += 1

        results.append({
            "case_id": case['case_id'],
            "tutor_id": tutor['tutor_id'],
            "distance_km": round(distance, 2),
            "price_gap": price_gap,
            "subject_match": subject_match,
            "level_match": level_match,
            "success": int(success),
            "match_date": datetime.now() - timedelta(days=random.randint(0, 365)),
        })

    # Update tutor success counts
    for tid, count in tutor_success_counts.items():
        tutors_df.loc[tutors_df['tutor_id'] == tid, 'successful_cases'] = count

    return pd.DataFrame(results)


def generate_all_data():
    """Generate complete sample dataset."""
    print("Generating tutors...")
    tutors = generate_tutors(500)

    print("Generating cases...")
    cases = generate_cases(300)

    print("Generating results...")
    results = generate_results(tutors, cases, 2000)

    return tutors, cases, results


if __name__ == "__main__":
    tutors, cases, results = generate_all_data()

    print(f"\nGenerated:")
    print(f"  - {len(tutors)} tutors")
    print(f"  - {len(cases)} cases")
    print(f"  - {len(results)} match results")
    print(f"  - Success rate: {results['success'].mean():.1%}")

    # Save to CSV
    tutors.to_csv("tutors.csv", index=False)
    cases.to_csv("cases.csv", index=False)
    results.to_csv("results.csv", index=False)
    print("\nSaved to CSV files.")
