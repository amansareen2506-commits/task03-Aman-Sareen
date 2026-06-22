# Project 3: AI Recommendation Logic - Tech Stack Recommender
# DecodeLabs Industrial Training - Batch 2026

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- DATASET: Job roles and their required skills ---
job_roles = {
    "Data Scientist": "python machine learning sql data analysis statistics numpy pandas tensorflow",
    "Web Developer": "html css javascript react nodejs frontend backend api",
    "DevOps Engineer": "aws docker kubernetes linux ci cd automation cloud git",
    "Android Developer": "java kotlin android mobile firebase api rest",
    "Cybersecurity Analyst": "networking linux ethical hacking encryption security firewall",
    "Backend Developer": "python java nodejs sql rest api databases server django",
    "AI Engineer": "python deep learning tensorflow pytorch neural networks nlp computer vision",
    "Cloud Architect": "aws azure google cloud infrastructure terraform networking automation",
    "Data Analyst": "excel sql python data visualization tableau powerbi reporting",
    "Full Stack Developer": "html css javascript react nodejs python sql mongodb rest api"
}

def get_recommendations(user_skills, top_n=3):
    # Combine dataset + user profile
    role_names = list(job_roles.keys())
    role_docs = list(job_roles.values())

    # Build TF-IDF vectors
    vectorizer = TfidfVectorizer()
    all_docs = role_docs + [user_skills]
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # User vector is the last one
    user_vector = tfidf_matrix[-1]
    role_vectors = tfidf_matrix[:-1]

    # Calculate cosine similarity
    scores = cosine_similarity(user_vector, role_vectors)[0]

    # Sort and filter top N
    ranked = sorted(zip(role_names, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

def main():
    print("=" * 50)
    print("  PROJECT 3: AI TECH STACK RECOMMENDER")
    print("=" * 50)
    print("\nEnter at least 3 skills (comma separated)")
    print("Example: python, machine learning, sql\n")

    raw_input_text = input("Your skills: ")
    user_skills = raw_input_text.lower().strip()

    # Validate minimum 3 inputs
    skill_list = [s.strip() for s in user_skills.split(",") if s.strip()]
    if len(skill_list) < 3:
        print("\n[ERROR] Please enter at least 3 skills.")
        return

    print("\n[PROCESS] Vectorizing skills using TF-IDF...")
    print("[PROCESS] Calculating Cosine Similarity scores...")

    recommendations = get_recommendations(user_skills)

    print("\n" + "=" * 50)
    print("  TOP 3 RECOMMENDED CAREER PATHS")
    print("=" * 50)

    for rank, (role, score) in enumerate(recommendations, 1):
        print(f"\n#{rank} {role}")
        print(f"    Match Score: {round(score * 100, 2)}%")

    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
