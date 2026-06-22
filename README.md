# AI Recommendation Logic DecodeLabs Project 3

A content-based Tech Stack Recommender built using Python and 
scikit-learn as part of the DecodeLabs AI Industrial Training (Batch 2026).

## How It Works
- User enters 3+ skills as input
- Skills are vectorized using TF-IDF weighting
- Cosine Similarity is calculated against 10 job role profiles
- Top 3 most relevant career paths are returned ranked by match score

## Pipeline
- Input: 3+ user skills (comma separated)
- Process: TF-IDF vectorization + Cosine Similarity scoring
- Output: Top-3 ranked job role recommendations with match %

## Example
Your skills: python, machine learning, sql
#1 Data Scientist       Match Score: 87.4%

#2 AI Engineer          Match Score: 71.2%

#3 Data Analyst         Match Score: 65.8%

## How to Run
```
py recommender.py
```

## Dependencies
```
py -m pip install scikit-learn
```
