---
title: Tutor Recommendation System
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Tutor Recommendation System

Hybrid ML recommendation system with explainable predictions.

## Features

- **Two-Stage Ranking:** Hard rule filter + LightGBM probability ranking
- **SHAP Explanations:** Understand why each tutor was ranked
- **Dynamic Pricing:** See how budget changes affect success probability
- **Risk Tags:** Automatic flagging of potential issues

## Tech Stack

- LightGBM for prediction
- SHAP for explainability
- Streamlit for UI
