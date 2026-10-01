# Novagen Streamlit App

This project turns the Novagen machine-learning notebook into a Streamlit web app.

## Files
- `app.py` — Streamlit application
- `Novagen.ipynb` — original ML notebook
- `requirements.txt` — Python dependencies
- `novagen_dataset.csv` — place the dataset here, or upload it from the app sidebar
- `novagen_random_forest.pkl` — optional saved Random Forest model from the notebook
- `novagen_scaler.pkl` — optional scaler from the notebook; the Random Forest in the notebook does **not** use scaling

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload this folder/repository to a GitHub repository and deploy `app.py` with Streamlit Community Cloud.
