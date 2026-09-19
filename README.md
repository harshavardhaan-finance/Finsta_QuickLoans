# Finsta QuickLoans — Streamlit

This folder is the Streamlit version of the Finsta QuickLoans prototype.

## Files

- `streamlit_app.py` — Streamlit application
- `finsta_rf_100_trees_compressed.joblib` — compressed Random Forest model
- `finsta_model_feature_schema.json` — model feature schema
- `requirements.txt` — deployment dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Upload this folder to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Create an app.
4. Choose the GitHub repo and select `streamlit_app.py`.
5. Deploy.

The app loads the `.joblib` model directly in Python; Flask and the original HTML frontend are not required for the Streamlit version.

## Important model compatibility note

The model was serialized with scikit-learn 1.6.1, so `requirements.txt` pins that version to avoid the compatibility problem caused by loading the model under another scikit-learn version.
