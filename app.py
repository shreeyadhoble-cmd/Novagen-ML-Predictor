import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Novagen ML Predictor",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 Novagen — ML Prediction")

st.write(
    "Random Forest classification model based on the Novagen notebook."
)


TARGET = "Target"


# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------

@st.cache_resource
def load_saved_model():
    if os.path.exists("novagen_random_forest.pkl"):
        return joblib.load("novagen_random_forest.pkl")

    return None


# --------------------------------------------------
# LOAD DEFAULT DATASET
# --------------------------------------------------

@st.cache_data
def load_default_data():
    if os.path.exists("novagen_dataset.csv"):
        return pd.read_csv("novagen_dataset.csv")

    return None


model = load_saved_model()
df = load_default_data()


# --------------------------------------------------
# SIDEBAR — DATASET UPLOAD
# --------------------------------------------------

with st.sidebar:

    st.header("Dataset")

    uploaded = st.file_uploader(
        "Upload novagen_dataset.csv",
        type=["csv"]
    )

    if uploaded is not None:

        df = pd.read_csv(uploaded)

        # Use uploaded dataset instead of saved model
        model = None

        st.success("Dataset loaded successfully.")


# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------

if df is None:

    st.info(
        "Upload the Novagen dataset CSV to start the predictor."
    )

    st.stop()


if TARGET not in df.columns:

    st.error(
        "The dataset must contain a column named 'Target'."
    )

    st.stop()


# --------------------------------------------------
# SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=[TARGET])
y = df[TARGET]


# --------------------------------------------------
# CONVERT BOOLEAN FEATURES TO 0/1
# --------------------------------------------------

for col in X.columns:

    # Actual Boolean values
    if X[col].dtype == bool:

        X[col] = X[col].astype(int)


# --------------------------------------------------
# CONVERT TEXT TRUE/FALSE TO 0/1
# --------------------------------------------------

for col in X.columns:

    if X[col].dtype == object:

        values = (
            X[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        unique_values = set(
            values.dropna().unique()
        )

        if unique_values.issubset({"true", "false"}):

            X[col] = values.map({
                "true": 1,
                "false": 0
            })


# --------------------------------------------------
# CONVERT REMAINING FEATURES TO NUMERIC
# --------------------------------------------------

for col in X.columns:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

for col in X.columns:

    if X[col].isna().any():

        median_value = X[col].median()

        if pd.isna(median_value):
            median_value = 0

        X[col] = X[col].fillna(median_value)


# --------------------------------------------------
# TRAIN RANDOM FOREST IF MODEL NOT AVAILABLE
# --------------------------------------------------

if model is None:

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        st.caption(
            "Random Forest trained from the uploaded dataset."
        )

    except Exception as e:

        st.error(
            f"Could not train the model: {e}"
        )

        st.stop()


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.subheader("Dataset")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rows",
    len(df)
)

col2.metric(
    "Features",
    len(X.columns)
)

col3.metric(
    "Classes",
    y.nunique()
)


# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

with st.expander("Preview dataset"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# --------------------------------------------------
# CHECK FOR NON-NUMERIC FEATURES
# --------------------------------------------------

st.divider()

st.subheader("Make a Prediction")


non_numeric = X.select_dtypes(
    exclude="number"
).columns.tolist()


if non_numeric:

    st.warning(
        "Some features are still non-numeric: "
        + ", ".join(map(str, non_numeric))
    )

    st.stop()


# --------------------------------------------------
# CREATE INPUT FIELDS
# --------------------------------------------------

inputs = {}

cols = st.columns(2)


for i, feature in enumerate(X.columns):

    series = X[feature].dropna()

    if len(series):

        default = float(series.median())

        minimum = float(series.min())

        maximum = float(series.max())

    else:

        default = 0.0

        minimum = 0.0

        maximum = 1.0


    # Avoid equal min and max
    if minimum == maximum:

        minimum -= 1.0

        maximum += 1.0


    step = (
        maximum - minimum
    ) / 100


    if step == 0:

        step = 0.01


    with cols[i % 2]:

        inputs[feature] = st.number_input(

            str(feature),

            min_value=minimum,

            max_value=maximum,

            value=min(
                max(
                    default,
                    minimum
                ),
                maximum
            ),

            step=step
        )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔮 Predict",
    type="primary",
    use_container_width=True
):

    input_df = pd.DataFrame(
        [inputs],
        columns=X.columns
    )


    try:

        prediction = model.predict(
            input_df
        )[0]


        st.success(
            f"Prediction: **{prediction}**"
        )


        # Prediction probabilities
        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_df
            )[0]


            st.write(
                "### Prediction probabilities"
            )


            prob_df = pd.DataFrame({

                "Class": model.classes_,

                "Probability": probabilities

            })


            prob_df["Probability"] = (
                prob_df["Probability"]
                .map(
                    lambda v: f"{v:.2%}"
                )
            )


            st.dataframe(
                prob_df,
                hide_index=True,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Model: RandomForestClassifier "
    "(n_estimators=200, random_state=42)"
)