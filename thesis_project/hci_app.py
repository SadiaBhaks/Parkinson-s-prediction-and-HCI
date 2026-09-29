import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="PD Motor-State Monitor", layout="centered")

SAMPLING_RATE = 100
TREMOR_BAND = (4, 6)
LABEL_NAMES = {0: "Healthy", 1: "Parkinson's", 2: "Other / Differential Diagnosis"}
BEST_K = 5
BEST_BOOST = 8

def extract_features(X):
    n_samples, timesteps, channels = X.shape
    feats = []
    freqs = np.fft.rfftfreq(timesteps, d=1.0 / SAMPLING_RATE)
    band_mask = (freqs >= TREMOR_BAND[0]) & (freqs <= TREMOR_BAND[1])
    for i in range(n_samples):
        sample = X[i]
        mean = sample.mean(axis=0)
        std = sample.std(axis=0)
        minimum = sample.min(axis=0)
        maximum = sample.max(axis=0)
        fft_vals = np.abs(np.fft.rfft(sample, axis=0))
        tremor_power = fft_vals[band_mask].mean(axis=0)
        feats.append(np.concatenate([mean, std, minimum, maximum, tremor_power]))
    return np.array(feats)

@st.cache_resource
def load_everything():
    general_clf = joblib.load("general_model.pkl")
    train_feat = np.load("train_features.npz")
    test = np.load("test_tasks.npz")
    return general_clf, train_feat["X"], train_feat["y"], test["X"], test["y"], test["ids"]

@st.cache_data
def load_personalization_results():
    try:
        return pd.read_csv("personalization_results.csv")
    except FileNotFoundError:
        return None

general_clf, X_train, y_train, X_test_raw, y_test, test_ids = load_everything()
all_results = load_personalization_results()

st.title("Parkinson's Motor-State Recognition")
st.caption("Research prototype for a final-year thesis — not a medical diagnostic tool.")
st.caption(
    "Note: results vary by patient. Across all 71 unseen test patients, personalization "
    "improved average accuracy by roughly 6.4 percentage points at k=5 (see the distribution "
    "below, and the thesis Results chapter, for the full picture)."
)

st.divider()

# ---------------------------------------------------------
# Section 1: live single-patient demo
# ---------------------------------------------------------
st.header("Try it on one patient")

unique_ids = sorted(set(test_ids))
selected_id = st.selectbox("Select a patient (simulating a new, unseen patient):", unique_ids)

idx = np.where(test_ids == selected_id)[0]
n_available = len(idx)
true_label = y_test[idx[0]]

st.write(
    f"This patient has **{n_available} recorded tasks** available. "
    f"True diagnosis (for evaluation purposes only): **{LABEL_NAMES[true_label]}**"
)

k = st.slider(
    "Number of this patient's own tasks to use for personalization:",
    1, min(5, n_available - 1), min(BEST_K, n_available - 1)
)

if st.button("Run comparison"):
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(idx)
    personalize_idx = shuffled[:k]
    eval_idx = shuffled[k:]

    X_person_raw = X_test_raw[personalize_idx]
    y_person = y_test[personalize_idx]
    X_eval_raw = X_test_raw[eval_idx]
    y_eval = y_test[eval_idx]

    X_person = extract_features(X_person_raw)
    X_eval = extract_features(X_eval_raw)

    # --- General model prediction ---
    general_preds = general_clf.predict(X_eval)
    general_proba = general_clf.predict_proba(X_eval)
    general_acc = (general_preds == y_eval).mean()

    # --- Personalized model (retrained live with boosted weighting) ---
    X_aug = np.vstack([X_train, X_person])
    y_aug = np.concatenate([y_train, y_person])
    sample_weights = np.concatenate([np.ones(len(X_train)), np.full(len(X_person), float(BEST_BOOST))])

    personalized_clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    personalized_clf.fit(X_aug, y_aug, sample_weight=sample_weights)
    personalized_preds = personalized_clf.predict(X_eval)
    personalized_proba = personalized_clf.predict_proba(X_eval)
    personalized_acc = (personalized_preds == y_eval).mean()

    st.divider()
    st.subheader("Results on this patient's remaining, held-out tasks")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**General Model** (before personalization)")
        st.metric("Accuracy on held-out tasks", f"{general_acc:.0%}")
        majority_pred = np.bincount(general_preds).argmax()
        st.write(f"Most common prediction: **{LABEL_NAMES[majority_pred]}**")
        avg_confidence = general_proba.max(axis=1).mean()
        st.progress(float(avg_confidence), text=f"Avg. confidence: {avg_confidence:.0%}")

    with col2:
        st.markdown(f"**Personalized Model** (using {k} of this patient's tasks)")
        st.metric("Accuracy on held-out tasks", f"{personalized_acc:.0%}",
                   delta=f"{personalized_acc - general_acc:+.0%}")
        majority_pred_p = np.bincount(personalized_preds).argmax()
        st.write(f"Most common prediction: **{LABEL_NAMES[majority_pred_p]}**")
        avg_confidence_p = personalized_proba.max(axis=1).mean()
        st.progress(float(avg_confidence_p), text=f"Avg. confidence: {avg_confidence_p:.0%}")

    st.divider()
    if personalized_acc > general_acc:
        st.success(f"Personalization improved accuracy on this patient by {personalized_acc - general_acc:.0%}.")
    elif personalized_acc < general_acc:
        st.warning(
            "For this particular patient, personalization did not improve accuracy — "
            "this happens for some patients and is discussed as a limitation in the thesis."
        )
    else:
        st.info("Personalization made no difference for this patient.")

# ---------------------------------------------------------
# Section 2: results across ALL patients (the honest full picture)
# ---------------------------------------------------------
st.divider()
st.header("Results across all unseen patients")
st.write(
    "The single-patient demo above shows one example. The chart below shows what "
    "actually happened across every one of the 71 unseen test patients, using the "
    "best personalization setting found during experimentation (k=5, boost=8)."
)

if all_results is not None:
    subset = all_results[(all_results["k"] == BEST_K) & (all_results["boost"] == BEST_BOOST)].copy()
    subset["improvement"] = subset["after_acc"] - subset["before_acc"]

    mean_before = subset["before_acc"].mean()
    mean_after = subset["after_acc"].mean()
    mean_improvement = subset["improvement"].mean()
    pct_improved = (subset["improvement"] > 0).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Average accuracy before", f"{mean_before:.0%}")
    c2.metric("Average accuracy after", f"{mean_after:.0%}", delta=f"{mean_improvement:+.1%}")
    c3.metric("Patients who improved", f"{pct_improved:.0%}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(subset["improvement"], bins=15, color="#444444", edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(mean_improvement, color="red", linestyle="-", linewidth=1.5,
               label=f"Mean improvement: {mean_improvement:+.1%}")
    ax.set_xlabel("Accuracy improvement after personalization (per patient)")
    ax.set_ylabel("Number of patients")
    ax.set_title("Distribution of Personalization Improvement Across 71 Unseen Patients")
    ax.legend()
    st.pyplot(fig)

    st.caption(
        "Patients to the right of the dashed line improved with personalization; "
        "patients to the left did not. The red line marks the average effect."
    )
else:
    st.warning(
        "personalization_results.csv not found in this folder — copy it here from "
        "your thesis_project directory to see the full-patient distribution chart."
    )

st.divider()
st.caption(
    "This tool is a research prototype developed for a final-year thesis on personalized "
    "motor-state recognition. It is not intended for clinical diagnosis or treatment decisions."
)