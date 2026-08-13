import streamlit as st

st.set_page_config(
    page_title="PhD ML Research Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 PhD ML Research Lab")
st.markdown("---")

st.markdown("""
Welcome to the PhD ML Research Lab. Use the sidebar to navigate between experiments.

### Available Experiments
""")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("### 🔍 Weakspot Identification Experiment")
    st.info(
        "Induce a data gap in a 2D regression dataset, train a model, "
        "then evaluate the weakspot detection methods side by side.\n\n"
        "**Adjustable:** exclusion radius, model complexity, training iterations, test/train split."
    )

with col2:
    st.markdown("### 📈 Weakspot Experiment — Visualise Results")
    st.info(
        "Aggregated comparison of the detection methods across every "
        "parameter-sweep run saved to the results CSV.\n\n"
        "**Shows:** method ranking, metric distributions, and parameter sensitivity."
    )

with col3:
    st.markdown("### 🎯 Data-Selective Training")
    st.info(
        "Train a model, find where it is weakest, feed it new data **where it "
        "knows the least**, retrain, and compare before vs after — with a random "
        "baseline. Single run or a resumable parameter sweep.\n\n"
        "**Steps:** Setup → Training → Evaluation → Weakspot ID → Data Selection "
        "→ Retraining → Re-Evaluation. Aggregate results in "
        "**Data Selective Training — Visualise Results**."
    )

with col4:
    st.markdown("### 🔁 Iterative Data-Selective Training")
    st.info(
        "The follow-up study: run the loop **many** rounds — detect → select → "
        "continue training — with a random baseline advancing in parallel from "
        "the same candidate pool.\n\n"
        "**Adds:** per-iteration schedules for the learning rate and for the two "
        "coverage controls (rehearsal mix α, kernel width σ), MLP architecture "
        "variants, accumulative vs new-only vs size-matched regimes, and a "
        "budget-matched single-shot control. Single run or a resumable sweep."
    )

with col5:
    st.markdown("### 📈 Iterative — Visualise Results")
    st.info(
        "Aggregated **trajectories** from the iterative sweep: learning curves "
        "with confidence bands, the guided-vs-random advantage per iteration, "
        "forgetting outside the weakspot, and the effect of architecture, "
        "learning-rate schedule, coverage schedules and staging."
    )

st.markdown("---")
st.caption("Use the **sidebar navigation** (top-left ☰) to open an experiment page.")
