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

col1, = st.columns(1)

with col1:
    st.markdown("### 🔍 Weakspot Identification Experiment")
    st.info(
        "Induce a data gap in a 2D regression dataset, train a model, "
        "then evaluate 8 weakspot detection methods side by side.\n\n"
        "**Adjustable:** exclusion radius, model complexity, training iterations, test/train split."
    )

st.markdown("---")
st.caption("Use the **sidebar navigation** (top-left ☰) to open an experiment page.")
