from __future__ import annotations

import pandas as pd
import streamlit as st

from train_demo import load_or_rebuild_bundle

try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except ModuleNotFoundError:
    px = None
    go = None

st.set_page_config(page_title="Multimodal RAG + Churn Demo", page_icon="AI", layout="wide")


@st.cache_resource
def get_bundle() -> dict:
    return load_or_rebuild_bundle()


def _risk_band(probability: float) -> tuple[str, str]:
    if probability >= 0.7:
        return "High", "#B03A2E"
    if probability >= 0.4:
        return "Moderate", "#C97A00"
    return "Low", "#1D7F5F"


def _plotly_available() -> bool:
    return px is not None and go is not None


def _show_plotly_warning() -> None:
    st.info("Plotly is not installed in this environment. Showing simplified Streamlit charts instead.")


def _render_contract_mix(contract_mix: pd.DataFrame) -> None:
    if _plotly_available():
        fig = px.bar(
            contract_mix,
            x="contract_type",
            y="customers",
            color="contract_type",
            title="Customer Mix by Contract Type",
            color_discrete_sequence=["#0A7B83", "#C97A00", "#A23B72"],
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    _show_plotly_warning()
    fallback = contract_mix.set_index("contract_type")[["customers"]]
    st.bar_chart(fallback, use_container_width=True)


def _render_model_metrics(results: pd.DataFrame) -> None:
    if _plotly_available():
        result_fig = px.bar(
            results,
            x="model",
            y=["roc_auc", "f1", "recall"],
            barmode="group",
            title="Model Metrics Comparison",
            color_discrete_sequence=["#0A7B83", "#A23B72", "#C97A00"],
        )
        st.plotly_chart(result_fig, use_container_width=True)
        return

    _show_plotly_warning()
    fallback = results.set_index("model")[["roc_auc", "f1", "recall"]]
    st.bar_chart(fallback, use_container_width=True)


def _render_confusion_matrix(confusion_matrix: list[list[int]]) -> None:
    if _plotly_available():
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=confusion_matrix,
                x=["Pred 0", "Pred 1"],
                y=["Actual 0", "Actual 1"],
                colorscale="Blues",
            )
        )
        cm_fig.update_layout(height=360)
        st.plotly_chart(cm_fig, use_container_width=True)
        return

    _show_plotly_warning()
    st.dataframe(
        pd.DataFrame(confusion_matrix, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
        use_container_width=True,
    )


def _render_feature_importance(feature_importance: pd.DataFrame) -> None:
    if _plotly_available():
        imp_fig = px.bar(
            feature_importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance",
            color="importance",
            color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(imp_fig, use_container_width=True)
        return

    _show_plotly_warning()
    fallback = feature_importance.sort_values("importance").set_index("feature")[["importance"]]
    st.bar_chart(fallback, use_container_width=True)


def _render_monthly_spend_distribution(dataframe: pd.DataFrame) -> None:
    if _plotly_available():
        dist_fig = px.histogram(
            dataframe,
            x="monthly_spend",
            color="churned",
            nbins=24,
            barmode="overlay",
            title="Monthly Spend Distribution by Churn",
            color_discrete_sequence=["#1D7F5F", "#B03A2E"],
        )
        st.plotly_chart(dist_fig, use_container_width=True)
        return

    _show_plotly_warning()
    grouped = dataframe.groupby("churned", as_index=False)["monthly_spend"].mean()
    st.bar_chart(grouped.set_index("churned"), use_container_width=True)


def _render_regional_churn(region_df: pd.DataFrame) -> None:
    if _plotly_available():
        region_fig = px.bar(
            region_df,
            x="region",
            y="churn_rate",
            title="Regional Churn Rate",
            color="churn_rate",
            color_continuous_scale="Sunsetdark",
        )
        st.plotly_chart(region_fig, use_container_width=True)
        return

    _show_plotly_warning()
    st.bar_chart(region_df.set_index("region")[["churn_rate"]], use_container_width=True)


def _render_probability_distribution(probabilities: list[float]) -> None:
    if _plotly_available():
        prob_fig = px.bar(
            pd.DataFrame({"class": ["Retain", "Churn"], "probability": probabilities}),
            x="class",
            y="probability",
            color="class",
            title="Class Probability Distribution",
            color_discrete_sequence=["#1D7F5F", "#B03A2E"],
        )
        prob_fig.update_yaxes(range=[0, 1])
        st.plotly_chart(prob_fig, use_container_width=True)
        return

    _show_plotly_warning()
    fallback = pd.DataFrame({"probability": probabilities}, index=["Retain", "Churn"])
    st.bar_chart(fallback, use_container_width=True)


bundle = get_bundle()
df = pd.DataFrame(bundle["full_dataframe"])
results_df = pd.DataFrame(bundle["model_results"])
sample_predictions_df = pd.DataFrame(bundle["sample_predictions"])
feature_importance_df = pd.DataFrame(bundle["feature_importance"])
schema = bundle["feature_schema"]
best_model = bundle["best_model"]

st.title("Real-Time Multimodal RAG System Demo Dashboard")
st.caption(
    "Portfolio dashboard for a production-oriented demo bundle with reproducible training, live inference, and deployment-facing system notes."
)

tabs = st.tabs(["Overview", "Model Results", "Analytics", "Pipeline/API", "Predict"])

with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    top_row = results_df.iloc[0]
    col1.metric("Best Model", bundle["best_model_name"])
    col2.metric("ROC-AUC", f"{top_row['roc_auc']:.3f}")
    col3.metric("Customers", f"{len(df):,}")
    col4.metric("Churn Rate", f"{bundle['analytics']['target_rate']:.1%}")

    st.markdown(
        "This dashboard complements the RAG API by demonstrating reproducible model packaging, artifact bundling, and self-healing demo inference. "
        "If the bundle is missing or corrupt, the app rebuilds it automatically from `train_demo.py`."
    )

    overview_cols = st.columns([1.4, 1])
    with overview_cols[0]:
        contract_mix = pd.DataFrame(bundle["analytics"]["contract_mix"])
        _render_contract_mix(contract_mix)
    with overview_cols[1]:
        st.subheader("Bundle Snapshot")
        st.json(
            {
                "bundle_version": bundle["bundle_version"],
                "generated_at": bundle["generated_at"],
                "train_rows": bundle["train_rows"],
                "test_rows": bundle["test_rows"],
            }
        )

with tabs[1]:
    st.subheader("Model Comparison")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    _render_model_metrics(results_df)

    lower_cols = st.columns(2)
    with lower_cols[0]:
        st.subheader("Best Model Confusion Matrix")
        cm = bundle["confusion_matrix"]
        _render_confusion_matrix(cm)
    with lower_cols[1]:
        st.subheader("Top Feature Signals")
        if not feature_importance_df.empty:
            _render_feature_importance(feature_importance_df)
        else:
            st.info("Feature importance is unavailable for the selected model.")

with tabs[2]:
    st.subheader("Analytics")
    analytics_cols = st.columns(2)
    with analytics_cols[0]:
        _render_monthly_spend_distribution(df)
    with analytics_cols[1]:
        region_df = pd.DataFrame(bundle["analytics"]["regional_churn"])
        _render_regional_churn(region_df)

    st.subheader("Prediction Sample")
    st.dataframe(sample_predictions_df, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Pipeline and API")
    st.markdown(
        "The project ships two delivery surfaces: a FastAPI retrieval service and this Streamlit dashboard. "
        "The dashboard uses a persisted bundle that contains the trained model, full analytics dataset, metrics, and feature schema."
    )
    st.code(
        """python train_demo.py
streamlit run streamlit_app.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000""",
        language="bash",
    )
    st.markdown(
        "API request body includes `mode`, `top_k`, `use_hybrid`, and reranking controls. The FastAPI service also exposes `/metrics` and `/metrics/prometheus` for observability."
    )
    st.json(
        {
            "bundle_keys": sorted(bundle.keys()),
            "api_routes": ["GET /health", "GET /metrics", "GET /metrics/prometheus", "POST /v1/query"],
        }
    )

with tabs[4]:
    st.subheader("Live Prediction")
    with st.form("predict-form"):
        input_cols = st.columns(2)
        with input_cols[0]:
            age = st.slider("Age", 18, 76, int(schema["numeric"]["age"]["default"]))
            tenure_months = st.slider("Tenure (months)", 1, 72, int(schema["numeric"]["tenure_months"]["default"]))
            monthly_spend = st.slider(
                "Monthly Spend",
                float(schema["numeric"]["monthly_spend"]["min"]),
                float(schema["numeric"]["monthly_spend"]["max"]),
                float(schema["numeric"]["monthly_spend"]["default"]),
            )
            support_tickets = st.slider("Support Tickets", 0, 12, int(schema["numeric"]["support_tickets"]["default"]))
            usage_score = st.slider(
                "Usage Score",
                float(schema["numeric"]["usage_score"]["min"]),
                float(schema["numeric"]["usage_score"]["max"]),
                float(schema["numeric"]["usage_score"]["default"]),
            )
        with input_cols[1]:
            satisfaction_score = st.slider(
                "Satisfaction Score",
                float(schema["numeric"]["satisfaction_score"]["min"]),
                float(schema["numeric"]["satisfaction_score"]["max"]),
                float(schema["numeric"]["satisfaction_score"]["default"]),
            )
            contract_type = st.selectbox("Contract Type", schema["categorical"]["contract_type"])
            region = st.selectbox("Region", schema["categorical"]["region"])
            autopay = st.selectbox("Autopay", schema["categorical"]["autopay"])
            paperless = st.selectbox("Paperless Billing", schema["categorical"]["paperless"])

        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        sample = pd.DataFrame(
            [
                {
                    "age": age,
                    "tenure_months": tenure_months,
                    "monthly_spend": monthly_spend,
                    "support_tickets": support_tickets,
                    "usage_score": usage_score,
                    "satisfaction_score": satisfaction_score,
                    "contract_type": contract_type,
                    "region": region,
                    "autopay": autopay,
                    "paperless": paperless,
                }
            ]
        )
        probabilities = best_model.predict_proba(sample)[0]
        churn_probability = float(probabilities[1])
        confidence = churn_probability if churn_probability >= 0.5 else 1.0 - churn_probability
        risk_label, risk_color = _risk_band(churn_probability)

        metric_cols = st.columns(3)
        metric_cols[0].metric("Predicted Class", "Churn" if churn_probability >= 0.5 else "Retain")
        metric_cols[1].metric("Confidence", f"{confidence:.1%}")
        metric_cols[2].markdown(
            f"<div style='padding:0.6rem 0.8rem;border-radius:0.6rem;background:{risk_color};color:white;font-weight:600;text-align:center;'>Risk: {risk_label}</div>",
            unsafe_allow_html=True,
        )

        _render_probability_distribution([float(probabilities[0]), churn_probability])

        with st.expander("Input Summary"):
            st.dataframe(sample, use_container_width=True, hide_index=True)
