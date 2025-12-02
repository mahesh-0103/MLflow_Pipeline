# app_modules/2_eda.py
import streamlit as st
import pandas as pd
from src.visualization.visualize import (
    heatmap_correlation,
    distribution_plot,
    histograms_per_feature,
    pairwise_scatter,
    correlation_matrix_values
)
import plotly.express as px # Added for pie chart visualization


def app():
    st.subheader("Statistical Summaries and Distribution Checks")
    st.markdown("---")
    df = st.session_state.get("df")
    if df is None:
        st.warning("Upload a dataset first on the **⬆️ Upload Data** page.")
        return

    st.subheader("Data Summary & Statistics")
    # DEPRECATION FIX
    with st.expander("Expand to see descriptive statistics"):
        st.dataframe(df.describe(include="all").T, width='stretch')
        
    st.markdown("---")

    # Filter out Boolean columns for plotting continuous distribution 
    num_cols = [
        c for c in df.columns 
        if pd.api.types.is_numeric_dtype(df[c]) and 
           not pd.api.types.is_bool_dtype(df[c])
    ]
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not num_cols and not cat_cols:
        st.info("No viable features found for analysis.")
    else:
        st.subheader("📈 Univariate & Bivariate Plots")
        
        # --- NEW: PIECHART VISUALIZATION (Categorical Data) ---
        if cat_cols:
            with st.container(border=True):
                st.markdown("#### 🥧 Categorical Distribution (Pie Chart)")
                col_to_plot = st.selectbox("Choose categorical column for distribution", cat_cols, key="eda_pie_col")
                
                if st.button("Generate Pie Chart"):
                    value_counts = df[col_to_plot].value_counts().reset_index()
                    value_counts.columns = ['Category', 'Count']
                    
                    fig = px.pie(
                        value_counts, 
                        values='Count', 
                        names='Category', 
                        title=f'Distribution of {col_to_plot.title()}',
                        hole=.3
                    )
                    # DEPRECATION FIX: Plotly charts use use_container_width which is not affected by this warning
                    st.plotly_chart(fig, use_container_width=True)

        # Distribution Plot section (Numeric Data)
        if num_cols:
            with st.container(border=True):
                st.markdown("#### 📉 Feature Distribution (Histogram)")
                col = st.selectbox("Choose numeric column for histogram", num_cols, key="eda_dist_col")
                if st.button("Generate Distribution Plot"):
                    path = distribution_plot(df, col)
                    if path:
                        # DEPRECATION FIX
                        st.image(path, width='stretch')
                        st.markdown(f"**Saved to:** `{path}`")
        
        # Correlation Heatmap section 
        with st.container(border=True):
            st.markdown("#### 🌡️ Correlation Heatmap")
            if st.button("Generate Heatmap Correlation"):
                path = heatmap_correlation(df)
                if path:
                    # DEPRECATION FIX
                    st.image(path, width='stretch')
                    st.markdown(f"**Saved to:** `{path}`")
                    
        # Other plots in an expander to save space
        with st.expander("More Plots (Histograms & Pairwise Scatter)"):
            if st.button("Generate Histograms for all eligible numeric columns"):
                st.info(f"Generating histograms for {len(num_cols)} columns...")
                paths = histograms_per_feature(df[num_cols])
                for p in paths:
                    st.image(p, width=500)
                    st.markdown(f"`{p}`")

            if len(num_cols) >= 2 and st.button("Generate Pairwise Scatter (first 6 eligible numeric columns)"):
                st.info("Generating Pairwise Scatter for first 6 numeric columns...")
                p = pairwise_scatter(df[num_cols])
                if p:
                    # DEPRECATION FIX
                    st.image(p, width='stretch')
                    st.markdown(f"`{p}`")