# 📚 README: Data Pipeline Tracker

## Project: Data Pipeline Tracker

This is a professional, end-to-end Machine Learning Operations (MLOps) dashboard built to automate the data processing, model experimentation, and deployment lifecycle for predictive models.

The application is designed for stability and efficiency on cloud hosting platforms (like Railway/Render).

### 🚀 Key Features

* **Custom Pipeline Orchestration:** Uses Prefect for defining and running reproducible workflows (Ingestion -> Preprocessing -> Training -> Registration).
* **Full Model Experimentation:** Automatically trains a diverse suite of 10+ classification/regression models (including high-performance ensembles like Random Forest, XGBoost, and CatBoost).
* **Resource Efficiency:** Implements **data type optimization (float32/int8)** and **parallel processing (`n_jobs=-1`)** to maintain stability and performance on resource-constrained cloud environments (e.g., Free Tier hosting).
* **Robust Data Handling:** Features **Univariate Feature Selection** to manage high-dimensional data and dynamic target encoding to handle mixed categorical/numerical input and prevent data leakage.
* **MLOps Toolchain:** Integrates Streamlit for the front-end, MLflow for experiment tracking and model registration, and Prefect for workflow orchestration.

### ⚙️ Technical Stack

| Category | Component | Purpose |
| :--- | :--- | :--- |
| **Front-end** | Streamlit | Interactive web dashboard and UI. |
| **Orchestration** | Prefect | Automates the execution and sequencing of pipeline tasks. |
| **ML/Data** | Scikit-learn, Pandas, NumPy | Data manipulation, preprocessing, and model implementation. |
| **Tracking/Registry** | MLflow | Logs metrics, versions models, and handles deployment staging. |
| **Deployment Target** | Render / Railway | Dedicated hosting for stable, resource-intensive operations. |

### 🧭 Workflow Stages

1. **⬆️ Upload Data:** Ingestion of data (CSV/PKL) directly via the browser.  
2. **📊 Data Analysis:** Exploratory Data Analysis (EDA) and visualization (Histograms, Pie Charts, Correlation).  
3. **⚙️ Run Process:** Triggers the **Continuous Training (CT)** pipeline.  
4. **🏆 Compare Results:** Displays logged MLflow metrics for all models to select the best performer.  
5. **📈 View Reports:** Generates final data and model performance reports for download.

---

### 🛠️ Local Setup Instructions

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/YourUsername/YourRepoName.git
    cd YourRepoName
    ```

2. **Create and Activate Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate  # macOS/Linux
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run Application:**
    ```bash
    streamlit run streamlit_app/main.py
    ```

