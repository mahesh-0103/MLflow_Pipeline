# streamlit_app/main.py
import streamlit as st
from pathlib import Path
import sys
import importlib.util
import traceback
import re 

# Add project root to sys.path so imports like `from src...` work when Streamlit runs main.py
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Set Page Config ---
st.set_page_config(
    page_title="Data Pipeline Tracker | MLOps Dashboard",
    layout="wide",
    initial_sidebar_state="expanded", # Set back to expanded
    menu_items={'About': "# Data Pipeline Tracker - A professional MLOps demo."}
)

# --- Session defaults (safe keys used by pages) ---
if "df" not in st.session_state: st.session_state["df"] = None
if "train_runs" not in st.session_state: st.session_state["train_runs"] = None
if "target_col" not in st.session_state: st.session_state["target_col"] = None
if "metric_name" not in st.session_state: st.session_state["metric_name"] = None
if "maximize" not in st.session_state: st.session_state["maximize"] = None
if "model_name" not in st.session_state: st.session_state["model_name"] = None
# Remove 'current_tab' and 'theme' state if present, or ignore them

# --- Setup Navigation and Page Imports ---
PAGES_DIR = Path(__file__).parent / "app_modules"

# Define the exact navigation order and names
NAVIGATION_ORDER = [
    "⬆️ Upload Data",
    "📊 Data Analysis",
    "⚙️ Run Process",
    "🏆 Compare Results",
    "📈 View Reports"
]

# Map file stems to their corresponding display names
FILE_STEM_MAP = {
    "1_upload": "⬆️ Upload Data",
    "2_eda": "📊 Data Analysis",
    "3_train": "⚙️ Run Process",
    "4_model_compare": "🏆 Compare Results",
    "5_visualize": "📈 View Reports"
}

# Collect and map page files
page_files = sorted([p for p in PAGES_DIR.glob("*.py") if p.name != "__init__.py" and not p.name.startswith(".")])

def friendly_name_from_path(p: Path) -> str:
    stem = p.stem
    stem2 = re.sub(r"^[0-9]+[_\\-]*", "", stem)
    
    # Use mapping based on file stem
    title_map = {stem: title for stem, title in FILE_STEM_MAP.items()}
    # Find the corresponding title from the file stem (e.g., '1_upload' -> '⬆️ Upload Data')
    for stem_key, nav_name in FILE_STEM_MAP.items():
        if stem_key.endswith(stem2):
            return nav_name
    return stem2.replace("_", " ").title()

page_name_map = {friendly_name_from_path(p): p for p in page_files}
page_names = [name for name in NAVIGATION_ORDER if name in page_name_map] # Ensure correct order

# Helper: import module by file path and return module object
def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None: raise ImportError(f"Cannot create spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Sidebar Content (Vertical Navigation) ---
with st.sidebar:
    st.title("Data Pipeline Tracker") # Project Name
    st.markdown("**Organized Workflow Interface**") # Subtitle
    st.markdown("---")
    
    # Navigation Section
    st.caption("Navigation Stages") # Use the caption shown in the image
    
    if page_names:
        selected = st.radio("**Select Stage**", page_names, index=0, key="sidebar_nav")
    else:
        st.info("No pages found.")
        selected = None
        
    st.markdown("---")
    
    # Dynamic Session Tips based on current page
    current_page = selected or NAVIGATION_ORDER[0] # Default to the first page if none selected
    TIPS_MAP = {
        "⬆️ Upload Data": ("Upload CSV/PKL/JSON via browser. This is the starting point for your data workflow.", 
                            "- Uploading data is the first step.",
                            "- Ensure your dataset is cleaned before uploading for best results."),
        "📊 Data Analysis": ("Explore data quality and relationships (EDA). Visualizations here check features and distributions.",
                            "- Look for data distribution skewness.",
                            "- Use the Pie Chart for categorical column counts."),
        "⚙️ Run Process": ("Configure parameters and execute the MLOps pipeline. This step trains all candidate models.",
                            "- Select the correct Target Column.",
                            "- Training is optimized for stability, but expect a wait time."),
        "🏆 Compare Results": ("Analyze all model runs. Review metrics (RMSE, F1-Score) and select the best model.",
                            "- Use the table to compare model performance directly.",
                            "- Metrics are dynamically filtered based on run results."),
        "📈 View Reports": ("Generate and download comprehensive data and model performance reports.",
                            "- The Full Automated Report includes key plots and charts.",
                            "- Download the full ZIP archive for external sharing.")
    }
    
    tips_title, tip1, tip2 = TIPS_MAP.get(current_page)
    with st.expander("✨ Stage Insights", expanded=True):
        st.markdown(f"**{tips_title}**")
        st.markdown(tip1)
        st.markdown(tip2)


# --- Main Content Area and Dynamic Heading ---

MAIN_HEADING_MAP = {
    "⬆️ Upload Data": "Data Ingestion Stage",
    "📊 Data Analysis": "Exploratory Analysis (EDA)",
    "⚙️ Run Process": "Current Stage: Training Pipeline",
    "🏆 Compare Results": "Results Review & Model Registry",
    "📈 View Reports": "Visualization and Reporting"
}
display_heading = MAIN_HEADING_MAP.get(selected, "Data Pipeline Tracker")

st.markdown(f"# {display_heading}")
st.write("---") 

# Load and run selected page
if selected:
    page_path = page_name_map[selected]
    module_name = f"streamlit_pages.{page_path.stem}"
    try:
        page_module = import_module_from_path(module_name, page_path)
    except Exception as e:
        st.error(f"❌ Failed to load page module `{page_path.name}`.")
        st.exception(e)
        st.code(traceback.format_exc(), language="python")
    else:
        if hasattr(page_module, "app"):
            try:
                page_module.app()
            except Exception as e:
                st.error(f"⚠️ An exception occurred while running `{page_path.name}`:")
                st.exception(e)
                st.code(traceback.format_exc(), language="python")
        else:
            st.warning(f"Page `{page_path.name}` loaded but no `app()` function found.")