import streamlit as st
import pandas as pd
import os
import shutil
import tempfile
from pathlib import Path
import logging
import zipfile
import io
import sys
import json
import time

# Ensure current directory is in path
sys.path.append(os.getcwd())

from workflow import OptimizedMicrokineticModeling
from config import SolverSettings
from plotting import create_plots, CoveragePlotter

# Configure logging
log_capture_string = io.StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.INFO)
logging.getLogger().addHandler(ch)

st.set_page_config(
    page_title="Microkinetic Modeling",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 0.5rem 1rem; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #45a049; }
    .sidebar .sidebar-content { background-color: #262730; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #ffffff; }
    .stTextInput>div>div>input { color: #ffffff; background-color: #262730; }
    .stNumberInput>div>div>input { color: #ffffff; }
    span[data-baseweb="tag"] { background-color: #262730; }
</style>
""", unsafe_allow_html=True)

st.title("⚛️ Transient Electrochemical Microkinetic Modeling Application")
st.markdown("""
**Here, we perform unsteady-state MKM (USS-MKM) with and without potential sweeping to capture transient dynamics and realistically model reaction kinetics.**
""")
st.markdown("Run optimized microkinetic simulations based on your Excel input and configuration.")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

# 1. File Upload
uploaded_file = st.sidebar.file_uploader("Upload Input Excel", type=["xlsx", "xls"])

# 2. Executable Path
import platform
import stat
if platform.system() == "Windows":
    exe_path = "D:/mkmcxx/mkmcxx-2.15.3-windows-x64/mkmcxx_2.15.3/bin/mkmcxx.exe"
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(base_dir, "mkmcxx")
    try:
        if not os.path.exists(exe_path):
            st.error(f"⚠️ Binary not found at: {exe_path}")
        else:
            st_mode = os.stat(exe_path).st_mode
            os.chmod(exe_path, st_mode | stat.S_IEXEC)
            new_mode = os.stat(exe_path).st_mode
            if not (new_mode & stat.S_IEXEC):
                logging.getLogger().warning(f"Failed to set executable bit! Mode: {oct(new_mode)}")
    except Exception as e:
        logging.getLogger().warning(f"Could not set executable permissions: {e}")

# 3. Solver Settings
st.sidebar.subheader("Solver Parameter Settings")
ph_input = st.sidebar.text_input("pH List (comma separated)", value="13")
v_start = st.sidebar.number_input("Voltage Start (V)", value=0.0, step=0.1)
v_end = st.sidebar.number_input("Voltage End (V)", value=-1.0, step=0.1)
v_step = st.sidebar.number_input("Voltage Step (V)", value=-0.1, step=0.1)

col1, col2 = st.sidebar.columns(2)
with col1:
    temperature = st.sidebar.number_input("Temperature (K)", value=298.0)
with col2:
    time_sim = st.sidebar.number_input("Time (s)", value=100000.0)

with st.sidebar.expander("Advanced Tolerances"):
    abstol = st.sidebar.number_input("Absolute Tolerance", value=1e-20, format="%.2e")
    reltol = st.sidebar.number_input("Relative Tolerance", value=1e-10, format="%.2e")
    pre_exp = st.sidebar.number_input("Pre-exponential Factor", value=6.21e12, format="%.2e")

st.sidebar.subheader("Sweep Mode Settings")
enable_sweep = st.sidebar.checkbox("Enable Sweep Mode", value=True)
sweep_rate = st.sidebar.number_input("Sweep Rate (V/s)", value=0.1)
use_prop = st.sidebar.checkbox("Use Coverage Propagation", value=True)

# 5. Output Settings
output_dir = "results_web"

# --- Helper Functions ---
def parse_float_list(input_str):
    try:
        return [float(x.strip()) for x in input_str.split(",")]
    except ValueError:
        return []

def generate_v_list(start, end, step):
    import numpy as np
    if step == 0: return [start]
    if start > end and step > 0: step = -step
    if start < end and step < 0: step = -step
    arr = np.arange(start, end + step/1000.0, step)
    return [round(x, 4) for x in arr]

# --- Main Logic ---

# Initialize Session State variables
if "simulation_complete" not in st.session_state:
    st.session_state.simulation_complete = False
if "logs" not in st.session_state:
    st.session_state.logs = ""
if "available_species" not in st.session_state:
    st.session_state.available_species = []

run_pressed = st.sidebar.button("Run Simulation", type="primary")

if run_pressed:
    if not uploaded_file:
        st.error("Please upload an input Excel file first.")
    elif not exe_path:
        st.error("Please specify the MKMCXX executable path.")
    else:
        # Reset State
        st.session_state.simulation_complete = False
        st.session_state.logs = ""
        st.session_state.available_species = []
        if 'electron_df' in st.session_state:
            del st.session_state.electron_df
        
        log_container = st.container()
        status_text = st.empty()
        
        with st.spinner("Preparing simulation..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                config = SolverSettings()
                config.pH_list = parse_float_list(ph_input)
                config.V_list = generate_v_list(v_start, v_end, v_step)
                config.temperature = temperature
                config.time = time_sim
                config.abstol = abstol
                config.reltol = reltol
                config.enable_sweep_mode = enable_sweep
                config.sweep_rate = sweep_rate
                config.use_coverage_propagation = use_prop
                config.input_excel_path = tmp_path
                config.executable_path = exe_path
                config.pre_exponential_factor = pre_exp
                
                # Initial default plotting config (Coverage only)
                config.site_density = 2.94e-5 
                config.target_species = [] 
                config.species_electrons = {}
                config.output_base_dir = output_dir
                config.create_plots = True
                
                errors = config.validate()
                if errors:
                    for e in errors: st.error(f"Config Error: {e}")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Error configuring settings: {e}")
                st.stop()

        try:
            if os.path.exists(output_dir): shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            with st.spinner("Running simulations..."):
                app = OptimizedMicrokineticModeling()
                app.config = config
                app.validate_setup()
                
                from data_parser import CachedExcelDataProcessor
                app.excel_processor = CachedExcelDataProcessor(app.config.input_excel_path)
                
                # Extract and Save Species List
                species_df = app.excel_processor._cached_data.get('Input-Output Species')
                if species_df is not None:
                     # Filter non-null species
                     raw_species = species_df['Species'].dropna().tolist()
                     st.session_state.available_species = [str(s) for s in raw_species]
                
                status_box = st.empty()
                progress_bar = st.progress(0)
                total_steps = len(config.pH_list) * len(config.V_list)
                step_tracker = {"current": 0}
                
                def status_update(pH, V):
                    step_tracker["current"] += 1
                    status_box.markdown(f"### 🔄 Running... pH: {pH}, V: {V:.2f}")
                    if total_steps > 0:
                        progress_bar.progress(min(step_tracker["current"] / total_steps, 1.0))
                
                app.run_full_workflow(status_callback=status_update)
                
                status_box.success("Simulation Completed!")
                progress_bar.progress(1.0)
                
                # Persist Context for Post-Processing
                st.session_state.simulation_complete = True
                st.session_state.logs = log_capture_string.getvalue()
                st.session_state.ph_list = config.pH_list
                st.session_state.v_list = config.V_list
                st.session_state.output_base_dir = output_dir
                st.rerun()
                
        except Exception as e:
            st.error(f"Execution Error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.logs = log_capture_string.getvalue()
            
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- Post-Processing & Results ---
if st.session_state.get("simulation_complete"):
    st.success("Simulation Completed Successfully! 🎉")
    
    st.subheader("Results")
    
    # 1. Display Existing Plots (Coverage)
    base_dir = st.session_state.get('output_base_dir', 'results_web')
    result_images = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if (file.startswith("coverage_pH_") or 
                file.startswith("current_density") or 
                file.startswith("selectivity_")) and file.endswith(".png"):
                if file not in ["coverage_line.png", "coverage_bar.png"]:
                    result_images.append(os.path.join(root, file))
    
    if result_images:
        for img_path in result_images:
            st.image(img_path, caption=os.path.basename(img_path))
    else:
        st.warning(f"No plots found initially in {base_dir}")

    # 2. Current Density & Selectivity Analysis
    st.markdown("---")
    st.subheader("⚡ Current Density Analysis")
        
    st.markdown("Calculate current density using:")
    st.latex(r"j = F \times \rho_{site} \times \sum (n_k \times r_k)")
    st.info("where $F$ is Faraday's constant, $\\rho_{site}$ is site density, $n_k$ is electron count, and $r_k$ is reaction rate.")

    with st.expander("Configure Analysis", expanded=True):
        site_density = st.number_input("Site Density (mol/m²)", value=2.94e-5, format="%.2e")
        
        avail_species = st.session_state.get('available_species', [])
        default_targets = [s for s in ['CH3OH', 'CH4', 'CO', 'HCOOH', 'H2', 'CH2O'] if s in avail_species]
        
        target_species = st.multiselect("Target Species for Current Density", avail_species, default=default_targets)
        
        st.write("**Species Electron Counts ($n_k$):**")
        if 'electron_df' not in st.session_state:
            defaults = {'CH3OH': 6, 'CH4': 8, 'CO': 2, 'HCOOH': 2, 'H2': 2, 'CH2O': 4}
            data = [{"Species": s, "Electrons": defaults.get(s, 0)} for s in avail_species]
            st.session_state.electron_df = pd.DataFrame(data)
        
        edited_df = st.data_editor(st.session_state.electron_df, num_rows="fixed", hide_index=True)
        species_electrons = dict(zip(edited_df['Species'], edited_df['Electrons']))
        
        if st.button("🔄 Generate Analysis Plot"):
            with st.spinner("Calculating and Plotting..."):
                try:
                    create_plots(
                        pH_list=st.session_state.ph_list,
                        V_list=st.session_state.v_list,
                        base_directory=st.session_state.output_base_dir,
                        save_plots=True,
                        output_dir=os.path.join(st.session_state.output_base_dir, "plots"),
                        site_density=site_density,
                        target_species=target_species,
                        species_electrons=species_electrons
                    )
                    st.success("Plots updated! Refreshing view...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Plotting Error: {e}")

    # 3. Logs
    with st.expander("Show Execution Logs"):
        st.code(st.session_state.get("logs", ""))

    # 4. Download
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, base_dir)
                zf.write(file_path, arcname)
    
    st.download_button(
        label="📥 Download All Results (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="simulation_results.zip",
        mime="application/zip"
    )

# --- Instructions ---
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Upload your Input Excel file**: Must contain the required sheets (Reactions, Local Environment, etc.).
    2. **Set the Executable Path**: Path to your local `mkmcxx.exe`.
    3. **Configure Settings**: Adjust pH, Voltage range, and other simulation parameters.
    4. **Run Simulation**: Click the button and wait for results.
    5. **Analyze**: Use the post-run section to calculate current density.
    6. **Download**: download the plots and data as a ZIP file.
    """)

with st.expander("📥 Excel Input File Format"):
    st.markdown("""
    This Streamlit app runs electrochemical microkinetic modeling using the MKMCXXS1 package.
    The model input is provided through an Excel file containing three required sheets.
    
    ⚠️ **Column names must match exactly for the file to be read correctly.**

    ---

    ### 🔁 Sheet 1: Reactions
    This sheet defines the reaction network and kinetic parameters.

    **Required Columns:**
    - **Reactions**: Reaction written with reactants and products separated by an arrow (`->` or `→`).
    - **G_f**: Forward activation barrier (J/mol).
    - **G_b**: Backward activation barrier (J/mol).

    **Notes:**
    - Activation barriers (`G_f`, `G_b`) may be numeric values or Excel formulas.
    - Excel formulas can reference the applied potential (`V`) from the Local Environment sheet to introduce potential-dependent kinetics.

    ---

    ### 🌍 Sheet 2: Local Environment
    This sheet specifies the operating conditions of the system.

    **Required Columns:**
    - **Pressure**: System pressure in bar.
    - **pH**: Solution pH (used to compute H⁺ and OH⁻ concentrations).
    - **V**: Applied potential in volts vs RHE (Note: Code expects `V`, not `E_RHE`).

    **Notes:**
    - These values are read automatically and used to generate operating-condition-dependent MKM input files.
    - pH-dependent concentrations of H⁺ and OH⁻ can be defined via Excel formulas in other sheets.

    ---

    ### 🔬 Sheet 3: Input-Output Species
    This sheet lists all species participating in the system.

    **Required Columns:**
    - **Species**: Name of the chemical species.
    - **Input MKMCXX**: Initial concentration or mole fraction (Note: Code expects exact header `Input MKMCXX`).

    **Notes:**
    - These values are used directly as species activities in the rate expressions.
    - Set concentrations to zero for species not initially present.

    ---

    ### ⚙️ Activity Convention
    MKMCXX normally assumes species activities are partial pressures. For electrochemical systems, this app automatically sets `PRESSURE = -1`.
    This enables the use of concentrations or mole fractions (from the Input-Output Species sheet) as activities.

    ### ℹ️ Important Notes
    - Sheet names and column headers are **case-sensitive**.
    - Do not rename or reorder required columns.
    - Modify reaction parameters and conditions directly in Excel for rapid testing.
    """)

# --- Citation ---
with st.expander("📄 Citation"):
    st.info("If you use this application in your work, please cite the following:")
    st.code("""@article{eMKM_2025,
    title={Transient microkinetic modeling of electrochemical reactions: capturing unsteady dynamics of CO reduction and oxygen evolution},
    url={https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adts.202500799},
    DOI={10.1002/adts.202500799},
    journal={Advanced Theory and Simulations},
    author={Chaturvedi, Shivam and Pathak, Amar Deep and Sinha, Nishant and Rajan, Ananth Govind},
    year={2025}, month=nov }""", language="latex")
