
import streamlit as st
import pandas as pd
import os
import shutil
import tempfile
from pathlib import Path
import logging
import zipfile
import io

# Import project modules
# We need to make sure the current directory is in sys.path
import sys
sys.path.append(os.getcwd())

from workflow import OptimizedMicrokineticModeling
from config import SolverSettings

# Configure logging to capture output for Streamlit
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

# Custom CSS for premium feel
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    /* Ensure input text is visible (Streamlit defaults can be dark on dark) */
    .stTextInput>div>div>input {
        color: #ffffff;
        background-color: #262730; 
    }
    .stNumberInput>div>div>input {
        color: #ffffff;
    }
    /* Fix for multiselect text color */
    span[data-baseweb="tag"] {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚛️ Microkinetic Modeling Application")
st.markdown("""
**Here, we perform unsteady-state MKM (USS-MKM) with and without potential sweeping to capture transient dynamics and realistically model reaction kinetics.**
""")
st.markdown("Run optimized microkinetic simulations based on your Excel input and configuration.")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

# 1. File Upload
uploaded_file = st.sidebar.file_uploader("Upload Input Excel", type=["xlsx", "xls"])

# 2. Executable Path (Dynamic handling for Cloud Deployment)
import platform
import stat

if platform.system() == "Windows":
    # Local Windows Path
    exe_path = "D:/mkmcxx/mkmcxx-2.15.3-windows-x64/mkmcxx_2.15.3/bin/mkmcxx.exe"
else:
    # Linux (Streamlit Cloud) Path - Assumes binary 'mkmcxx' is in root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(base_dir, "mkmcxx")
    
    # Ensure it's executable
    try:
        # DEBUG: Check if file exists
        if not os.path.exists(exe_path):
            st.error(f"⚠️ Binary not found at: {exe_path}")
            # st.write(f"Current Directory: `{os.getcwd()}`") # Clean up debug
        else:
            # Force executable permissions (sometimes Git loses them)
            st_mode = os.stat(exe_path).st_mode
            os.chmod(exe_path, st_mode | stat.S_IEXEC)
            
            # Verify permissions silently
            new_mode = os.stat(exe_path).st_mode
            if not (new_mode & stat.S_IEXEC):
                logger.warning(f"Failed to set executable bit! Mode: {oct(new_mode)}")
            else:
                pass # st.success(f"Found binary at `{exe_path}` and set executable permissions.")
                
    except Exception as e:
        logger.warning(f"Could not set executable permissions: {e}")

# exe_path = st.sidebar.text_input("MKMCXX Executable Path", value=default_exe_path)

# 3. Solver Settings
st.sidebar.subheader("Solver Parameter Settings")

# pH Range
ph_input = st.sidebar.text_input("pH List (comma separated)", value="13")

# Voltage Range
v_start = st.sidebar.number_input("Voltage Start (V)", value=0.0, step=0.1)
v_end = st.sidebar.number_input("Voltage End (V)", value=-1.0, step=0.1)
v_step = st.sidebar.number_input("Voltage Step (V)", value=-0.1, step=0.1)

# Temp and Time
col1, col2 = st.sidebar.columns(2)
with col1:
    temperature = st.sidebar.number_input("Temperature (K)", value=298.0)
with col2:
    time_sim = st.sidebar.number_input("Time (s)", value=100000.0)

# Tolerances
with st.sidebar.expander("Advanced Tolerances"):
    abstol = st.sidebar.number_input("Absolute Tolerance", value=1e-20, format="%.2e")
    reltol = st.sidebar.number_input("Relative Tolerance", value=1e-10, format="%.2e")
    pre_exp = st.sidebar.number_input("Pre-exponential Factor", value=6.21e12, format="%.2e")

# Sweep Mode
st.sidebar.subheader("Sweep Mode Settings")
enable_sweep = st.sidebar.checkbox("Enable Sweep Mode", value=True)
sweep_rate = st.sidebar.number_input("Sweep Rate (V/s)", value=0.1)
use_prop = st.sidebar.checkbox("Use Coverage Propagation", value=True)

# 4. Plotting Configuration
with st.sidebar.expander("Plotting Configuration"):
    site_density = st.sidebar.number_input("Site Density (mol/m²)", value=2.94e-5, format="%.2e")
    
    default_species = ['CH3OH', 'CH4', 'CO', 'HCOOH', 'H2', 'CH2O']
    target_species_sel = st.sidebar.multiselect("Target Species", 
                                              ['CH3OH', 'CH4', 'CO', 'HCOOH', 'H2', 'CH2O', 'C2H4', 'C2H5OH'],
                                              default=default_species)
    
    # Simple JSON input for electrons map
    import json
    default_electrons = {
        'CH3OH': 6, 'CH4': 8, 'CO': 2, 'HCOOH': 2, 'H2': 2, 'CH2O': 4
    }
    species_e_str = st.sidebar.text_area("Species Electron Count (JSON)", 
                                       value=json.dumps(default_electrons, indent=2),
                                       height=150)

# 5. Output Settings
output_dir = "results_web"

# --- Main Logic ---

def parse_float_list(input_str):
    try:
        return [float(x.strip()) for x in input_str.split(",")]
    except ValueError:
        return []

def generate_v_list(start, end, step):
    # Handle floating point range generation
    import numpy as np
    # Determine direction
    if step == 0: return [start]
    
    # Ensure step has correct sign
    if start > end and step > 0: step = -step
    if start < end and step < 0: step = -step
    
    # Create range (inclusive of end if possible)
    arr = np.arange(start, end + step/1000.0, step) # small epsilon to include end
    # Round to avoid fp errors
    return [round(x, 4) for x in arr]

run_pressed = st.sidebar.button("Run Simulation", type="primary")

if run_pressed:
    if not uploaded_file:
        st.error("Please upload an input Excel file first.")
    elif not exe_path:
        st.error("Please specify the MKMCXX executable path.")
    else:
        # Create a placeholder for logs
        log_container = st.container()
        status_text = st.empty()
        
        with st.spinner("Preparing simulation..."):
            # 1. Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # 2. Construct Configuration
            try:
                config = SolverSettings()
                config.pH_list = parse_float_list(ph_input)
                
                # Generate V list from range inputs
                # Or could allow custom list input. For now, let's use the range generator
                # But wait, original config had a list. 
                # Let's offer a choice? For simplicty, let's use the range logic
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
                
                # Plotting config
                config.site_density = site_density
                config.target_species = target_species_sel
                try:
                    config.species_electrons = json.loads(species_e_str)
                except Exception as e:
                    st.error(f"Invalid JSON for Species Electron Count: {e}")
                    st.stop()
                
                config.output_base_dir = output_dir
                config.create_plots = True # Always create plots for web
                
                # Validate config manually to show errors nicely
                errors = config.validate()
                if errors:
                    for e in errors:
                        st.error(f"Config Error: {e}")
                    st.stop()
                
            except Exception as e:
                st.error(f"Error configuring settings: {e}")
                st.stop()

        # 3. Run Application
        try:
            # Clean output dir
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            with st.spinner("Running simulations... This may take a while."):
                # Capture stdout/logs
                import io
                from contextlib import redirect_stdout
                
                # We need to re-initialize logging to capture it if we want to show it live
                # But file logging is safer.
                
                app = OptimizedMicrokineticModeling()
                # Inject the config manually since we didn't load from file
                app.config = config
                # We need to re-validate setup since we bypassed init with path
                app.validate_setup()
                
                # Re-initialize the excel processor with the new temp path
                from data_parser import CachedExcelDataProcessor
                app.excel_processor = CachedExcelDataProcessor(app.config.input_excel_path)
                
                # Define status update callback
                status_box = st.empty()
                progress_bar = st.progress(0)
                
                # Calculate total steps for progress bar
                total_steps = len(config.pH_list) * len(config.V_list)
                # Use a mutable container for step tracking
                step_tracker = {"current": 0}
                
                def status_update(pH, V):
                    step_tracker["current"] += 1
                    status_box.markdown(f"""
                    ### 🔄 Running Simulation...
                    - **pH**: {pH}
                    - **Potential**: {V:.2f} V
                    """)
                    if total_steps > 0:
                        progress_bar.progress(min(step_tracker["current"] / total_steps, 1.0))
                
                # Run Workflow with callback
                app.run_full_workflow(status_callback=status_update)
                
                # Clear status
                status_box.success("Simulation & Plotting Completed!")
                progress_bar.progress(1.0)
                
            st.success("Simulation Completed Successfully! 🎉")
            
            # 4. Display Results
            st.subheader("Results")
            
            # List images in output directory - ONLY COVERAGE PLOTS
            # Modified search logic to look for coverage_pH_*.png
            # These are now saved in output_dir/plots (as per our modification in main_app)
            result_images = []
            
            # Search logic
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    # Match pattern coverage_pH_*.png, current_density.png, selectivity_pH_*.png
                    # Explicitly exclude coverage_line and coverage_bar if they exist
                    if (file.startswith("coverage_pH_") or 
                        file.startswith("current_density") or 
                        file.startswith("selectivity_")) and file.endswith(".png"):
                        
                        if file not in ["coverage_line.png", "coverage_bar.png"]:
                            result_images.append(os.path.join(root, file))
            
            if result_images:
                # Use full width for better visibility since we're filtering
                for img_path in result_images:
                    st.image(img_path, caption=os.path.basename(img_path))
            else:
                st.warning(f"No plot images found in {output_dir}. Check logs.")

            # 6. Show Logs
            with st.expander("Show Execution Logs"):
                log_contents = log_capture_string.getvalue()
                st.code(log_contents)

            # 5. Zip and Download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zf.write(file_path, arcname)
            
            st.download_button(
                label="📥 Download All Results (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="simulation_results.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
            import traceback
            st.code(traceback.format_exc())
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# --- Instructions ---
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Upload your Input Excel file**: Must contain the required sheets (Reactions, Local Environment, etc.).
    2. **Configure Settings**: Adjust pH, Voltage range, and other simulation parameters.
    3. **Run Simulation**: Click the button and wait for results.
    4. **Download**: download the plots and data as a ZIP file.
    """)

with st.expander("📥 Excel Input File Format"):
    st.markdown("""
    This Streamlit app runs electrochemical microkinetic modeling using the MKMCXX package.
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
#st.markdown("---")
with st.expander("📄 Citation"):
    st.info("If you use this application in your work, please cite the following:")
    st.code("""@article{eMKM_2025,
    title={Transient microkinetic modeling of electrochemical reactions: capturing unsteady dynamics of CO reduction and oxygen evolution},
    url={https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adts.202500799},
    DOI={10.1002/adts.202500799},
    journal={Advanced Theory and Simulations},
    author={Chaturvedi, Shivam and Pathak, Amar Deep and Sinha, Nishant and Rajan, Ananth Govind},
    year={2025}, month=nov }""", language="latex")

