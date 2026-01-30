import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import plotting utils to read data
try:
    from plotting import CoveragePlotter
except ImportError:
    st.error("Could not import plotting module. Please ensure you are running from the correct directory.")
    st.stop()

st.set_page_config(page_title="Mass Transfer Analysis", page_icon="🌪️", layout="wide")

st.title("🌪️ Mass Transfer Analysis")
st.markdown("""
This tool applies **Levich** and **Koutecký–Levich** corrections to your microkinetic simulation results 
to analyze the impact of mass transfer limitations.
""")

# --- Constants & Defaults ---
F = 96485  # C/mol

# --- Configuration Section ---
st.sidebar.header("Configuration")

# 1. Select Results Directory
default_dir = st.session_state.get('output_base_dir', 'results_web')
base_dir = st.sidebar.text_input("Results Directory", value=default_dir)

# 2. Site Density
default_site_density = 2.94e-5
site_density = st.sidebar.number_input("Site Density (mol/m²)", value=default_site_density, format="%.2e")

@st.cache_data
def load_simulation_data(base_path):
    """Load and cache simulation data structure."""
    if not os.path.exists(base_path):
        return None
    
    data_struct = {} # pH -> V -> {species: rate}
    
    # helper
    plotter = CoveragePlotter(base_path)
    
    # Scan directories
    try:
        ph_dirs = [d for d in os.listdir(base_path) if d.startswith("pH_") and os.path.isdir(os.path.join(base_path, d))]
        for ph_d in ph_dirs:
            ph_val = float(ph_d.split("_")[1])
            ph_path = os.path.join(base_path, ph_d)
            
            data_struct[ph_val] = {}
            
            v_dirs = [d for d in os.listdir(ph_path) if d.startswith("V_") and os.path.isdir(os.path.join(ph_path, d))]
            for v_d in v_dirs:
                try:
                    v_val = float(v_d.split("_")[1])
                    # Read derivatives
                    rates = plotter.read_derivatives_data(ph_val, v_val)
                    if rates:
                        data_struct[ph_val][v_val] = rates
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Error scanning directory: {e}")
        return None
        
    return data_struct

# Load Data
if os.path.exists(base_dir):
    sim_data = load_simulation_data(base_dir)
    if not sim_data:
        st.warning("No valid simulation data found in specified directory.")
        st.stop()
else:
    st.error(f"Directory not found: {base_dir}")
    st.stop()

# Helper: Koutecky-Levich
def apply_koutecky_levich(j_kin_mAcm2, j_lim_mAcm2):
    """Mixing formula: 1/j = 1/j_kin + 1/j_lim"""
    j_kin = np.asarray(j_kin_mAcm2, dtype=float)
    # Avoid div by zero
    eps = 1e-30
    # If j_lim is effectively infinite (0 mass transfer resistance), return j_kin
    if j_lim_mAcm2 > 1e10: return j_kin
    
    return 1.0 / (1.0 / (j_kin + eps) + 1.0 / j_lim_mAcm2)

# Helper: Levich Current
def calc_levich_jlim(n, D, nu, omega_rpm, c_bulk):
    """
    j_lim = 0.62 n F D^(2/3) nu^(-1/6) w^(1/2) c_bulk
    Returns A/m^2
    """
    omega_rad = 2.0 * np.pi * omega_rpm / 60.0
    j_lim = 0.62 * n * F * (D**(2/3)) * (nu**(-1/6)) * (omega_rad**0.5) * c_bulk
    return j_lim

# UI Tabs
tab1, tab2 = st.tabs(["📉 Mass Transfer Limits (Fixed Layer)", "🔄 RDE Analysis (Rotation Speed)"])

# Common Species Setup
avail_species = sorted(list(set([s for ph in sim_data.values() for v in ph.values() for s in v.keys()])))

# --- TAB 1: Fixed Layer ---
with tab1:
    st.subheader("Mass Transfer Limit (Fixed Diffusion Layer)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🧪 Species Parameters")
        target_species_tabs = st.multiselect("Select Species to Plot", avail_species, default=['CH4'] if 'CH4' in avail_species else avail_species[:1], key="tab1_specs")
        
        # Electron counts input
        species_e_map = {}
        for s in target_species_tabs:
            def_e = 2
            if s == 'CH4': def_e = 8
            elif s == 'CH3OH': def_e = 6
            elif s == 'CO': def_e = 2
            elif s == 'H2': def_e = 2
            species_e_map[s] = st.number_input(f"Electrons ($n$) for {s}", value=def_e, key=f"e_{s}")

    with col2:
        st.markdown("##### 🌊 Mass Transfer Parameters")
        st.info("Uses Film Model: $j_{lim} = n F (D / \delta) C_{bulk}$")
        
        limiting_species = st.selectbox("Limiting Reactant (e.g., CO)", ["CO", "CO2", "H+"], index=0)
        n_lim = st.number_input(f"Electrons per {limiting_species} ($n_{{lim}}$)", value=4)
        c_bulk = st.number_input(f"Bulk Concentration ($C_{{bulk}}$) [mol/m³]", value=1.0)
        D_diff = st.number_input(f"Diffusivity ($D$) [m²/s]", value=2.0e-9, format="%.1e")
        delta = st.number_input(f"Boundary Layer Thickness ($\delta$) [m]", value=1.23e-4, format="%.1e")
        
        # Calc limit
        k_m = D_diff / delta
        j_lim_val = n_lim * F * k_m * c_bulk # A/m^2
        j_lim_mA = j_lim_val * 0.1 # mA/cm^2
        st.write(f"**Calculated Limiting Current:** `{j_lim_mA:.2f} mA/cm²`")

    # Plot
    if st.button("Generate Fixed Layer Plot", type="primary"):
        ph_vals = sorted(sim_data.keys())
        
        # Setup Figure
        fig, ax = plt.subplots(figsize=(9, 6))
        
        colors = plt.cm.jet(np.linspace(0, 1, len(ph_vals)))
        markers = ['o', 's', 'D', '^', 'v', 'x']
        
        mtl_threshold = 0.9
        
        for i, ph in enumerate(ph_vals):
            ph_data = sim_data[ph]
            # Sort by V
            sorted_v = sorted(ph_data.keys())
            
            for j, spec in enumerate(target_species_tabs):
                n_e = species_e_map[spec]
                
                v_list = []
                j_eff_list = []
                
                for v in sorted_v:
                    rate = ph_data[v].get(spec, 0.0)
                    if rate <= 0: continue
                    
                    # Kinetic Current
                    j_kin = rate * site_density * n_e * F * 0.1 # mA/cm2
                    
                    # Apply KL
                    j_eff = apply_koutecky_levich(j_kin, j_lim_mA)
                    
                    if j_eff > 1e-10:
                        v_list.append(v)
                        j_eff_list.append(j_eff)
                
                if v_list:
                    label = f"{spec}, pH {ph}"
                    ax.plot(v_list, j_eff_list, 
                           label=label, 
                           color=colors[i], 
                           marker=markers[j % len(markers)],
                           linestyle='-' if j==0 else '--',
                           linewidth=2)

        # Plot Limit Line
        ax.axhline(j_lim_mA, color='k', linestyle=':', label=f'{limiting_species} Limit')
        
        ax.set_xlabel('Potential U vs RHE (V)', fontsize=14, fontweight='bold')
        ax.set_ylabel('j (mA/cm²)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


# --- TAB 2: RDE Analysis ---
with tab2:
    st.subheader("RDE Analysis (Variable Rotation Speed)")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("##### 🧪 Species Selection")
        # Single species selection typically for RDE
        target_spec_rde = st.selectbox("Select Target Product", avail_species, index=0 if avail_species else None)
        n_e_rde = st.number_input(f"Electrons for {target_spec_rde}", value=2, key="e_rde")
        
        st.markdown("##### ⚙️ RDE Parameters")
        rpm_str = st.text_input("Rotation Speeds (RPM, comma separated)", value="400, 900, 1600, 2500")
        try:
            rpm_list = [float(x.strip()) for x in rpm_str.split(",")]
        except:
            rpm_list = [1600]
            st.error("Invalid RPM list")
            
    with col_r2:
        st.markdown("##### 🌊 Fluid Parameters")
        st.info("Uses Levich Equation")
        nu_visc = st.number_input("Kinematic Viscosity ($\\nu$) [m²/s]", value=1.0e-6, format="%.2e")
        c_bulk_rde = st.number_input("Reactant Bulk Conc ($C_{{bulk}}$) [mol/m³]", value=1.0, key="c_rde")
        D_diff_rde = st.number_input("Reactant Diffusivity ($D$) [m²/s]", value=2.0e-9, format="%.1e", key="d_rde")
        n_lim_rde = st.number_input("Reactant Electrons ($n_{{lim}}$)", value=4, key="n_lim_rde")

    # Filter pH
    st.markdown("##### 🔍 Filter Data")
    all_phs = sorted(sim_data.keys())
    selected_ph = st.selectbox("Select pH to Analyze", all_phs, index=0 if all_phs else None)
    
    if st.button("Generate RDE Plot", type="primary"):
        if selected_ph is None:
            st.error("No pH selected")
            st.stop()
            
        ph_data = sim_data[selected_ph]
        sorted_v = sorted(ph_data.keys())
        
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Colors for RPM
        rpm_colors = plt.cm.viridis(np.linspace(0, 1, len(rpm_list)))
        
        has_plot = False
        
        for i, rpm in enumerate(rpm_list):
            # Calc J_lim for this RPM
            j_lim_A = calc_levich_jlim(n_lim_rde, D_diff_rde, nu_visc, rpm, c_bulk_rde)
            j_lim_mA = j_lim_A * 0.1
            
            v_plot = []
            j_plot = []
            mtl_points = [] # (v, j)
            
            for v in sorted_v:
                rate = ph_data[v].get(target_spec_rde, 0.0)
                if rate <= 0: continue
                
                j_kin = rate * site_density * n_e_rde * F * 0.1
                j_eff = apply_koutecky_levich(j_kin, j_lim_mA)
                
                if j_eff > 1e-10:
                    v_plot.append(v)
                    j_plot.append(j_eff)
                    
                    # Check limit
                    if j_eff / j_lim_mA >= 0.9:
                        mtl_points.append((v, j_eff))
            
            if v_plot:
                has_plot = True
                label = f"{rpm} RPM"
                ax.plot(v_plot, j_plot, label=label, color=rpm_colors[i], linewidth=2)
                
                # Highlight MTL points
                if mtl_points:
                    mx, my = zip(*mtl_points)
                    ax.plot(mx, my, 'o', markerfacecolor='none', markeredgecolor=rpm_colors[i], markersize=8)

        ax.set_xlabel('Potential U vs RHE (V)', fontsize=14, fontweight='bold')
        ax.set_ylabel('j (mA/cm²)', fontsize=14, fontweight='bold')
        ax.set_title(f"RDE Voltammograms ({target_spec_rde}, pH {selected_ph})", fontsize=16)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if has_plot:
            st.pyplot(fig)
        else:
            st.warning("No non-zero current data found for this species.")
