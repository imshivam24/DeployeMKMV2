# Transient eMKM Web Application

This directory contains the Streamlit-based web application for the Transient eMKM Solver. It provides a graphical user interface (GUI) for users to configure, run, and visualize electrochemical microkinetic simulations without needing to use the command line.

## 🔗 Live Application

**Access the running application here:**  
👉 **[https://transient-emkm.streamlit.app/](https://transient-emkm.streamlit.app/)**

---

## 📂 Structure

- **`Transient_eMKM.py`**: The main Streamlit application entry point.
- **`pages/`**: Contains additional pages for the multi-page Streamlit app.
- **`simulation.py`**: Core simulation logic (optimized runner).
- **`data_parser.py`**: Handles Excel file parsing and caching.
- **`plotting.py`**: Utilities for generating plots (Coverage, Current Density, etc.).
- **`mkmcxx/`**: (Optional) Directory containing the MKMCXX executable binaries for local execution.

## 🚀 Running Locally

If you wish to run the web interface on your local machine:

1.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run with Streamlit**
    ```bash
    streamlit run Transient_eMKM.py
    ```

3.  **Local Configuration**
    - The app will attempt to locate `mkmcxx` in the local directory or at the path specified in the sidebar.
    - Ensure your input Excel file follows the required format (Reactions, Local Environment, Input-Output Species).

## 📊 Features

*   **Drag-and-Drop Input**: Upload your Excel configuration file directly.
*   **Interactive Config**: Set pH ranges, Voltage sweeps, and Tolerances via sliders and inputs.
*   **Real-time Progress**: View simulation status and progress bars.
*   **Visualization Dashboard**:
    *   Interactive Coverage Plots
    *   Current Density Calculations ($j = F \cdot \rho \cdot n \cdot r$)
    *   Selectivity Analysis
*   **One-Click Download**: Download all results, inputs, and plots as a ZIP file.
