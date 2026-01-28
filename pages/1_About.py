
import streamlit as st

st.set_page_config(
    page_title="About - Microkinetic Modeling",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About This Project")

st.markdown("""
# Transient Microkinetic Modeling (USS-MKM)

This application serves as a comprehensive tool for performing unsteady-state microkinetic modeling (USS-MKM) of electrochemical reactions. By incorporating potential sweeping and capturing transient dynamics, it allows for a more realistic modeling of reaction kinetics compared to traditional steady-state approaches.

## Key Features
- **Dynamic Potential Sweeps**: Simulate reactions under varying potential conditions.
- **Unsteady-State Modeling**: Capture transient behaviors and time-dependent surface coverages.
- **Automated Workflow**: From Excel input to final coverage and current density plots.
- **Optimized Performance**: Caches input data to minimize file I/O during parameter sweeps.

## Research & Background
This work enables the study of complex electrochemical systems such as CO reduction and Oxygen Evolution Reaction (OER) by accounting for the dynamic nature of the catalyst surface and reaction environment.

For detailed documentation, methodology, and source code, please visit our GitHub repository:

👉 **[AGR Group - TransientMKM](https://github.com/agrgroup/TransienteMKM)**

---

## Citation
If you use this tool in your research, please cite our work:

**Publication Title**: Transient Microkinetic Modeling of Electrochemical Reactions: Capturing Unsteady Dynamics of CO Reduction and Oxygen Evolution  
**Authors**: Shivam Chaturvedi, Amar Deep Pathak, Nishant Sinha, Ananth Govind Rajan  
**Journal**: Advanced Theory and Simulations  
**Year**: 2025  
**DOI**: [10.1002/adts.202500799](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adts.202500799)

### BibTeX
```latex
@article{shivam_2025,
title={Transient microkinetic modeling of electrochemical reactions: capturing unsteady dynamics of CO reduction and oxygen evolution},
url={https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adts.202500799},
DOI={10.1002/adts.202500799},
journal={Advanced Theory and Simulations},
author={Chaturvedi, Shivam and Pathak, Amar Deep and Sinha, Nishant and Rajan, Ananth Govind},
year={2025}, month=nov }
```
""")
