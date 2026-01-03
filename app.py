import streamlit as st
import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import imageio.v3 as iio
import os
import pandas as pd
import tempfile
import shutil
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec

# ---------------- STREAMLIT PAGE CONFIG ----------------
st.set_page_config(
    page_title="MD Pore Forensics Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- VISUAL STYLE CONSTANTS ----------------
plt.style.use('seaborn-v0_8-white')
LIPID_STYLES = {
    'POPEE': {'c': '#e7298a', 'm': 'o'}, 'POPC': {'c': '#4daf4a', 'm': 's'},
    'POGL': {'c': '#e6ab02', 'm': '^'}, 'POPE': {'c': '#377eb8', 'm': 'v'},
    'NSM': {'c': '#ff7f00', 'm': 'D'}, 'POPI': {'c': '#984ea3', 'm': 'P'},
    'CHL1': {'c': '#a6cee3', 'm': '*'}, 'POPS': {'c': '#f781bf', 'm': 'p'},
    'POPCE': {'c': '#a65628', 'm': 'X'}, 'POPG': {'c': '#fdbf6f', 'm': 'h'},
    'POPA': {'c': '#117A65', 'm': '<'}, 'PIP2': {'c': '#d95f02', 'm': 'H'}
}
LIPID_ATOMS = {
    'POPEE': 'P', 'POPC': 'P', 'POGL': 'O11', 'POPE': 'P', 'NSM': 'P',
    'POPI': 'P', 'CHL1': 'O3', 'POPS': 'P', 'POPCE': 'P', 'POPG': 'P',
    'POPA': 'P', 'PIP2': 'P1'
}

# ---------------- HELPER FUNCTIONS ----------------
def draw_stylish_table(ax, title, data_dict, y_start, theme_color='#333333', is_history=False):
    """Draws a custom table on a Matplotlib axis."""
    header_height = 0.05
    rect = Rectangle((0.02, y_start - header_height), 0.96, header_height,
                     facecolor=theme_color, edgecolor='none', transform=ax.transAxes, alpha=0.9, zorder=1)
    ax.add_patch(rect)
    ax.text(0.05, y_start - 0.035, title.upper(), color='white', weight='bold', fontsize=10, transform=ax.transAxes, zorder=2, va='center')

    y_row = y_start - 0.08
    ax.text(0.05, y_row, "LIPID", weight='bold', fontsize=9, color='#555', transform=ax.transAxes)
    ax.text(0.60, y_row, "COUNT", weight='bold', fontsize=9, color='#555', ha='right', transform=ax.transAxes)
    ax.text(0.92, y_row, "%", weight='bold', fontsize=9, color='#555', ha='right', transform=ax.transAxes)
    y_row -= 0.04

    total = sum(data_dict.values())
    if total == 0:
        ax.text(0.5, y_row - 0.02, "No Data Logged", ha='center', color='#999', style='italic', fontsize=9, transform=ax.transAxes)
        return y_row - 0.1

    sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    limit = 8 if is_history else 5

    for i, (k, v) in enumerate(sorted_items[:limit]):
        row_h = 0.035
        bg_color = 'white' if i % 2 == 0 else '#E6E6E6'
        bg_rect = Rectangle((0.02, y_row - row_h + 0.01), 0.96, row_h, facecolor=bg_color, transform=ax.transAxes, zorder=0)
        ax.add_patch(bg_rect)
        pct = (v / total) * 100
        ax.text(0.05, y_row, k, fontsize=9, fontweight='bold', color='#333', transform=ax.transAxes, va='bottom')
        ax.text(0.60, y_row, f"{v}", fontsize=9, ha='right', fontfamily='monospace', transform=ax.transAxes, va='bottom')
        ax.text(0.92, y_row, f"{pct:.1f}", fontsize=9, ha='right', fontfamily='monospace', transform=ax.transAxes, va='bottom')
        y_row -= 0.04
    return y_row

def run_processing(u, params, temp_dir):
    """Main Analysis Loop"""
    
    # Unpack Parameters
    grid_res = params['grid_res']
    z_min = params['z_min']
    z_max = params['z_max']
    pore_thresh = params['pore_thresh']
    prepore_low = params['prepore_low']
    prepore_high = params['prepore_high']
    rupture_dens = params['rupture_dens']
    stride = params['stride']
    
    # Init Data
    global_pore_history = {}
    global_prepore_history = {}
    pore_history_log = []
    dataset_records = []
    frame_files = []
    RUPTURE_FLAG = False
    
    # Pre-calc constants
    box_x_A, box_y_A = u.dimensions[0], u.dimensions[1]
    dx, dy = box_x_A / grid_res, box_y_A / grid_res
    bin_vol = dx * dy * (z_max - z_min)
    CONV_FACTOR = ((18.01528 / 1000) / 6.022e23) * 1e30
    FORENSIC_RADIUS_NM = 1.2
    
    legend_handles = [Line2D([0], [0], marker=s['m'], color='w', markerfacecolor=s['c'], 
                             markeredgecolor='black', markersize=9, label=res) 
                      for res, s in LIPID_STYLES.items()]

    # UI Progress Elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = len(u.trajectory[::stride])
    
    for i, ts in enumerate(u.trajectory[::stride]):
        # Update UI
        progress_bar.progress((i + 1) / total_frames)
        status_text.text(f"Processing Frame {ts.frame} ({ts.time:.0f} ps)...")
        
        current_box_nm = u.dimensions[:2] / 10
        
        # Figure Setup
        fig = plt.figure(figsize=(26, 12))
        gs = GridSpec(1, 4, width_ratios=[0.8, 4.2, 2.0, 2.0], wspace=0.15, hspace=0, figure=fig)
        ax_key, ax_map, ax_pore, ax_prepore = [fig.add_subplot(gs[i]) for i in range(4)]
        
        for ax in [ax_key, ax_pore, ax_prepore]: ax.axis('off')
        
        # Map Styling
        ax_map.set_xticks([]); ax_map.set_yticks([])
        for spine in ax_map.spines.values(): spine.set_edgecolor('#444444'); spine.set_linewidth(2)
        ax_map.set_aspect('equal')

        # --- PHYSICS ---
        sel_water = f"resname TIP3 and name OH2 and prop z > {z_min} and prop z < {z_max}"
        water = u.select_atoms(sel_water)
        
        density_smooth = np.zeros((grid_res, grid_res))
        active_pore_centers = []
        active_prepore_centers = []
        max_core_density = 0.0
        current_pore_area_pixels = 0
        
        if len(water) > 0:
            c_map, _, _ = np.histogram2d(water.positions[:, 0], water.positions[:, 1], 
                                         bins=grid_res, range=[[0, box_x_A], [0, box_y_A]])
            d_map = (c_map / bin_vol) * CONV_FACTOR
            density_smooth = ndimage.gaussian_filter(d_map, sigma=1.5)
            max_core_density = np.max(density_smooth)
            
            if max_core_density >= rupture_dens: RUPTURE_FLAG = True
            
            # Pore Detect
            pore_bin = density_smooth >= pore_thresh
            lbl, n_f = ndimage.label(pore_bin)
            if n_f > 0:
                sizes = ndimage.sum(pore_bin, lbl, range(1, n_f+1))
                valid = np.where(sizes >= 8)[0] + 1
                if len(valid) > 0:
                    current_pore_area_pixels = np.sum(sizes[valid-1])
                    coms = np.atleast_2d(ndimage.center_of_mass(pore_bin, lbl, valid))
                    active_pore_centers = [np.array([(c[0]*dx)/10, (c[1]*dy)/10]) for c in coms]
            
            # Pre-Pore Detect
            if not RUPTURE_FLAG:
                for h in pore_history_log:
                    p_c = h['center']
                    gx, gy = int(np.clip((p_c[0]*10)/dx, 0, grid_res-1)), int(np.clip((p_c[1]*10)/dy, 0, grid_res-1))
                    val = density_smooth[gx, gy]
                    is_pore = any(np.linalg.norm(p_c - p) < FORENSIC_RADIUS_NM for p in active_pore_centers)
                    if not is_pore and (prepore_low <= val < prepore_high):
                        active_prepore_centers.append(p_c)

        # --- LIPIDS ---
        curr_pore_lipids = {}
        curr_prepore_lipids = {}
        
        for resname, atom_name in LIPID_ATOMS.items():
            atoms = u.select_atoms(f"resname {resname} and name {atom_name} and prop z > 55")
            if len(atoms) > 0:
                pos_nm = atoms.positions[:, :2] / 10
                s = LIPID_STYLES.get(resname)
                
                # Arrays for plotting
                sizes = np.full(len(atoms), 40); colors = np.full(len(atoms), s['c'])
                edges = np.full(len(atoms), 'white'); lws = np.full(len(atoms), 0.5)
                alphas = np.full(len(atoms), 0.6)
                
                # Pore Check
                for p in active_pore_centers:
                    delta = pos_nm - p
                    delta -= np.round(delta / current_box_nm) * current_box_nm
                    mask = np.linalg.norm(delta, axis=1) < FORENSIC_RADIUS_NM
                    if np.any(mask):
                        count = np.sum(mask)
                        curr_pore_lipids[resname] = curr_pore_lipids.get(resname, 0) + count
                        if not RUPTURE_FLAG: global_pore_history[resname] = global_pore_history.get(resname, 0) + count
                        sizes[mask] = 140; edges[mask] = '#CC0000'; lws[mask] = 2.5; alphas[mask] = 1.0
                
                # Pre-Pore Check
                if not RUPTURE_FLAG:
                    for p in active_prepore_centers:
                        delta = pos_nm - p
                        delta -= np.round(delta / current_box_nm) * current_box_nm
                        mask = np.linalg.norm(delta, axis=1) < FORENSIC_RADIUS_NM
                        if np.any(mask):
                            count = np.sum(mask)
                            curr_prepore_lipids[resname] = curr_prepore_lipids.get(resname, 0) + count
                            if not RUPTURE_FLAG: global_prepore_history[resname] = global_prepore_history.get(resname, 0) + count
                            sizes[mask] = 120; edges[mask] = '#00AA00'; lws[mask] = 2.5; alphas[mask] = 1.0

                ax_map.scatter(pos_nm[:,0], pos_nm[:,1], c=colors, marker=s['m'], s=sizes, 
                               linewidths=lws, edgecolors=edges, alpha=alphas)

        # --- CSV LOGGING ---
        pixel_area_nm2 = (dx/10) * (dy/10)
        final_area = current_pore_area_pixels * pixel_area_nm2
        
        row_label = 0
        row_state = "Stable"
        active_lipids = {}
        
        if RUPTURE_FLAG:
            row_label, row_state, active_lipids = 3, "Ruptured", curr_pore_lipids
        elif active_pore_centers:
            row_label, row_state, active_lipids = 2, "Pore", curr_pore_lipids
        elif active_prepore_centers:
            row_label, row_state, active_lipids = 1, "Pre-Pore", curr_prepore_lipids
            
        record = {
            'Frame': ts.frame, 'Time_ps': ts.time, 'Label': row_label, 
            'State': row_state, 'Local_Density': max_core_density, 
            'Pore_Area_nm2': final_area
        }
        for lip in LIPID_STYLES.keys(): record[lip] = active_lipids.get(lip, 0)
        dataset_records.append(record)
        
        # --- HISTORY LOGIC ---
        if not RUPTURE_FLAG:
            all_c = [(c, 'Pore') for c in active_pore_centers] + [(c, 'Pre-Pore') for c in active_prepore_centers]
            for c, t in all_c:
                known = False
                for h in pore_history_log:
                    if np.linalg.norm(c - h['center']) < FORENSIC_RADIUS_NM:
                        h['type'] = t; known = True; break
                if not known and len(pore_history_log) < 5: pore_history_log.append({'center': c, 'type': t})

        # --- PLOTTING FINALIZATION ---
        ax_map.imshow(density_smooth.T, cmap='Blues', origin='lower', extent=[0, box_x_A/10, 0, box_y_A/10], 
                      vmin=0, vmax=1000, alpha=0.9)
        ax_map.text(0.02, 0.98, f"TIME: {ts.time:.0f} ps", transform=ax_map.transAxes, 
                    fontsize=14, fontweight='bold', color='white', bbox=dict(facecolor='black', alpha=0.7))
        
        if not RUPTURE_FLAG:
            for p in active_prepore_centers: ax_map.add_patch(Circle(p, FORENSIC_RADIUS_NM, fill=False, edgecolor='#00FF00', lw=2))
            for p in active_pore_centers: ax_map.add_patch(Circle(p, FORENSIC_RADIUS_NM, fill=False, edgecolor='red', lw=2))
            
        # Panels
        ax_key.text(0.5, 0.88, f"{max_core_density:.0f}", ha='center', fontsize=20, fontweight='bold', 
                    color='red' if RUPTURE_FLAG else '#0066CC', transform=ax_key.transAxes)
        ax_key.legend(handles=legend_handles, loc='center', title="LIPID KEY", fontsize=8)
        
        draw_stylish_table(ax_pore, "Live Composition", curr_pore_lipids, 0.82, theme_color='#D32F2F')
        draw_stylish_table(ax_pore, "History", global_pore_history, 0.45, theme_color='#8B0000', is_history=True)
        draw_stylish_table(ax_prepore, "Live Composition", curr_prepore_lipids, 0.82, theme_color='#388E3C')
        draw_stylish_table(ax_prepore, "History", global_prepore_history, 0.45, theme_color='#1B5E20', is_history=True)
        
        fname = os.path.join(temp_dir, f"frame_{i:05d}.png")
        plt.savefig(fname, dpi=80, facecolor='#F0F2F5', bbox_inches='tight')
        plt.close(fig)
        frame_files.append(fname)
        
    return pd.DataFrame(dataset_records), frame_files

# ---------------- MAIN APP LAYOUT ----------------

# TITLE
st.title("🧬 Forensic MD Dashboard: Pore & Lipid Analytics")
st.markdown("---")

# SIDEBAR CONTROLS
with st.sidebar:
    st.header("1. Upload Data")
    gro_file = st.file_uploader("Structure File (.gro)", type=['gro'])
    xtc_file = st.file_uploader("Trajectory File (.xtc)", type=['xtc'])
    
    st.header("2. Physics Parameters")
    stride = st.slider("Frame Stride (Skip Frames)", 1, 100, 1, help="Higher = Faster Preview")
    grid_res = st.slider("Grid Resolution", 50, 200, 120)
    
    with st.expander("Advanced Thresholds"):
        pore_thresh = st.number_input("Pore Threshold (kg/m³)", value=450)
        prepore_low = st.number_input("Pre-Pore Low (kg/m³)", value=200)
        prepore_high = st.number_input("Pre-Pore High (kg/m³)", value=440)
        rupture_dens = st.number_input("Rupture Density (kg/m³)", value=990)
        z_min = st.number_input("Z Min (Å)", value=40.0)
        z_max = st.number_input("Z Max (Å)", value=70.0)

    start_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# MAIN EXECUTION
if start_btn and gro_file and xtc_file:
    
    # SETUP TEMP FILES
    t_dir = tempfile.mkdtemp()
    gro_path = os.path.join(t_dir, "input.gro")
    xtc_path = os.path.join(t_dir, "input.xtc")
    
    with open(gro_path, "wb") as f: f.write(gro_file.getbuffer())
    with open(xtc_path, "wb") as f: f.write(xtc_file.getbuffer())
    
    try:
        u = mda.Universe(gro_path, xtc_path)
        params = {
            'grid_res': grid_res, 'z_min': z_min, 'z_max': z_max,
            'pore_thresh': pore_thresh, 'prepore_low': prepore_low,
            'prepore_high': prepore_high, 'rupture_dens': rupture_dens,
            'stride': stride
        }
        
        # RUN ANALYSIS
        with st.spinner("Processing Trajectory... Please Wait."):
            df, frames = run_processing(u, params, t_dir)
            
            # GENERATE GIF
            gif_path = os.path.join(t_dir, "dashboard.gif")
            images = [iio.imread(f) for f in frames]
            iio.imwrite(gif_path, images, duration=150, loop=0)

        # RESULTS TABS
        st.success("Analysis Complete!")
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard View", "📈 Data Analytics", "📥 Downloads"])
        
        with tab1:
            st.image(gif_path, caption="Generated Dashboard Animation", use_container_width=True)
            
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Pore Area Evolution")
                st.line_chart(df, x="Time_ps", y="Pore_Area_nm2")
            with col2:
                st.subheader("State Classification")
                state_counts = df['State'].value_counts()
                st.bar_chart(state_counts)
                
            st.subheader("Local Density vs Time")
            st.area_chart(df, x="Time_ps", y="Local_Density", color="#FF5555")

        with tab3:
            colA, colB = st.columns(2)
            
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            colA.download_button(
                "📄 Download Dataset (CSV)",
                csv,
                "pore_analysis.csv",
                "text/csv",
                key='download-csv'
            )
            
            # GIF Download
            with open(gif_path, "rb") as file:
                btn = colB.download_button(
                    label="🎬 Download Animation (GIF)",
                    data=file,
                    file_name="dashboard_render.gif",
                    mime="image/gif"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    finally:
        # Cleanup temp directory logic could go here, 
        # but Streamlit cleans up session temp files eventually.
        pass

elif start_btn:
    st.warning("⚠️ Please upload both GRO and XTC files.")
else:
    st.info("👈 Upload your files in the sidebar to begin.")
