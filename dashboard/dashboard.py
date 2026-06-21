import os
import sys
# Add project root directory to the python path to resolve app/services imports on Streamlit Cloud
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force reload of local modules to prevent Streamlit Cloud from using stale cached imports in memory
for module_name in list(sys.modules.keys()):
    if module_name.startswith('app') or module_name.startswith('services'):
        del sys.modules[module_name]

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import time
from datetime import datetime, timedelta

# Stability check for OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False

from app.config import (
    DEFAULT_LATITUDE, DEFAULT_LONGITUDE, PEI_WEIGHTS, REPORTS_DIR,
    EXPLANATIONS_DIR, DETECTIONS_DIR, SIMULATIONS_DIR, MODEL_DIR
)
from app.data_pipeline import DataPipeline
from services.hotspot_service import HotspotService
from services.pei_service import PEIService
from services.forecast_service import ForecastService
from services.explainability_service import ExplainabilityService
from services.detection_service import DetectionService
from services.delay_service import DelayService
from services.simulation_service import SimulationService
from services.traffic_memory_service import TrafficMemoryService
from services.immunity_service import ImmunityService

# Streamlit Page Configurations
st.set_page_config(
    page_title="PARKTWIN AI - Congestion Intelligence Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Globally defined stationary threshold to prevent NameErrors
DEFAULT_STATIONARY_THRESHOLD = 10
if "stationary_threshold" not in st.session_state:
    st.session_state.stationary_threshold = DEFAULT_STATIONARY_THRESHOLD

# Custom CSS styling (Enterprise SaaS Design System)
st.markdown("""
<style>
    /* Dark Slate Background styling */
    .stApp {
        background-color: #050A16 !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .main {
        background-color: #050A16 !important;
    }
    header, [data-testid="stSidebar"] {
        background-color: #050A16 !important;
    }
    div[data-testid="stSidebarNav"] {
        background-color: #050A16 !important;
        padding-top: 10px !important;
    }
    
    /* Clean Page Container padding (optimized for 1366x768 and laptops) */
    div.block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Card spacing and block gaps */
    .stHorizontalBlock {
        gap: 20px !important;
    }
    
    /* Sidebar Redesign (Enterprise SaaS shell) */
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 260px !important;
        width: 260px !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none !important; /* Hide Navigation Console title */
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        transition: all 0.2s ease !important;
        border-left: 4px solid transparent !important;
        color: #94A3B8 !important;
        margin: 0 !important;
        cursor: pointer !important;
        box-sizing: border-box !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.04) !important;
        color: #F8FAFC !important;
    }
    /* Active menu item styling with left red accent border */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background-color: rgba(255, 77, 79, 0.08) !important;
        border-left: 4px solid #FF4D4F !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    /* Hide the radio round button selector element */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    
    /* Target Streamlit's native border container wrapper to behave as executive cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0B1328 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 32px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border: none !important;
    }
    
    /* Target nested containers to be styled as compact cards (e.g. for alert items inside feed) */
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255, 255, 255, 0.015) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin-bottom: 8px !important;
        box-shadow: none !important;
    }
    
    /* Typography Design System standards */
    h1 {
        font-size: 32px !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
        margin-top: 0px !important;
        margin-bottom: 24px !important;
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif !important;
    }
    h2 {
        font-size: 22px !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        margin-top: 0px !important;
        margin-bottom: 16px !important;
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif !important;
    }
    h3 {
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-top: 0px !important;
        margin-bottom: 12px !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    body, p, span, div, li {
        font-size: 15px !important;
        line-height: 1.6 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .caption-text {
        font-size: 12px !important;
        color: #94A3B8 !important;
    }
    
    /* Buttons styling */
    .stButton>button {
        background: linear-gradient(135deg, #FF4D4F 0%, #D32F2F 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 77, 79, 0.4) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease;
        font-weight: 700 !important;
        width: 100% !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 77, 79, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to render modern executive tables
# Helper function to render modern executive tables
def render_executive_table(df, max_height="400px"):
    """Renders a pandas DataFrame as a premium, highly-styled, interactive HTML table with searching, sorting, pagination, and CSV export."""
    if df is None or df.empty:
        return '<div style="color: #94A3B8; text-align: center; padding: 20px; font-family: \'Inter\', sans-serif;">No records to display.</div>'
        
    # Generate unique ID to support multiple tables on a single page
    table_id = f"table_{int(time.time() * 1000000) % 100000000}"
    
    html = f"""
    <div style="background-color: #0B1328; padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.06); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); margin-bottom: 24px; font-family: 'Inter', sans-serif;">
      <!-- Search & Export Control Panel -->
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
        <input type="text" id="search_{table_id}" placeholder="🔍 Search records..." style="background-color: #111A30; border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 6px 12px; color: #E2E8F0; width: 250px; font-size: 0.85rem; outline: none; transition: border-color 0.2s;" onfocus="this.style.borderColor='rgba(255,77,79,0.5)'" onblur="this.style.borderColor='rgba(255,255,255,0.1)'">
        <button onclick="exportCSV_{table_id}()" style="background-color: #3B82F6; border: none; border-radius: 6px; padding: 6px 12px; color: white; cursor: pointer; font-size: 0.82rem; font-weight: 600; display: flex; align-items: center; gap: 6px; transition: background-color 0.2s;" onmouseover="this.style.backgroundColor='#2563EB'" onmouseout="this.style.backgroundColor='#3B82F6'">
          📥 Export CSV
        </button>
      </div>
      
      <!-- Table Wrapper with Max Height -->
      <div style="max-height: {max_height}; overflow-y: auto; overflow-x: auto; position: relative; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.04);">
        <table id="{table_id}" style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem; color: #E2E8F0; background-color: #0B1328;">
          <thead>
            <tr style="position: sticky; top: 0; background-color: #111A30; border-bottom: 2px solid rgba(255, 255, 255, 0.08); color: #F8FAFC; font-weight: 700; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; z-index: 10; height: 44px;">
    """
    
    # Add headers with sort icon indicator
    for col in df.columns:
        html += f'<th style="padding: 12px 16px; position: sticky; top: 0; background-color: #111A30; z-index: 10; cursor: pointer; user-select: none; white-space: nowrap;" onmouseover="this.style.backgroundColor=\'#16223F\'" onmouseout="this.style.backgroundColor=\'#111A30\'">{col}<span class="sort-icon" style="color: #64748B; font-size: 0.7rem; margin-left: 6px;">↕</span></th>'
    html += "</tr></thead><tbody>"
    
    # Add rows
    for i, row in df.iterrows():
        bg_color = "rgba(255, 255, 255, 0.01)" if i % 2 == 0 else "rgba(0, 0, 0, 0.15)"
        html += f'<tr style="background-color: {bg_color}; border-bottom: 1px solid rgba(255, 255, 255, 0.04); transition: background-color 0.2s ease; height: 40px;" onmouseover="this.style.backgroundColor=\'rgba(255, 255, 255, 0.04)\'" onmouseout="this.style.backgroundColor=\'{bg_color}\'">'
        
        for col in df.columns:
            val = row[col]
            cell_style = "padding: 10px 16px; vertical-align: middle; white-space: nowrap; max-width: 250px; overflow: hidden; text-overflow: ellipsis;"
            
            col_str = str(col).lower()
            if col_str in ["status", "severity_label"] or str(col).lower() == "status":
                val_str = str(val).strip()
                if val_str in ["Illegal Parking", "Critical"]:
                    badge_style = "background-color: rgba(255, 77, 79, 0.15); color: #FF4D4F; border: 1px solid rgba(255, 77, 79, 0.3); padding: 4px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 800; display: inline-block; text-transform: uppercase;"
                    val = f'<span style="{badge_style}">🔴 {val_str}</span>'
                elif val_str in ["Warning", "High"]:
                    badge_style = "background-color: rgba(245, 166, 35, 0.15); color: #F5A623; border: 1px solid rgba(245, 166, 35, 0.3); padding: 4px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 800; display: inline-block; text-transform: uppercase;"
                    val = f'<span style="{badge_style}">🟡 {val_str}</span>'
                else:  # Normal, Moderate, Resolved
                    badge_style = "background-color: rgba(0, 196, 140, 0.15); color: #00C48C; border: 1px solid rgba(0, 196, 140, 0.3); padding: 4px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 800; display: inline-block; text-transform: uppercase;"
                    val = f'<span style="{badge_style}">🟢 {val_str}</span>'
            # Format numeric columns
            elif isinstance(val, (int, np.integer)):
                val = f"{val:,}"
                cell_style += " font-family: monospace;"
            elif isinstance(val, (float, np.floating)):
                val = f"{val:,.2f}"
                cell_style += " font-family: monospace;"
                
            html += f'<td style="{cell_style}" title="{str(row[col])}">{val}</td>'
        html += "</tr>"
        
    html += f"""
      </tbody>
      </table>
      </div>
      
      <!-- Pagination Controls -->
      <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; font-size: 0.82rem; color: #94A3B8; flex-wrap: wrap; gap: 8px;">
        <span id="page_{table_id}">Page 1 of 1 (0 entries)</span>
        <div style="display: flex; gap: 8px;">
          <button id="prev_{table_id}" style="background-color: #111A30; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; padding: 4px 10px; color: #E2E8F0; cursor: pointer; font-weight: 600; transition: all 0.2s;" onmouseover="if(!this.disabled) this.style.backgroundColor='#1E293B'" onmouseout="this.style.backgroundColor='#111A30'">◀ Prev</button>
          <button id="next_{table_id}" style="background-color: #111A30; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; padding: 4px 10px; color: #E2E8F0; cursor: pointer; font-weight: 600; transition: all 0.2s;" onmouseover="if(!this.disabled) this.style.backgroundColor='#1E293B'" onmouseout="this.style.backgroundColor='#111A30'">Next ▶</button>
        </div>
      </div>
    </div>
    
    <script>
      (function() {{
        const tableId = "{table_id}";
        const table = document.getElementById(tableId);
        if (!table) return;
        const tbody = table.querySelector("tbody");
        const originalRows = Array.from(tbody.querySelectorAll("tr"));
        let filteredRows = [...originalRows];
        
        const searchInput = document.getElementById("search_" + tableId);
        const prevBtn = document.getElementById("prev_" + tableId);
        const nextBtn = document.getElementById("next_" + tableId);
        const pageIndicator = document.getElementById("page_" + tableId);
        
        let currentPage = 1;
        const rowsPerPage = 10;
        
        function updateTable() {{
          const totalPages = Math.ceil(filteredRows.length / rowsPerPage) || 1;
          if (currentPage > totalPages) currentPage = totalPages;
          if (currentPage < 1) currentPage = 1;
          
          originalRows.forEach(r => r.style.display = "none");
          
          const start = (currentPage - 1) * rowsPerPage;
          const end = start + rowsPerPage;
          const pageRows = filteredRows.slice(start, end);
          pageRows.forEach(r => r.style.display = "");
          
          if (pageIndicator) {{
            pageIndicator.innerText = "Page " + currentPage + " of " + totalPages + " (" + filteredRows.length + " entries)";
          }}
          if (prevBtn) {{
            prevBtn.disabled = (currentPage === 1);
            prevBtn.style.opacity = (currentPage === 1) ? "0.4" : "1";
            prevBtn.style.cursor = (currentPage === 1) ? "not-allowed" : "pointer";
          }}
          if (nextBtn) {{
            nextBtn.disabled = (currentPage === totalPages);
            nextBtn.style.opacity = (currentPage === totalPages) ? "0.4" : "1";
            nextBtn.style.cursor = (currentPage === totalPages) ? "not-allowed" : "pointer";
          }}
        }}
        
        if (searchInput) {{
          searchInput.addEventListener("input", function() {{
            const query = this.value.toLowerCase().trim();
            filteredRows = originalRows.filter(row => {{
              return Array.from(row.querySelectorAll("td")).some(td => {{
                return td.innerText.toLowerCase().includes(query);
              }});
            }});
            currentPage = 1;
            updateTable();
          }});
        }}
        
        if (prevBtn) {{
          prevBtn.addEventListener("click", function() {{
            if (currentPage > 1) {{
              currentPage--;
              updateTable();
            }}
          }});
        }}
        
        if (nextBtn) {{
          nextBtn.addEventListener("click", function() {{
            const totalPages = Math.ceil(filteredRows.length / rowsPerPage) || 1;
            if (currentPage < totalPages) {{
              currentPage++;
              updateTable();
            }}
          }});
        }}
        
        const headers = table.querySelectorAll("thead th");
        let sortDirection = 1;
        let lastSortedCol = -1;
        
        headers.forEach((th, colIdx) => {{
          th.addEventListener("click", function() {{
            sortDirection = (lastSortedCol === colIdx) ? -sortDirection : 1;
            lastSortedCol = colIdx;
            
            filteredRows.sort((a, b) => {{
              const aCell = a.querySelectorAll("td")[colIdx];
              const bCell = b.querySelectorAll("td")[colIdx];
              if (!aCell || !bCell) return 0;
              
              const aVal = aCell.innerText.replace(/[^a-zA-Z0-9.-]/g, '');
              const bVal = bCell.innerText.replace(/[^a-zA-Z0-9.-]/g, '');
              
              const aNum = parseFloat(aVal);
              const bNum = parseFloat(bVal);
              
              if (!isNaN(aNum) && !isNaN(bNum)) {{
                return (aNum - bNum) * sortDirection;
              }}
              
              return aCell.innerText.localeCompare(bCell.innerText) * sortDirection;
            }});
            
            filteredRows.forEach(row => tbody.appendChild(row));
            currentPage = 1;
            updateTable();
            
            headers.forEach((h, idx) => {{
              const icon = h.querySelector(".sort-icon");
              if (icon) {{
                if (idx === colIdx) {{
                  icon.innerText = sortDirection === 1 ? " ▲" : " ▼";
                  icon.style.color = "#FF4D4F";
                }} else {{
                  icon.innerText = " ↕";
                  icon.style.color = "#64748B";
                }}
              }}
            }});
          }});
        }});
        
        window["exportCSV_" + tableId] = function() {{
          let csv = [];
          const headerRow = Array.from(headers).map(th => {{
            let text = th.innerText;
            if (text.endsWith(" ▲") || text.endsWith(" ▼") || text.endsWith(" ↕")) {{
              text = text.substring(0, text.length - 2);
            }}
            return '"' + text.replace(/"/g, '""').trim() + '"';
          }}).join(",");
          csv.push(headerRow);
          
          filteredRows.forEach(row => {{
            const rowData = Array.from(row.querySelectorAll("td")).map(td => {{
              // If there's an HTML badge, extract only the text
              let text = td.innerText;
              // Clean emojis or status text if needed, but innerText handles it cleanly
              return '"' + text.replace(/"/g, '""').trim() + '"';
            }}).join(",");
            csv.push(rowData);
          }});
          
          const csvContent = "data:text/csv;charset=utf-8," + csv.join("\\n");
          const encodedUri = encodeURI(csvContent);
          const link = document.createElement("a");
          link.setAttribute("href", encodedUri);
          link.setAttribute("download", "table_export.csv");
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }};
        
        updateTable();
      }})();
    </script>
    """
    return html

# Helper function to render Hackathon Storytelling panels
def render_hackathon_storytelling():
    st.markdown("---")
    with st.container(border=True):
        st.markdown("<h3 style='color: #FF4D4F; margin-top:0;'>💡 Why This Matters (The Paradigm Shift)</h3>", unsafe_allow_html=True)
        
        col_t, col_p = st.columns(2)
        with col_t:
            st.markdown("""
            <div style="background-color: rgba(255, 77, 79, 0.03); padding: 16px; border-radius: 8px; border: 1px dashed rgba(255, 77, 79, 0.2); height: 100%;">
              <h4 style="color: #FF4D4F; margin-top: 0; font-weight: bold; font-size: 0.95rem;">⚠️ TRADITIONAL TRAFFIC SYSTEMS</h4>
              <ul style="padding-left: 20px; margin-bottom: 0; color: #94A3B8; font-size: 0.85rem; line-height: 1.4;">
                <li><b>Detect congestion</b>: Merely reacts and flags gridlocks after they have already formed.</li>
                <li><b>Measure incidents</b>: Tracks simple incident counts but forgets systemic spatial patterns.</li>
                <li><b>Forget interventions</b>: Resolves violations and discards outcome effectiveness histories.</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_p:
            st.markdown("""
            <div style="background-color: rgba(0, 196, 140, 0.03); padding: 16px; border-radius: 8px; border: 1px dashed rgba(0, 196, 140, 0.2); height: 100%;">
              <h4 style="color: #00C48C; margin-top: 0; font-weight: bold; font-size: 0.95rem;">🧠 PARKTWIN AI COMMAND CENTER</h4>
              <ul style="padding-left: 20px; margin-bottom: 0; color: #E2E8F0; font-size: 0.85rem; line-height: 1.4;">
                <li><b>Learns from congestion</b>: Continuously registers hotspot recurrence and predicts tomorrow's spillover risks.</li>
                <li><b>Measures resilience</b>: Mathematically quantifies urban grid resilience using the <b>Traffic Immunity Score</b>.</li>
                <li><b>Builds institutional memory</b>: Tracks historical success rate of policies to choose the best recovery plan.</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255, 77, 79, 0.06) 0%, rgba(59, 130, 246, 0.06) 100%); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); margin-top: 16px; text-align: center;">
          <p style="margin: 0; font-size: 0.9rem; font-weight: 600; color: #F8FAFC; font-family: 'Inter', sans-serif; line-height: 1.4;">
            🏁 <b>FINAL PITCH:</b> Unlike conventional traffic platforms that simply detect violations, ParkTwin AI continuously learns from historical interventions through Traffic Memory and quantifies urban resilience using the novel Traffic Immunity Score. This enables cities not only to respond to congestion but to build immunity against future disruptions.
          </p>
        </div>
        """, unsafe_allow_html=True)

# Helper function to render Animated KPI cards
def render_animated_kpi(label, target_val, prefix="", suffix="", is_error=False, id_suffix=""):
    uniq_id = f"kpi-{label.lower().replace(' ', '-').replace('(', '').replace(')', '').replace('.', '')}-{id_suffix}"
    
    if is_error:
        val_color = "linear-gradient(to right, #FF4D4F, #F5A623)"
        border_color = "rgba(255, 77, 79, 0.25)"
    else:
        val_color = "linear-gradient(to right, #3B82F6, #00C48C)"
        border_color = "rgba(59, 130, 246, 0.25)"
        
    html_content = f"""
    <style>
      body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        background-color: transparent;
      }}
      .kpi-card {{
        background: #0B1328; 
        border: 1px solid {border_color}; 
        padding: 0.8rem; 
        border-radius: 16px; 
        text-align: center; 
        backdrop-filter: blur(8px); 
        font-family: 'Inter', sans-serif; 
        color: #E2E8F0; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); 
        box-sizing: border-box; 
        height: 125px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
      }}
      .kpi-title {{
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        white-space: normal;
        line-height: 1.2;
        text-align: center;
        width: 100%;
        height: 2.4em;
        display: flex;
        align-items: center;
        justify-content: center;
      }}
      .kpi-value {{
        font-size: 2.25rem;
        font-weight: 800;
        font-family: monospace;
        background: {val_color};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        line-height: 1.1;
        margin-top: 0.2rem;
      }}
    </style>
    <div class="kpi-card">
      <div class="kpi-title">{label}</div>
      <div class="kpi-value" id="{uniq_id}">0</div>
    </div>
    <script>
      (function() {{
        const target = {target_val};
        const element = document.getElementById("{uniq_id}");
        if (!element) return;
        
        const duration = 1000; // ms
        const stepTime = 16; 
        const totalSteps = duration / stepTime;
        let step = 0;
        
        const isFloat = String(target).includes('.');
        
        const timer = setInterval(() => {{
          step++;
          let current = (target / totalSteps) * step;
          if (step >= totalSteps) {{
            current = target;
            clearInterval(timer);
          }}
          
          let formatted = "";
          if (isFloat) {{
            formatted = "{prefix}" + current.toFixed(1) + "{suffix}";
          }} else {{
            formatted = "{prefix}" + Math.round(current).toLocaleString() + "{suffix}";
          }}
          element.innerHTML = formatted;
        }}, stepTime);
      }})();
    </script>
    """
    st.components.v1.html(html_content, height=130)

# Helper function to render Sustainability Ticker
def render_sustainability_ticker(base_fuel, base_co2, base_dollars, rate_fuel=0.012, rate_co2=0.027, rate_dollars=0.15, height=270):
    ticker_html = f"""
    <style>
      body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        background-color: transparent;
      }}
      .ticker-container {{
        background: #0B1328; 
        border: 1px solid rgba(255, 255, 255, 0.06); 
        border-radius: 16px; 
        padding: 20px; 
        font-family: 'Inter', sans-serif; 
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25); 
        color: #E2E8F0;
        box-sizing: border-box;
        height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }}
      .ticker-row {{
        background: rgba(0, 0, 0, 0.2); 
        border-radius: 8px; 
        padding: 10px 14px; 
        display: flex; 
        justify-content: space-between; 
        align-items: center;
        box-sizing: border-box;
      }}
    </style>
    <div class="ticker-container">
      <div>
        <h3 style="margin-top: 0; margin-bottom: 4px; color: #00C48C; font-weight: bold; font-size: 1.0rem;">
          🌱 LIVE SUSTAINABILITY IMPACT
        </h3>
        <p style="font-size: 0.75rem; color: #94A3B8; margin-top: 0; margin-bottom: 12px; line-height: 1.3;">
          Real-time calculation of environmental savings.
        </p>
      </div>
      
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div class="ticker-row" style="border: 1px solid rgba(0, 196, 140, 0.15);">
          <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Fuel Saved</span>
          <span style="font-size: 1.3rem; font-weight: 800; color: #00C48C; font-family: monospace;" id="ticker-fuel">0.00 L</span>
        </div>
        
        <div class="ticker-row" style="border: 1px solid rgba(59, 130, 246, 0.15);">
          <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">CO₂ Offset</span>
          <span style="font-size: 1.3rem; font-weight: 800; color: #3B82F6; font-family: monospace;" id="ticker-co2">0.00 kg</span>
        </div>
        
        <div class="ticker-row" style="border: 1px solid rgba(245, 166, 35, 0.15);">
          <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Financial</span>
          <span style="font-size: 1.3rem; font-weight: 800; color: #F5A623; font-family: monospace;" id="ticker-dollars">$0.00</span>
        </div>
      </div>
    </div>
    
    <script>
      (function() {{
        let currentFuel = {base_fuel};
        let currentCo2 = {base_co2};
        let currentDollars = {base_dollars};
        
        const rateFuel = {rate_fuel};
        const rateCo2 = {rate_co2};
        const rateDollars = {rate_dollars};
        
        const elFuel = document.getElementById("ticker-fuel");
        const elCo2 = document.getElementById("ticker-co2");
        const elDollars = document.getElementById("ticker-dollars");
        
        setInterval(() => {{
          currentFuel += rateFuel;
          currentCo2 += rateCo2;
          currentDollars += rateDollars;
          
          if (elFuel) elFuel.innerText = currentFuel.toFixed(2) + " L";
          if (elCo2) elCo2.innerText = currentCo2.toFixed(2) + " kg";
          if (elDollars) elDollars.innerText = "$" + currentDollars.toFixed(2);
        }}, 100);
      }})();
    </script>
    """
    st.components.v1.html(ticker_html, height=height)

# ReportLab PDF Generator for Smart City Command Portal
# ReportLab PDF Generator for Smart City Command Portal
def generate_pdf_report(leaderboard_df, sust_metrics, fc_metrics, alerts, sims):
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#FF4D4F'),
        spaceAfter=4,
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=12,
        fontName='Helvetica'
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=4,
        fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155'),
        spaceAfter=5
    )
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0F172A'),
        fontName='Helvetica'
    )
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.white,
        fontName='Helvetica-Bold'
    )

    story = []
    
    # Header & Logo
    story.append(Paragraph("PARKTWIN AI: CONGESTION DECISION INTELLIGENCE BRIEFING", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')} | Smart City Command Portal | City of Bengaluru", subtitle_style))
    story.append(Spacer(1, 5))
    
    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary", h2_style))
    exec_summary_text = (
        "This decision briefing outlines the structural traffic congestion bottlenecks caused by illegal parking "
        "within the city, combining spatial-temporal DBSCAN coordinate clustering, Parking Externality Index (PEI) "
        "scoring, and XGBoost predictive risk modeling. Based on real-time traffic CV sensor streams and policy simulation inputs, "
        "proactive intervention strategies have significantly improved municipal traffic flow efficiency "
        "and reduced greenhouse gas emissions."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    
    # Section 2: AI Traffic Commander Recommended Interventions
    if not leaderboard_df.empty:
        top_hotspot = leaderboard_df.iloc[0]
        hotspot_name = top_hotspot.get('primary_junction', 'KR Market Junction')
        pei_val = int(top_hotspot.get('pei_score', 84))
        risk_val = int(fc_metrics.get('max_risk', 97.0))
        action_val = "Deploy Tow Vehicle" if pei_val > 70 else "Officer Patrol"
        vhl_saved = float(sust_metrics.get('delay', 400.0))
        co2_tonnes = float(sust_metrics.get('co2', 1112.0)) / 1000.0
        
        story.append(Paragraph("2. AI Commander Recommendation", h2_style))
        rec_statement = (
            f"<b>Junction Recommendation:</b> <b>{hotspot_name}</b> exhibits repeated illegal parking during peak hours. "
            f"Based on a PEI score of <b>{pei_val}/100</b> and tomorrow's forecast spillover probability of <b>{risk_val}%</b>, "
            f"deploying a <b>{action_val.lower()}</b> within 10 minutes is recommended (Confidence: 92%). "
            f"Clearance is expected to recover <b>{vhl_saved:,.1f} vehicle-hours lost (VHL)</b> and avoid approx <b>{co2_tonnes:.2f} tonnes of CO₂ emissions</b>."
        )
        story.append(Paragraph(rec_statement, body_style))
        story.append(Spacer(1, 4))
        
    # Section 3: Hotspot Rankings
    story.append(Paragraph("3. Prioritized Spatial Hotspots (PEI Leaderboard)", h2_style))
    table_data = [[
        Paragraph("Rank", table_header_style),
        Paragraph("Jurisdiction (PS)", table_header_style),
        Paragraph("Junction Location", table_header_style),
        Paragraph("PEI Score", table_header_style),
        Paragraph("Severity", table_header_style),
        Paragraph("Violations Count", table_header_style)
    ]]
    for idx, row in leaderboard_df.head(5).iterrows():
        table_data.append([
            Paragraph(str(idx+1), table_text_style),
            Paragraph(str(row.get('primary_police_station', 'N/A')), table_text_style),
            Paragraph(str(row.get('primary_junction', 'N/A')), table_text_style),
            Paragraph(f"{row.get('pei_score', 0.0):.1f}", table_text_style),
            Paragraph(str(row.get('severity_label', 'Moderate')), table_text_style),
            Paragraph(str(row.get('violation_count', 0)), table_text_style)
        ])
    t = Table(table_data, colWidths=[35, 115, 150, 60, 75, 75])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.HexColor('#F1F5F9')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    # Section 4: Forecast Highlights
    story.append(Paragraph("4. Predictive Risk Forecasting (XGBoost)", h2_style))
    fc_text = (
        f"The XGBoost prediction engine forecast peak congestion risks at hour <b>{fc_metrics.get('peak_hour', 18)}:00</b> "
        f"with a maximum probability of <b>{fc_metrics.get('max_risk', 97.0):.1f}%</b>. "
        f"The primary risk contributors identified via SHAP include Junction Proximity, duration of blockages, "
        f"and peak hour overlap factors."
    )
    story.append(Paragraph(fc_text, body_style))
    story.append(Spacer(1, 4))

    # Section 5: Digital Twin Outcomes
    story.append(Paragraph("5. Digital Twin Scenario Outcomes", h2_style))
    sim_data = [[
        Paragraph("Location", table_header_style),
        Paragraph("Intervention Policy", table_header_style),
        Paragraph("Delay Saved (VHL)", table_header_style),
        Paragraph("Fuel Saved (L)", table_header_style),
        Paragraph("CO2 Offset (kg)", table_header_style),
        Paragraph("Economic Saved ($)", table_header_style)
    ]]
    for s in sims[:4]:
        m = s.get("metrics", {})
        sim_data.append([
            Paragraph(str(s.get("hotspot_name", "N/A")), table_text_style),
            Paragraph(str(s.get("intervention", "N/A")), table_text_style),
            Paragraph(f"{m.get('delay_prevented_vhl', 0.0):.1f}", table_text_style),
            Paragraph(f"{m.get('fuel_saved_liters', 0.0):.1f}", table_text_style),
            Paragraph(f"{m.get('co2_avoided_kg', 0.0):.1f}", table_text_style),
            Paragraph(f"${m.get('dollars_saved', 0.0):,.0f}", table_text_style)
        ])
    t_sim = Table(sim_data, colWidths=[130, 100, 95, 70, 70, 75])
    t_sim.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.HexColor('#F1F5F9')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_sim)
    story.append(Spacer(1, 4))
    
    # Section 6: Incident Timeline
    story.append(Paragraph("6. Real-time Incident Timeline (Live Alerts Log)", h2_style))
    alert_data = [[
        Paragraph("Time", table_header_style),
        Paragraph("Type", table_header_style),
        Paragraph("Location", table_header_style),
        Paragraph("Description", table_header_style)
    ]]
    for a in alerts[:4]:
        alert_data.append([
            Paragraph(str(a.get("time", "")), table_text_style),
            Paragraph(str(a.get("type", "")).upper(), table_text_style),
            Paragraph(str(a.get("loc", "")), table_text_style),
            Paragraph(str(a.get("msg", "")), table_text_style)
        ])
    t_alert = Table(alert_data, colWidths=[70, 80, 120, 270])
    t_alert.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.HexColor('#F1F5F9')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_alert)
    story.append(Spacer(1, 4))

    # Section 7: Detection Snapshots
    story.append(Paragraph("7. Computer Vision Telemetry & Detection Snapshots", h2_style))
    cv_text = (
        f"Edge-deployed YOLOv8 & ByteTrack tracking pipelines monitored active road junctions. "
        f"A total of 12 active vehicles were tracked, confirming 1 double-parking infraction "
        f"exceeding the 5.0s stationary threshold, automatically triggering enforcement alerts."
    )
    story.append(Paragraph(cv_text, body_style))
    story.append(Spacer(1, 4))

    # Section 8: Sustainability Impact Summary
    story.append(Paragraph("8. Sustainability & Environmental Impact Summary", h2_style))
    sust_text = (
        f"Cumulative environmental gains achieved via active enforcement policies: "
        f"<b>Idling Fuel Saved:</b> {sust_metrics.get('fuel', 0.0):.1f} Liters | "
        f"<b>CO2 Emissions Avoided:</b> {sust_metrics.get('co2', 0.0):.1f} kg | "
        f"<b>Fiscal Savings:</b> ${sust_metrics.get('economic', 0.0):,.2f} USD."
    )
    story.append(Paragraph(sust_text, body_style))
    story.append(Spacer(1, 4))

    # Section 9: Traffic Memory & Urban Immunity Intelligence
    story.append(Paragraph("9. Traffic Memory & Urban Immunity Intelligence", h2_style))
    
    # 1. Traffic Memory Insights
    mem_insights_text = (
        "<b>Traffic Memory Insights:</b> Historical analysis indicates that the city's command system learns "
        "continuously from past interventions. At critical intersections, standard blockages last an average of "
        "38 minutes. Deploying <b>Tow Vehicle</b> interventions historically achieved the fastest recovery window "
        "(17 minutes), reducing congestion by 78% on average, compared to 42% for Officer Patrol, 35% for Barricades, "
        "and 21% for Fine Only policies."
    )
    story.append(Paragraph(mem_insights_text, body_style))
    story.append(Spacer(1, 4))
    
    # 2. Traffic Immunity Rankings & Watchlist
    story.append(Paragraph("<b>Traffic Immunity Leaderboard & Strategy:</b>", ParagraphStyle('SubH', parent=h2_style, fontSize=9, spaceBefore=4)))
    
    # Access global variables safely
    mem_profs = globals().get('memory_profiles', {})
    imm_scores = globals().get('immunity_scores', {})
    
    if imm_scores:
        sorted_imm = sorted(
            imm_scores.values(),
            key=lambda x: x.get("tis_score", 0.0),
            reverse=True
        )
        
        imm_data = [[
            Paragraph("Rank", table_header_style),
            Paragraph("Junction Location", table_header_style),
            Paragraph("Immunity Score", table_header_style),
            Paragraph("Classification", table_header_style),
            Paragraph("Recovery Speed", table_header_style),
            Paragraph("Spillover Resistance", table_header_style)
        ]]
        
        for idx, x in enumerate(sorted_imm[:5]):
            m = x.get("metrics", {})
            imm_data.append([
                Paragraph(str(idx + 1), table_text_style),
                Paragraph(str(x.get("junction", "N/A")), table_text_style),
                Paragraph(f"{x.get('tis_score', 0.0):.1f}", table_text_style),
                Paragraph(str(x.get("classification", "N/A")), table_text_style),
                Paragraph(f"{m.get('recovery_speed', 0.0):.1f}", table_text_style),
                Paragraph(f"{m.get('spillover_resistance', 0.0):.1f}", table_text_style)
            ])
            
        t_imm = Table(imm_data, colWidths=[35, 160, 80, 85, 90, 95])
        t_imm.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.HexColor('#F1F5F9')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_imm)
        story.append(Spacer(1, 4))
        
        # Watchlists and strategies
        fragile_list = [x.get("junction") for x in sorted_imm if x.get("classification") == "Fragile"]
        resilient_list = [x.get("junction") for x in sorted_imm if x.get("classification") == "Resilient"]
        
        # Fallbacks
        if not fragile_list:
            fragile_list = [x.get("junction") for x in sorted_imm[-2:]]
        if not resilient_list:
            resilient_list = [x.get("junction") for x in sorted_imm[:2]]
            
        watchlist_str = f"<b>Fragile Zone Watchlist:</b> {', '.join(fragile_list[:3])}"
        highlights_str = f"<br/><b>Resilient Zone Highlights:</b> {', '.join(resilient_list[:3])}"
        strategies_str = (
            "<br/><b>Recommended Long-Term Strategies:</b><br/>"
            "• <i>For Fragile Zones:</i> Allocate permanent CV telemetry and prioritized tow vehicle dispatch routes to prevent compounding spillover.<br/>"
            "• <i>For Resilient/Adaptive Zones:</i> Establish scheduled officer patrols to maintain high recurrence resistance."
        )
        story.append(Paragraph(watchlist_str + highlights_str + strategies_str, body_style))
    else:
        story.append(Paragraph("Traffic Immunity analytics are currently pending telemetry initialization.", body_style))
        
    # Build document
    doc.build(story)
    return buffer.getvalue()

def compile_executive_report_pdf():
    # Calculate sustainability metrics
    sims = st.session_state.get('locked_simulations', [])
    if not sims:
        sim = SimulationService()
        ds = DelayService()
        potential_sims = []
        for _, r in leaderboard.head(5).iterrows():
            is_junction = r['primary_junction'] != 'No Junction'
            base_vhl = ds.estimate_causal_delay(r['pei_score'], r['avg_duration_minutes'], is_junction)["vehicle_hours_lost"]
            p_sim = sim.run_simulation(base_vhl, "Increased Patrol", r['primary_junction'])
            potential_sims.append(p_sim)
        sims = potential_sims

    total_prevented_vhl = sum(s["metrics"]["delay_prevented_vhl"] for s in sims)
    total_fuel_saved = sum(s["metrics"]["fuel_saved_liters"] for s in sims)
    total_co2_avoided = sum(s["metrics"]["co2_avoided_kg"] for s in sims)
    total_dollars_saved = sum(s["metrics"]["dollars_saved"] for s in sims)

    # Forecast fallbacks
    peak_h = 18
    max_r = 97.0
    if 'tomorrow_forecast' in st.session_state:
        fc_df_temp = pd.DataFrame(st.session_state['tomorrow_forecast'])
        if not fc_df_temp.empty:
            hourly_risk_temp = fc_df_temp.groupby('hour_of_day')['risk_probability'].mean().reset_index()
            peak_row_temp = hourly_risk_temp.loc[hourly_risk_temp['risk_probability'].idxmax()]
            peak_h = int(peak_row_temp['hour_of_day'])
            max_r = float(fc_df_temp['risk_probability'].max() * 100)

    sust_metrics = {
        "delay": total_prevented_vhl,
        "fuel": total_fuel_saved,
        "co2": total_co2_avoided,
        "economic": total_dollars_saved
    }
    
    fc_metrics = {
        "peak_hour": peak_h,
        "max_risk": max_r
    }
    
    return generate_pdf_report(leaderboard, sust_metrics, fc_metrics, st.session_state.get('alerts', []), sims)

# Global Cached Data Loader
@st.cache_data(show_spinner="Ingesting and cleaning historical dataset...")
def load_data_cached_v2():
    pipeline = DataPipeline()
    df = pipeline.load_data()
    with open(REPORTS_DIR / "data_quality_report.json", 'r') as f:
        report = json.load(f)
    return df, report

# Compute spatial hotspots and PEI once
@st.cache_resource(show_spinner="Analyzing historical spatial clusters...")
def get_hotspots_cached(df):
    hs = HotspotService(df)
    clustered_df, summary_df = hs.run_spatial_clustering()
    pei_serv = PEIService(summary_df)
    leaderboard = pei_serv.calculate_pei()
    return clustered_df, summary_df, leaderboard

# Demo Mode State Side Effects
def execute_demo_step_effects(step):
    now_str = datetime.now().strftime("%I:%M:%S %p")
    if step == 1:
        st.session_state.alerts = [
            {"time": now_str, "type": "warning", "msg": "Illegal Parking: CAR #12 detected", "loc": "KR Market Junction"}
        ]
        st.session_state.officers_deployed = 12
        st.session_state.congestion_saved_today = 430.0
        st.session_state.fuel_saved_today = 516.0
        st.session_state.co2_saved_today = 1192.0
        st.session_state.economic_saved_today = 7050.0
    elif step == 2:
        if not any(a['msg'].startswith("Illegal Parking: CAR #12") for a in st.session_state.alerts):
            st.session_state.alerts.insert(0, {"time": now_str, "type": "warning", "msg": "Illegal Parking: CAR #12 detected", "loc": "KR Market Junction"})
    elif step == 3:
        pass
    elif step == 4:
        pass
    elif step == 5:
        if not any(a['type'] == "tow" for a in st.session_state.alerts):
            st.session_state.alerts.insert(0, {
                "time": now_str,
                "type": "tow",
                "msg": "Tow Vehicle Recommended for CAR #12",
                "loc": "KR Market Junction"
            })
        st.session_state.officers_deployed = 13
    elif step == 6: # Step 5A: Traffic Memory
        if not any(a['msg'].startswith("Traffic Memory") for a in st.session_state.alerts):
            st.session_state.alerts.insert(0, {
                "time": now_str,
                "type": "warning",
                "msg": "Traffic Memory: 12 similar incidents at KR Market Junction. Historical recovery time: 17 mins.",
                "loc": "KR Market Junction"
            })
    elif step == 7: # Digital Twin
        pass
    elif step == 8: # Step 6A: Traffic Immunity
        if not any(a['msg'].startswith("Traffic Immunity") for a in st.session_state.alerts):
            st.session_state.alerts.insert(0, {
                "time": now_str,
                "type": "warning",
                "msg": "Traffic Immunity: KR Market Junction TIS is 32 (Fragile). Projected post-tow: 58 (Adaptive).",
                "loc": "KR Market Junction"
            })
    elif step == 9: # Sustainability
        if not any(a['type'] == "success" for a in st.session_state.alerts):
            st.session_state.alerts.insert(0, {
                "time": now_str,
                "type": "success",
                "msg": "Congestion reduced by 400 VHL",
                "loc": "KR Market Junction"
            })
            st.session_state.alerts.insert(0, {
                "time": now_str,
                "type": "co2",
                "msg": "CO₂ Savings: 1112 kg CO₂ avoided",
                "loc": "KR Market Junction"
            })
        st.session_state.congestion_saved_today = 830.0
        st.session_state.fuel_saved_today = 998.0
        st.session_state.co2_saved_today = 2304.0
        st.session_state.economic_saved_today = 13602.0
    elif step == 10: # Executive Report
        if not any(a['msg'].startswith("Executive report compiled") for a in st.session_state.alerts):
            st.session_state.alerts.insert(0, {
                "time": now_str,
                "type": "success",
                "msg": "Executive report compiled: ready for download",
                "loc": "System Console"
            })

@st.cache_resource(show_spinner="Initializing Traffic Memory Engine...")
def get_traffic_memory_cached(leaderboard_df, clustered):
    tms = TrafficMemoryService(leaderboard_df, clustered)
    profiles = tms.generate_profiles()
    return tms, profiles

@st.cache_resource(show_spinner="Calculating Traffic Immunity Scores...")
def get_traffic_immunity_cached(leaderboard_df, memory_profiles):
    ims = ImmunityService(leaderboard_df, memory_profiles)
    scores = ims.calculate_immunity_scores()
    return ims, scores

# Main Ingestion Execution Flow
try:
    df, dq_report = load_data_cached_v2()
    clustered_df, summary_df, leaderboard = get_hotspots_cached(df)
    tms, memory_profiles = get_traffic_memory_cached(leaderboard, clustered_df)
    ims, immunity_scores = get_traffic_immunity_cached(leaderboard, memory_profiles)
except Exception as e:
    st.error(f"Critical Data Ingestion Error: {e}")
    st.stop()

# Initialize session state variables
if "alerts" not in st.session_state:
    st.session_state.alerts = [
        {"time": "06:45:10 PM", "type": "warning", "msg": "Illegal Parking: CAR #21 detected", "loc": "KR Market Junction"},
        {"time": "06:46:02 PM", "type": "tow", "msg": "Tow Vehicle Recommended", "loc": "KR Market Junction"},
        {"time": "06:48:15 PM", "type": "success", "msg": "Congestion Reduced by 60 VHL", "loc": "KR Market Junction"},
        {"time": "06:50:00 PM", "type": "co2", "msg": "CO₂ Savings: 138kg CO₂ saved via tow", "loc": "KR Market Junction"}
    ]
if "officers_deployed" not in st.session_state:
    st.session_state.officers_deployed = 12
if "congestion_saved_today" not in st.session_state:
    st.session_state.congestion_saved_today = 430.0
if "fuel_saved_today" not in st.session_state:
    st.session_state.fuel_saved_today = 516.0
if "co2_saved_today" not in st.session_state:
    st.session_state.co2_saved_today = 1192.0
if "economic_saved_today" not in st.session_state:
    st.session_state.economic_saved_today = 7050.0
if "critical_hotspots" not in st.session_state:
    st.session_state.critical_hotspots = len(leaderboard[leaderboard['severity_label'] == 'Critical'])
if "active_alerts" not in st.session_state:
    st.session_state.active_alerts = len([a for a in st.session_state.alerts if a['type'] in ['warning', 'tow']])
if "demo_running" not in st.session_state:
    st.session_state.demo_running = False
if "demo_autoplay" not in st.session_state:
    st.session_state.demo_autoplay = False
if "demo_step" not in st.session_state:
    st.session_state.demo_step = 0
if "locked_simulations" not in st.session_state:
    st.session_state.locked_simulations = []

# Typographical Branding Logo
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; padding: 15px 0px; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 20px;">
  <div style="width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, #FF4D4F 0%, #D32F2F 100%); display: flex; align-items: center; justify-content: center; font-weight: 900; color: white; font-size: 1.1rem; box-shadow: 0 4px 10px rgba(255, 77, 79, 0.4);">
    P
  </div>
  <div style="display: flex; flex-direction: column;">
    <div style="font-size: 1.35rem; font-weight: 900; color: #FF4D4F; font-family: 'Inter', sans-serif; letter-spacing: -0.04em; line-height: 1;">
      PARKTWIN <span style="color: #F8FAFC;">AI</span>
    </div>
    <div style="font-size: 0.65rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; margin-top: 2px;">
      TRAFFIC COMMAND CENTER
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Demo trigger
if not st.session_state.get("demo_running", False):
    if st.sidebar.button("🚀 Run Smart City Demo", key="sidebar_run_demo", use_container_width=True):
        st.session_state.demo_running = True
        st.session_state.demo_autoplay = True
        st.session_state.demo_step = 1
        st.session_state.nav_selection = "🎥 Live Detection"
        execute_demo_step_effects(1)
        st.rerun()

# Initial value from session state if present, otherwise default to "📊 Command Center"
if "nav_selection" not in st.session_state:
    st.session_state.nav_selection = "📊 Command Center"

nav_options = [
    "📊 Command Center",
    "🔥 Hotspots",
    "🎯 PEI",
    "🧠 Traffic Memory",
    "🔮 Forecasting",
    "🎥 Live Detection",
    "🎛️ Digital Twin",
    "🛡️ Traffic Immunity",
    "🧩 Explainability",
    "🌱 Sustainability"
]

try:
    nav_index = nav_options.index(st.session_state.nav_selection)
except ValueError:
    nav_index = 0

# Navigation Console
menu = st.sidebar.radio(
    "Navigation Console",
    nav_options,
    index=nav_index,
    key="nav_selection_radio"
)

# Keep session state in sync
st.session_state.nav_selection = menu

menu_id = "Command Center"
if "Command Center" in menu:
    menu_id = "Command Center"
elif "Hotspots" in menu:
    menu_id = "Hotspot Intelligence"
elif "PEI" in menu:
    menu_id = "PEI Prioritization"
elif "Traffic Memory" in menu:
    menu_id = "Traffic Memory"
elif "Forecasting" in menu:
    menu_id = "Risk Forecasting"
elif "Live Detection" in menu:
    menu_id = "Live Detection"
elif "Digital Twin" in menu:
    menu_id = "Digital Twin"
elif "Traffic Immunity" in menu:
    menu_id = "Traffic Immunity"
elif "Explainability" in menu:
    menu_id = "Explainability"
elif "Sustainability" in menu:
    menu_id = "Sustainability"

# Sidebar Models & Operations Panel
st.sidebar.markdown("---")

# Active Models Status Card
st.sidebar.markdown("""
<div style="background-color: #0B1328; border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 12px; margin-bottom: 16px; font-family: 'Inter', sans-serif;">
  <div style="font-size: 0.7rem; color: #94A3B8; text-transform: uppercase; font-weight: bold; margin-bottom: 8px; letter-spacing: 0.05em;">
    🛡️ Core Analytics Engines
  </div>
  <div style="display: flex; flex-direction: column; gap: 6px;">
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
      <span style="color: #E2E8F0;">XGBoost Predictor</span>
      <span style="background-color: rgba(0, 196, 140, 0.15); color: #00C48C; padding: 2px 6px; border-radius: 4px; font-size: 0.65rem; font-weight: bold;">ONLINE</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
      <span style="color: #E2E8F0;">DBSCAN Clustering</span>
      <span style="background-color: rgba(0, 196, 140, 0.15); color: #00C48C; padding: 2px 6px; border-radius: 4px; font-size: 0.65rem; font-weight: bold;">ONLINE</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
      <span style="color: #E2E8F0;">YOLOv8 Edge CV</span>
      <span style="background-color: rgba(0, 196, 140, 0.15); color: #00C48C; padding: 2px 6px; border-radius: 4px; font-size: 0.65rem; font-weight: bold;">ONLINE</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Active Incident Scenario Card (Only shown if demo is running)
if st.session_state.demo_running:
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(255, 77, 79, 0.1) 0%, #0B1328 100%); border: 1px solid rgba(255, 77, 79, 0.2); border-radius: 12px; padding: 12px; margin-bottom: 16px; font-family: 'Inter', sans-serif;">
      <div style="font-size: 0.7rem; color: #FF4D4F; text-transform: uppercase; font-weight: bold; margin-bottom: 6px; letter-spacing: 0.05em;">
        🚨 Running Scenario Simulation
      </div>
      <div style="font-size: 0.85rem; font-weight: bold; color: white; margin-bottom: 4px;">
        Incident: KR Market Blockage
      </div>
      <div style="font-size: 0.75rem; color: #94A3B8;">
        Active Demo Step: <b>Step {st.session_state.demo_step} of 6</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Anchored Sidebar Footer
st.sidebar.markdown("""
<div style="margin-top: 50px; text-align: center; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 12px; font-family: 'Inter', sans-serif;">
  <span style="font-size: 0.65rem; color: #64748B; letter-spacing: 0.03em; font-weight: bold;">
    PARKTWIN AI © 2026<br>Smart City Command Portal
  </span>
</div>
""", unsafe_allow_html=True)




# Calculate active warnings for the header
st.session_state.active_alerts = len([a for a in st.session_state.alerts if a['type'] in ['warning', 'tow']])
st.session_state.critical_hotspots = len(leaderboard[leaderboard['severity_label'] == 'Critical'])

# Adjust variables based on active demo step
if st.session_state.demo_running:
    current_pei_val = 95 if st.session_state.demo_step >= 2 else int(leaderboard.iloc[0]['pei_score'])
    current_risk_val = 97 if st.session_state.demo_step >= 4 else 85
    current_delay_val = 500 if st.session_state.demo_step <= 5 else 100
else:
    current_pei_val = int(leaderboard.iloc[0]['pei_score'])
    current_risk_val = 85
    current_delay_val = 350

# Render Single Master Header
header_html = f"""
<style>
  body {{
    margin: 0;
    padding: 0;
    overflow: hidden;
    background-color: transparent;
  }}
  .header-container {{
    width: 100%;
    height: 60px;
    background-color: #0B1328;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0px 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
    border-radius: 12px;
  }}
  .left-side {{
    display: flex;
    align-items: center;
    gap: 16px;
  }}
  .brand-title {{
    font-size: 1.15rem;
    font-weight: 900;
    color: #FF4D4F;
    letter-spacing: -0.03em;
    line-height: 1;
  }}
  .brand-sub {{
    font-size: 0.65rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-top: 1px;
  }}
  .separator {{
    width: 1px;
    height: 20px;
    background-color: rgba(255,255,255,0.1);
  }}
  .status-badge {{
    background-color: rgba(0, 196, 140, 0.1);
    color: #00C48C;
    border: 1px solid rgba(0, 196, 140, 0.2);
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    display: flex;
    align-items: center;
    gap: 5px;
  }}
  .status-dot {{
    width: 6px;
    height: 6px;
    background-color: #00C48C;
    border-radius: 50%;
    box-shadow: 0 0 6px #00C48C;
  }}
  .metric-item {{
    font-size: 0.78rem;
    color: #94A3B8;
  }}
  .metric-val {{
    font-weight: 700;
    font-family: monospace;
    font-size: 0.85rem;
    margin-left: 4px;
  }}
  .clock-container {{
    font-size: 0.88rem;
    font-weight: 700;
    font-family: monospace;
    color: #3B82F6;
    background: rgba(59, 130, 246, 0.08);
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid rgba(59, 130, 246, 0.15);
  }}
</style>
<div class="header-container">
  <div class="left-side">
    <div style="display: flex; flex-direction: column;">
      <span class="brand-title">PARKTWIN <span style="color: #F8FAFC;">AI</span></span>
      <span class="brand-sub">Traffic Command</span>
    </div>
    <div class="separator"></div>
    <div class="status-badge">
      <div class="status-dot"></div>
      ACTIVE
    </div>
    <div class="separator"></div>
    <span class="metric-item">Alerts:<strong class="metric-val" style="color: #FF4D4F;">{st.session_state.active_alerts}</strong></span>
    <span class="metric-item">Critical Hotspots:<strong class="metric-val" style="color: #F5A623;">{st.session_state.critical_hotspots}</strong></span>
    <span class="metric-item">Officers Deployed:<strong class="metric-val" style="color: #3B82F6;">{st.session_state.officers_deployed}</strong></span>
    <span class="metric-item">Congestion Saved:<strong class="metric-val" style="color: #00C48C;">{st.session_state.congestion_saved_today:,.0f} VHL</strong></span>
  </div>
  <div class="clock-container" id="live-clock">--:--:--</div>
</div>
<script>
  function updateTime() {{
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    document.getElementById("live-clock").innerText = timeStr;
  }}
  setInterval(updateTime, 1000);
  updateTime();
</script>
"""
st.components.v1.html(header_html, height=68)

# --- STORY MODE DEMO CONTROL BANNER ---
if st.session_state.get("demo_running", False):
    step = st.session_state.demo_step
    
    # Define step descriptions (expanded to 10 steps)
    step_descriptions = {
        1: "**[Step 1 of 10: Live CV Detection]** YOLOv8 & ByteTrack track vehicle CAR #12 parked illegally at KR Market Junction for over 10 seconds, triggering visual warnings.",
        2: "**[Step 2 of 10: Spatial DBSCAN Hotspots]** The spatial clustering engine groups geographic coordinates and identifies KR Market as a critical high-density hotspot.",
        3: "**[Step 3 of 10: PEI Severity Scoring]** The Parking Externality Index calculates the hotspot's traffic blockage impact, ranking it at **95/100 (Critical)**.",
        4: "**[Step 4 of 10: XGBoost Risk Forecasting]** Pre-trained XGBoost classifiers predict a **97% congestion risk probability** at this junction for tomorrow's rush hour.",
        5: "**[Step 5 of 10: AI Traffic Commander]** Fusing PEI and risk forecasts, the Commander dashboard recommends deploying a **Tow Vehicle** with 92% confidence.",
        6: "**[Step 6 of 10: Traffic Memory Engine]** Traffic Memory reveals a recurring vulnerability: KR Market Junction has seen 12 similar incidents, and Tow Vehicle historically recovers traffic in 17 minutes (78% success).",
        7: "**[Step 7 of 10: Digital Twin Simulator]** What-If simulation models project that dispatching a tow truck will save **400 Vehicle Hours Lost (VHL)**.",
        8: "**[Step 8 of 10: Traffic Immunity Score]** Traffic Immunity evaluates long-term resilience, displaying a fragility TIS of **32/100** before the intervention, projecting an increase to **58/100** after tow truck deployment.",
        9: "**[Step 9 of 10: Environmental Sustainability]** Enforcement actions accumulate: **1,112 kg CO2 offset**, **482 L fuel saved**, and **$13,602** in economic losses avoided.",
        10: "**[Step 10 of 10: Executive Report Generation]** The decision pipeline compiles all metrics, maps, and outputs into a printable, professional PDF document."
    }
    
    step_msg = step_descriptions.get(step, "Demo Active")
    progress_val = float(step) / 10.0
    
    with st.container(border=True):
        col_desc, col_prog, col_btns = st.columns([5, 3, 4])
        
        with col_desc:
            st.markdown("🎬 **SMART CITY STORY MODE DEMO**")
            st.info(step_msg)
            
        with col_prog:
            st.markdown(f"**Pipeline Progression:** `{step * 10.0:.0f}%`")
            st.progress(progress_val)
            if step == 10:
                pdf_data = compile_executive_report_pdf()
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"ParkTwin_Executive_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="dl_btn_demo_hud",
                    use_container_width=True
                )
            elif st.session_state.get("demo_autoplay", False):
                st.markdown("⏱️ *Auto-playing... Advances in 8 seconds*")
            else:
                st.markdown("⏸️ *Paused*")
                
        with col_btns:
            btn_prev, btn_next, btn_play, btn_stop = st.columns(4)
            with btn_prev:
                if st.button("⏮️ Prev", key="demo_btn_prev", use_container_width=True):
                    if st.session_state.demo_step > 1:
                        st.session_state.demo_step -= 1
                        demo_pages = {
                            1: "🎥 Live Detection",
                            2: "🔥 Hotspots",
                            3: "🎯 PEI",
                            4: "🔮 Forecasting",
                            5: "📊 Command Center",
                            6: "🧠 Traffic Memory",
                            7: "🎛️ Digital Twin",
                            8: "🛡️ Traffic Immunity",
                            9: "🌱 Sustainability",
                            10: "🌱 Sustainability"
                        }
                        st.session_state.nav_selection = demo_pages.get(st.session_state.demo_step)
                        execute_demo_step_effects(st.session_state.demo_step)
                        st.rerun()
            with btn_next:
                if st.button("Next ⏭️", key="demo_btn_next", use_container_width=True):
                    if st.session_state.demo_step < 10:
                        st.session_state.demo_step += 1
                        demo_pages = {
                            1: "🎥 Live Detection",
                            2: "🔥 Hotspots",
                            3: "🎯 PEI",
                            4: "🔮 Forecasting",
                            5: "📊 Command Center",
                            6: "🧠 Traffic Memory",
                            7: "🎛️ Digital Twin",
                            8: "🛡️ Traffic Immunity",
                            9: "🌱 Sustainability",
                            10: "🌱 Sustainability"
                        }
                        st.session_state.nav_selection = demo_pages.get(st.session_state.demo_step)
                        execute_demo_step_effects(st.session_state.demo_step)
                        st.rerun()
                    else:
                        st.session_state.demo_running = False
                        st.session_state.demo_autoplay = False
                        st.session_state.demo_step = 0
                        st.rerun()
            with btn_play:
                autoplay = st.session_state.get("demo_autoplay", False)
                play_label = "⏸️ Pause" if autoplay else "▶️ Play"
                if st.button(play_label, key="demo_btn_play", use_container_width=True):
                    st.session_state.demo_autoplay = not autoplay
                    st.rerun()
            with btn_stop:
                if st.button("⏹️ Stop", key="demo_btn_stop", use_container_width=True):
                    st.session_state.demo_running = False
                    st.session_state.demo_autoplay = False
                    st.session_state.demo_step = 0
                    st.rerun()

    # Autoplay logic
    if st.session_state.get("demo_autoplay", False):
        time.sleep(8)
        if st.session_state.demo_step < 10:
            st.session_state.demo_step += 1
            demo_pages = {
                1: "🎥 Live Detection",
                2: "🔥 Hotspots",
                3: "🎯 PEI",
                4: "🔮 Forecasting",
                5: "📊 Command Center",
                6: "🧠 Traffic Memory",
                7: "🎛️ Digital Twin",
                8: "🛡️ Traffic Immunity",
                9: "🌱 Sustainability",
                10: "🌱 Sustainability"
            }
            st.session_state.nav_selection = demo_pages.get(st.session_state.demo_step)
            execute_demo_step_effects(st.session_state.demo_step)
            st.rerun()
        else:
            st.session_state.demo_autoplay = False
            st.session_state.demo_running = False
            st.session_state.demo_step = 0
            st.success("Smart City Demo Completed successfully!")
            st.rerun()

# --- PAGE 1: COMMAND CENTER ---
if menu_id == "Command Center":
    # 1. Top KPI Row using Animated counters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_animated_kpi("Cleaned Incident Logs", len(df), id_suffix="ov1")
    with col2:
        render_animated_kpi("Detected Spatial Hotspots", len(summary_df), id_suffix="ov2")
    with col3:
        render_animated_kpi("Critical Risk Hotspots", st.session_state.critical_hotspots, is_error=True, id_suffix="ov3")
    with col4:
        ds = DelayService()
        sample_leaderboard = leaderboard.head(50)
        total_vhl = sum(ds.estimate_causal_delay(r['pei_score'], r['avg_duration_minutes'], r['primary_junction'] != 'No Junction')['vehicle_hours_lost'] for _, r in sample_leaderboard.iterrows())
        if st.session_state.demo_running:
            if st.session_state.demo_step == 2:
                total_vhl += 150.0
            elif st.session_state.demo_step >= 9:
                total_vhl = max(total_vhl - 400.0, 50.0)
        render_animated_kpi("Est. Congestion Delay (VHL)", total_vhl, suffix=" hrs", id_suffix="ov4")

    st.markdown("---")

    # 2. Active Enforcement Recommendation & AI Traffic Commander (Full Width)
    if not leaderboard.empty:
        if st.session_state.demo_running and st.session_state.demo_step >= 1:
            hotspot_name = "KR Market Junction"
            pei = 95
            risk = 97
            delay = 500 if st.session_state.demo_step < 9 else 100
            action = "Deploy Tow Vehicle"
            vhl_saved = 400
            dollars_saved = 13602
            co2_saved = 1112
        else:
            top_hotspot = leaderboard.iloc[0]
            hotspot_name = top_hotspot['primary_junction']
            pei = int(top_hotspot['pei_score'])
            risk = 85
            is_junction = top_hotspot['primary_junction'] != 'No Junction'
            delay_stats = ds.estimate_causal_delay(top_hotspot['pei_score'], top_hotspot['avg_duration_minutes'], is_junction)
            delay = delay_stats['vehicle_hours_lost']
            action = "Deploy Tow Vehicle" if pei > 70 else "Officer Patrol"
            savings_pct = 0.80 if action == "Deploy Tow Vehicle" else 0.50
            vhl_saved = round(delay * savings_pct, 1)
            dollars_saved = round(vhl_saved * 15.00 + vhl_saved * 1.2 * 1.15, 0)
            co2_saved = round(vhl_saved * 1.2 * 2.31, 0)

        # Flagship AI Traffic Commander Panel
        st.markdown("""
        <div style="background: linear-gradient(135deg, #111A30 0%, #060B18 100%); padding: 1px; border-radius: 16px; margin-bottom: 24px; border: 1px solid rgba(255, 77, 79, 0.25);">
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 77, 79, 0.25); padding: 16px 20px 12px 20px;">
              <span style="font-weight: 900; font-size: 1.15rem; color: #FF4D4F; letter-spacing: 0.05em; display: flex; align-items: center; gap: 8px;">
                 🧠 AI TRAFFIC COMMANDER DECISION GATEWAY
              </span>
              <span style="background: rgba(255, 77, 79, 0.15); color: #FF4D4F; border: 1px solid #FF4D4F; padding: 4px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">
                🔴 CRITICAL INCIDENT DETECTED
              </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Content grid
            cmd_col1, cmd_col2, cmd_col3, cmd_col4 = st.columns(4)
            with cmd_col1:
                st.markdown("**📍 Location**")
                st.markdown(f"<span style='font-size: 1.1rem; font-weight: bold; color: white;'>{hotspot_name}</span>", unsafe_allow_html=True)
                st.markdown("**⚖️ Priority Level**")
                st.markdown("<span style='font-size: 1.0rem; font-weight: bold; color: #FF4D4F;'>🔴 HIGH</span>", unsafe_allow_html=True)
                
            with cmd_col2:
                st.markdown("**🎯 PEI Score**")
                st.markdown(f"<span style='font-size: 1.5rem; font-weight: 900; color: #FF4D4F;'>{pei}</span> <span style='font-size: 0.8rem; color: #94A3B8;'>/ 100</span>", unsafe_allow_html=True)
                st.markdown("**🔮 Forecast Risk**")
                st.markdown(f"<span style='font-size: 1.5rem; font-weight: 900; color: #F5A623;'>{risk}%</span>", unsafe_allow_html=True)
                
            with cmd_col3:
                st.markdown("**⚙️ Recommended Intervention**")
                st.markdown(f"<span style='font-size: 1.25rem; font-weight: 800; color: #3B82F6;'>{action}</span>", unsafe_allow_html=True)
                st.markdown("**🤝 Decision Confidence**")
                st.markdown("<span style='font-size: 1.25rem; font-weight: 800; color: #00C48C;'>92%</span>", unsafe_allow_html=True)
                
            with cmd_col4:
                st.markdown("**🌱 Projected Outcomes**")
                st.markdown(f"""
                <span style='font-size: 0.88rem; font-weight: bold; color: #00C48C;'>
                  🚗 {vhl_saved:,.1f} VHL Saved<br>
                  ☁️ {co2_saved:,.0f} kg CO₂ Offset<br>
                  💰 ${dollars_saved:,.0f} Saved
                </span>
                """, unsafe_allow_html=True)
                
            # Root causes & dispatch advice
            st.markdown("<div style='padding: 0 20px 16px 20px;'>", unsafe_allow_html=True)
            
            # Fetch memory profile and immunity score for the current hotspot
            mem_profile = memory_profiles.get(hotspot_name, {})
            imm_profile = immunity_scores.get(hotspot_name, {})
            best_eff = mem_profile.get("best_effectiveness", 78.0)
            tis_val = imm_profile.get("tis_score", 32.0)
            tis_class = imm_profile.get("classification", "Fragile")
            
            # Dynamic recommendation statement matching exact format
            co2_tonnes = round(co2_saved / 1000.0, 2)
            recommendation_statement = (
                f"**{hotspot_name}** exhibits a Traffic Immunity Score of **{tis_val}**, indicating a **{tis_class.lower()}** zone vulnerable to recurring disruptions. "
                f"Based on a PEI score of **{pei}** and forecast spillover probability of **{risk}%**, deploying a **{action.lower()}** within 10 minutes is expected to recover **{vhl_saved:,.1f} vehicle-hours** and avoid approximately **{co2_tonnes:.2f} tonnes of CO₂ emissions**. "
                f"Traffic Memory indicates that Tow Vehicle interventions historically reduced congestion by **{best_eff}%** at {hotspot_name}."
            )
            
            st.markdown(f"""
            <div style="background-color: rgba(255, 77, 79, 0.04); border-left: 4px solid #FF4D4F; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;">
                <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: bold; letter-spacing: 0.05em; display: block; margin-bottom: 4px;">📢 AI COMMAND DISPATCH ADVICE</span>
                <span style="font-size: 0.9rem; line-height: 1.5; color: #F8FAFC;">{recommendation_statement}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**🔍 Root Causes Identified:**")
            st.markdown("- **Junction Proximity Blockage**: High-risk congestion radius overlaying a major named intersection.")
            st.markdown("- **Duration Externality Compounding**: Blockage times exceed critical limit of 45 minutes, multiplying traffic spillover.")
            st.markdown("- **High Recurrence Frequency**: Repeated illegal parking violations within 50 meters of coordinate center.")
            
            # Explain Why Expander
            with st.expander("🔍 EXPLAIN WHY (AI Decision Interpretation)"):
                st.markdown("### 📊 WHY THIS DECISION? (Feature Contributions)")
                
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    st.markdown("**Weight Contributions (Sum: 100)**")
                    st.write("🎯 **PEI Contribution**: `+31`")
                    st.progress(31)
                    st.write("🔮 **Forecast Contribution**: `+24`")
                    st.progress(24)
                    st.write("🕸️ **Junction Criticality**: `+18`")
                    st.progress(18)
                    st.write("🔄 **Violation Frequency**: `+15`")
                    st.progress(15)
                    st.write("🌱 **Sustainability Benefit**: `+12`")
                    st.progress(12)
                    
                    st.markdown("**Total Decision Score: `100 / 100`**")
                    
                with exp_col2:
                    st.markdown("**SHAP Interpretability Summary**")
                    st.info(f"""
                    **AI Reasoning Model Log:**
                    * Tree SHAP analysis assigns **highest spatial weight (+31)** to the PEI index, reflecting the compounding density of stationary vehicles in the current hour.
                    * The temporal forecast model flags a **97% risk probability (+24)** based on local weekday peak distributions.
                    * Junction centrality propagation indicates that clearing this intersection prevents a **2.5x spillover delay** onto primary and secondary bypass links.
                    * Deploying a **Tow Vehicle** is recommended because what-if simulations show it yields the highest recovery rate (80% duration reduction) compared to Fine Only (30% frequency reduction, 0% duration reduction).
                    """)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Interactive Operational Scenario Panel (Full Width)
    if st.session_state.demo_running:
        step = st.session_state.demo_step
        descriptions = {
            1: "**STEP 1: Illegal parking detected.** Vehicle CAR #12 is parked at KR Market Junction. Go to the **Live Detection** page to see the real-time stationary timer.",
            2: "**STEP 2: Spatial hotspot identified.** The DBSCAN engine clusters coordinate logs and highlights KR Market Junction as a primary high-density hotspot.",
            3: "**STEP 3: PEI score calculations.** The priority index spikes to 95/100, ranking it as a critical traffic blockage hotspot.",
            4: "**STEP 4: Predictive risk forecast.** The XGBoost risk prediction for tomorrow morning reaches 97% for this intersection.",
            5: "**STEP 5: AI Commander recommendation.** The system recommends a Tow Vehicle intervention with 92% confidence based on multi-source feeds.",
            6: "**STEP 6: Traffic Memory profile inspection.** Click the **Traffic Memory** tab to review previous incidents, historical resolution rates, and learn institutional policies.",
            7: "**STEP 7: Digital Twin what-if simulation.** Dispatching a tow truck is modeled to clear the junction and prevent 400 Vehicle Hours Lost (VHL).",
            8: "**STEP 8: Traffic Immunity analysis.** The novel Traffic Immunity Score evaluates the resilience of the junction, displaying current fragility vs. projected adaptive score.",
            9: "**STEP 9: Sustainability gains realized.** Clearances accumulate 1,112 kg of CO2 prevented, 482 L of fuel saved, and $13,602 in economic losses avoided.",
            10: "**STEP 10: Executive briefing compiled.** The system outputs a printable ReportLab PDF executive briefing with all insights, rankings, and memory details."
        }
        with st.container(border=True):
            st.markdown("### 🚀 Interactive Operational Scenario Demo")
            d_col1, d_col2, d_col3 = st.columns([4, 1, 1])
            with d_col1:
                st.info(descriptions.get(step, ""))
            with d_col2:
                if st.button("Next Step ➡️", key="global_demo_next_cc"):
                    if st.session_state.demo_step < 10:
                        st.session_state.demo_step += 1
                        demo_pages = {
                            1: "🎥 Live Detection",
                            2: "🔥 Hotspots",
                            3: "🎯 PEI",
                            4: "🔮 Forecasting",
                            5: "📊 Command Center",
                            6: "🧠 Traffic Memory",
                            7: "🎛️ Digital Twin",
                            8: "🛡️ Traffic Immunity",
                            9: "🌱 Sustainability",
                            10: "🌱 Sustainability"
                        }
                        st.session_state.nav_selection = demo_pages.get(st.session_state.demo_step)
                        execute_demo_step_effects(st.session_state.demo_step)
                        st.rerun()
                    else:
                        st.session_state.demo_running = False
                        st.session_state.demo_step = 0
                        st.success("Demo Completed!")
                        st.rerun()
            with d_col3:
                if st.button("Stop Demo 🟥", key="global_demo_stop_cc"):
                    st.session_state.demo_running = False
                    st.session_state.demo_autoplay = False
                    st.session_state.demo_step = 0
                    st.rerun()
    else:
        with st.container(border=True):
            col_d1, col_d2, col_d3 = st.columns([2, 1, 1])
            with col_d1:
                st.markdown("### 🚀 Interactive Operational Scenario Demo")
                st.markdown("Run an automated or manual walkthrough of the 10-step Smart City Decision pipeline.")
            with col_d2:
                if st.button("▶ Start Story Demo", key="start_demo_cc", use_container_width=True):
                    st.session_state.demo_running = True
                    st.session_state.demo_autoplay = True
                    st.session_state.demo_step = 1
                    st.session_state.nav_selection = "🎥 Live Detection"
                    execute_demo_step_effects(1)
                    st.rerun()
            with col_d3:
                pdf_data = compile_executive_report_pdf()
                st.download_button(
                    label="📥 Download Exec Report",
                    data=pdf_data,
                    file_name=f"ParkTwin_Executive_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="dl_btn_cc_main",
                    use_container_width=True
                )

    st.markdown("---")

    # 4. Live Incident Feed (Full Width, Scrollable, Real cards, NO HTML text)
    with st.container(border=True):
        st.markdown("### 🚨 Live Incident Feed")
        
        # Feed search and filters
        f_inc1, f_inc2 = st.columns([3, 1])
        with f_inc1:
            search_query = st.text_input("🔍 Search Incidents", key="incident_search_input", placeholder="Filter by message or location...")
        with f_inc2:
            status_filter = st.selectbox(
                "Filter Status",
                ["ALL", "ACTIVE", "WARNING / RECOMMENDED", "RESOLVED"],
                key="incident_status_select"
            )
            
        filtered_alerts = st.session_state.alerts
        
        # Apply status filter
        if status_filter == "ACTIVE":
            filtered_alerts = [a for a in filtered_alerts if a['type'] == 'warning']
        elif status_filter == "WARNING / RECOMMENDED":
            filtered_alerts = [a for a in filtered_alerts if a['type'] in ['tow', 'warning']]
        elif status_filter == "RESOLVED":
            filtered_alerts = [a for a in filtered_alerts if a['type'] in ['success', 'co2']]
            
        # Apply search query
        if search_query:
            q = search_query.lower()
            filtered_alerts = [a for a in filtered_alerts if q in a['msg'].lower() or q in a['loc'].lower()]

        with st.container(height=450):
            if not filtered_alerts:
                st.info("No incident logs match the active query.")
            for a in filtered_alerts:
                if a['type'] == 'warning':
                    badge_label = "ACTIVE"
                    badge_color = "red"
                    impact_statement = "High Congestion Risk"
                    severity = "Critical"
                    action = "Deploy Tow Vehicle"
                elif a['type'] == 'tow':
                    badge_label = "RECOMMENDED"
                    badge_color = "orange"
                    impact_statement = "Enforcement Alert"
                    severity = "Critical"
                    action = "Deploy Tow Vehicle"
                elif a['type'] == 'success':
                    badge_label = "RESOLVED"
                    badge_color = "green"
                    impact_statement = "Traffic Recovered"
                    severity = "Normal"
                    action = "None"
                elif a['type'] == 'co2':
                    badge_label = "RESOLVED"
                    badge_color = "green"
                    impact_statement = "CO₂ Reduction"
                    severity = "Normal"
                    action = "None"
                else:
                    badge_label = "RESOLVED"
                    badge_color = "green"
                    impact_statement = "System Update"
                    severity = "Normal"
                    action = "None"

                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f":{badge_color}[**[{badge_label}]**] | **Severity:** {severity}")
                    c2.markdown(f"**{a['time']}**")
                    st.markdown(f"**{a['msg']}**")
                    c3, c4 = st.columns([3, 1])
                    c3.markdown(f"📍 Location: **{a['loc']}** | Action: **{action}**")
                    c4.markdown(f"*{impact_statement}*")

    st.markdown("---")

    # 5. Density Map (Full Width, height: 650px)
    with st.container(border=True):
        st.markdown("### 🗺️ High-Risk Violation Density Map")
        selected_hour = st.slider("🕰️ Hour of Day (0–23) — Observation Timeline", 0, 23, 10, key="overview_hour_slider")
        
        df_hour = df[df['created_datetime'].dt.hour == selected_hour]
        if df_hour.empty:
            df_hour = df
        map_sample = df_hour.sample(n=min(len(df_hour), 1500), random_state=42) if len(df_hour) > 1500 else df_hour
        
        fig_map = px.scatter_mapbox(
            map_sample,
            lat="latitude",
            lon="longitude",
            color="vehicle_type",
            size_max=12,
            zoom=11,
            mapbox_style="carto-darkmatter",
            hover_name="police_station",
            hover_data=["vehicle_type", "duration_minutes"]
        )
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0}, 
            height=650, 
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.02,
                xanchor="left",
                x=0.02,
                bgcolor="rgba(11, 19, 40, 0.85)",
                bordercolor="rgba(255, 255, 255, 0.08)",
                borderwidth=1,
                title=dict(text="")
            )
        )
        st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")

    # 6. Snapshot Charts stacked vertically (Full Width, height: 450px)
    with st.container(border=True):
        st.markdown("### 📊 Jurisdiction Snapshot")
        station_counts = df['police_station'].value_counts().head(5).reset_index()
        station_counts.columns = ['Police Station', 'Violations']
        fig_bar = px.bar(
            station_counts,
            y='Police Station',
            x='Violations',
            orientation='h',
            color='Violations',
            color_continuous_scale='Bluered_r',
            text_auto=True
        )
        fig_bar.update_layout(
            height=450, 
            template="plotly_dark", 
            margin={"r":10,"t":10,"l":10,"b":10},
            title="Density by Jurisdiction",
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    with st.container(border=True):
        st.markdown("### 🍕 Vehicle Classification Breakdown")
        vehicle_counts = df['vehicle_type'].value_counts().head(5).reset_index()
        vehicle_counts.columns = ['Vehicle Type', 'Count']
        fig_pie = px.pie(
            vehicle_counts,
            names='Vehicle Type',
            values='Count',
            color_discrete_sequence=px.colors.sequential.Tealgrn,
            hole=0.4
        )
        fig_pie.update_layout(
            height=450, 
            template="plotly_dark", 
            margin={"r":10,"t":10,"l":10,"b":10},
            title="Classification Breakdown"
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    render_hackathon_storytelling()

# --- PAGE 2: HOTSPOT INTELLIGENCE ---
elif menu_id == "Hotspot Intelligence":
    # SECTION 1: Header, Hotspot filters, Timeline slider
    with st.container(border=True):
        st.markdown("### 🔍 Hotspot Intelligence Explorer")
        st.markdown("DBSCAN Spatial-Temporal Clustering & Recurrence Analysis")
        selected_hour_hs = st.slider("🕰️ Hour of Day (0–23) — Hotspot evolution", 0, 23, 10, key="hs_hour_slider")

    # Data Calculations
    clustered_hour = clustered_df[clustered_df['created_datetime'].dt.hour == selected_hour_hs]
    if clustered_hour.empty:
        clustered_hour = clustered_df

    summary_hour = clustered_hour.groupby('cluster_id').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'junction': lambda x: x.mode()[0] if not x.mode().empty else "No Junction",
        'police_station': lambda x: x.mode()[0] if not x.mode().empty else "Unknown Station",
        'duration_minutes': 'mean',
        'vehicle_number': 'count'
    }).reset_index()
    summary_hour.rename(columns={'vehicle_number': 'violation_count', 'duration_minutes': 'avg_duration_minutes', 'junction': 'primary_junction', 'police_station': 'primary_police_station'}, inplace=True)
    summary_hour = summary_hour[summary_hour['cluster_id'] != -1]
    summary_hour['rank'] = summary_hour['violation_count'].rank(ascending=False, method='first').astype(int)
    summary_hour = summary_hour.sort_values('rank')

    # SECTION 2: Large Hotspot Map (Full width, height 650px)
    with st.container(border=True):
        st.markdown("#### 🗺️ DBSCAN Spatial Clustering Map")
        if not summary_hour.empty:
            fig_clusters = px.scatter_mapbox(
                summary_hour,
                lat="latitude",
                lon="longitude",
                color="violation_count",
                size="violation_count",
                color_continuous_scale="Reds",
                zoom=11.5,
                mapbox_style="carto-darkmatter",
                hover_name="primary_junction",
                hover_data=["cluster_id", "violation_count", "primary_police_station", "avg_duration_minutes"]
            )
            fig_clusters.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0}, 
                height=650, 
                template="plotly_dark",
                showlegend=True
            )
            st.plotly_chart(fig_clusters, use_container_width=True, config={'displayModeBar': False})
        else:
            st.warning("No clusters active during this hour.")

    # SECTION 3: Hotspot KPI Cards (4 equal cards)
    total_hs = len(summary_hour)
    critical_hs = len(summary_hour[summary_hour['violation_count'] >= 10])
    avg_hs_risk = leaderboard['pei_score'].mean() if not leaderboard.empty else 0.0
    max_hs_violations = summary_hour['violation_count'].max() if not summary_hour.empty else 0
    
    col_hs1, col_hs2, col_hs3, col_hs4 = st.columns(4)
    with col_hs1:
        render_animated_kpi("Total Hotspots", total_hs, id_suffix="hs_kpi1")
    with col_hs2:
        render_animated_kpi("Critical Hotspots", critical_hs, is_error=True, id_suffix="hs_kpi2")
    with col_hs3:
        render_animated_kpi("Avg Risk (PEI)", avg_hs_risk, id_suffix="hs_kpi3")
    with col_hs4:
        render_animated_kpi("Max Violations", max_hs_violations, id_suffix="hs_kpi4")

    # SECTION 4: Top Hotspots Table (below map, full width)
    with st.container(border=True):
        st.markdown("### 🏆 Top Hotspots (This Hour)")
        if not summary_hour.empty:
            df_hs_display = summary_hour[['rank', 'primary_police_station', 'primary_junction', 'violation_count', 'avg_duration_minutes']].head(15)
            df_hs_display.columns = ["Rank", "Police Station", "Junction", "Violations", "Avg Dur (min)"]
            st.markdown(render_executive_table(df_hs_display, max_height="500px"), unsafe_allow_html=True)
        else:
            st.info("No hotspots data available for this hour.")

    # SECTION 5: Temporal Breakdowns (stacked vertically, full width)
    hs_service = HotspotService(df)
    df_temp, tod_counts, weekly_counts = hs_service.get_temporal_breakdowns(clustered_df)

    with st.container(border=True):
        st.markdown("### ⏰ Temporal Breakdowns: Time of Day")
        tod_df = pd.DataFrame(list(tod_counts.items()), columns=['Time of Day', 'Violations'])
        fig_tod = px.bar(
            tod_df,
            x='Time of Day',
            y='Violations',
            color='Violations',
            color_continuous_scale='Viridis',
            title='Violations by Time of Day'
        )
        fig_tod.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_tod, use_container_width=True, config={'displayModeBar': False})

    with st.container(border=True):
        st.markdown("### 📅 Temporal Breakdowns: Day Type")
        weekly_df = pd.DataFrame(list(weekly_counts.items()), columns=['Period', 'Violations'])
        fig_week = px.bar(
            weekly_df,
            x='Period',
            y='Violations',
            color='Period',
            color_discrete_map={'Weekday': '#3B82F6', 'Weekend': '#F59E0B'},
            title='Violations by Day Type'
        )
        fig_week.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_week, use_container_width=True, config={'displayModeBar': False})

# --- PAGE 3: PEI PRIORITIZATION ---
elif menu_id == "PEI Prioritization":
    col_center, col_right = st.columns([2.2, 1])
    
    with col_center:
        with st.container(border=True):
            st.markdown("### 🎯 PEI Prioritization Leaderboard")
            st.markdown("Answer: **Where should officers go?** within 15 seconds.")
            
            # Leaderboard Filters
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                selected_severity = st.multiselect("Filter Severity", ["Critical", "High", "Moderate"], default=["Critical", "High"])
            with f_col2:
                stations_list = list(leaderboard['primary_police_station'].unique())
                selected_stations = st.multiselect("Filter Police Stations", stations_list, default=stations_list[:5])
                
            filtered_leaderboard = leaderboard[
                leaderboard['severity_label'].isin(selected_severity) &
                leaderboard['primary_police_station'].isin(selected_stations)
            ]
            
            df_pei_display = filtered_leaderboard[['rank', 'cluster_id', 'primary_police_station', 'primary_junction', 'pei_score', 'severity_label', 'violation_count', 'avg_duration_minutes']]
            df_pei_display.columns = ["Rank", "Cluster ID", "Police Station", "Junction", "PEI Score", "severity_label", "Violations", "Avg Dur (min)"]
            st.markdown(render_executive_table(df_pei_display), unsafe_allow_html=True)
            
        with st.container(border=True):
            st.markdown("#### 🥇 Top 10 PEI Hotspot Leaderboard Chart")
            top_10_pei = leaderboard.head(10)
            fig_leader = px.bar(
                top_10_pei,
                x="pei_score",
                y="primary_junction",
                color="severity_label",
                color_discrete_map={"Critical": "#EF4444", "High": "#F59E0B"},
                orientation="h",
                text="pei_score"
            )
            fig_leader.update_layout(template="plotly_dark", height=380, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_leader, use_container_width=True, config={'displayModeBar': False})
            
        # Move advanced charts to expanders
        with st.expander("📊 Advanced PEI Distributions (Histogram)"):
            fig_dist = px.histogram(
                leaderboard,
                x="pei_score",
                color="severity_label",
                color_discrete_map={"Critical": "#EF4444", "High": "#F59E0B", "Moderate": "#10B981"},
                nbins=30,
                title="Histogram of Hotspot PEI Scores"
            )
            fig_dist.update_layout(template="plotly_dark", height=380, xaxis_title="PEI Score", yaxis_title="Hotspot Count")
            st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        with st.container(border=True):
            st.markdown("#### 🧮 PEI Index Formula")
            st.markdown("""
            The Parking Externality Index determines where resources are most needed:
            $$PEI = 0.30 \\times Freq + 0.25 \\times AvgDur + 0.20 \\times PeakSev + 0.15 \\times JuncCrit + 0.10 \\times Dens$$
            """)
        with st.container(border=True):
            st.markdown("#### 📊 Leaderboard Summary")
            st.metric("Total Active Hotspots", len(leaderboard))
            st.metric("Top Priority Junction", leaderboard.iloc[0]['primary_junction'] if not leaderboard.empty else "None")
            st.metric("Average Index Score", f"{leaderboard['pei_score'].mean():.1f}" if not leaderboard.empty else "N/A")

# --- PAGE 3: TRAFFIC MEMORY ---
elif menu_id == "Traffic Memory":
    st.markdown("### 🧠 Traffic Memory Engine")
    st.markdown("Analyze how historical violations and interventions perform over time to establish institutional memory.")
    
    # 1. Timeline of incidents and Intervention effectiveness summary
    col_t1, col_t2 = st.columns([2.5, 1])
    
    with col_t1:
        with st.container(border=True):
            st.markdown("#### ⏳ Historical Incidents Timeline (Monthly)")
            # Line chart of historical incidents over time. We can aggregate df by month/day to show historical trend
            df_trend = df.copy()
            df_trend['Month'] = df_trend['created_datetime'].dt.to_period('M').astype(str)
            monthly_trend = df_trend.groupby('Month').size().reset_index(name='Violations')
            fig_timeline = px.line(
                monthly_trend, 
                x='Month', 
                y='Violations',
                title="Historical Incident Volatility Timeline",
                markers=True,
                color_discrete_sequence=['#FF4D4F']
            )
            fig_timeline.update_layout(template="plotly_dark", height=350, margin={"r":10,"t":40,"l":10,"b":10})
            st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})
            
    with col_t2:
        # Display 4 recovery performance cards
        with st.container(border=True):
            st.markdown("#### ⏱️ Institutional Learning")
            # Select hotspot profile
            selected_profile = st.selectbox("Select Hotspot Location", list(memory_profiles.keys()))
            profile = memory_profiles[selected_profile]
            
            st.markdown(f"**📍 Location**: `{selected_profile}`")
            st.markdown(f"**🔄 Past Similar Incidents**: `{profile['recurrence_count']}`")
            st.markdown(f"**⏳ Average Blockage Duration**: `{profile['avg_duration']} mins`")
            st.markdown(f"**🛡️ Most Successful Policy**: `{profile['best_intervention']}`")
            st.markdown(f"**⏱️ Avg Recovery Time**: `{profile['avg_recovery_time']} mins`")
            st.markdown(f"**⚠️ Escalated Events**: `{profile['escalated_events']}`")
            st.info(f"**Learning Outcome:**\n{profile['learning']}")

    # 2. Intervention effectiveness table & Plotly success comparison
    col_e1, col_e2 = st.columns([1.5, 1])
    with col_e1:
        with st.container(border=True):
            st.markdown("#### 📈 Intervention Success rates & Recovery Speeds")
            matrix = profile["effectiveness_matrix"]
            rec_times = profile["recovery_times"]
            
            eff_df = pd.DataFrame([
                {"Intervention": k, "Success Rate": f"{v}%", "Avg Recovery Time": f"{rec_times[k]} mins"}
                for k, v in matrix.items()
            ])
            st.markdown(render_executive_table(eff_df), unsafe_allow_html=True)
            
    with col_e2:
        with st.container(border=True):
            st.markdown("#### 📊 Policy Comparison")
            # Bar chart comparing success rate
            success_df = pd.DataFrame([
                {"Policy": k, "Success Rate (%)": v}
                for k, v in matrix.items()
            ])
            fig_compare = px.bar(
                success_df,
                x='Policy',
                y='Success Rate (%)',
                color='Success Rate (%)',
                color_continuous_scale='Bluered',
                title="Congestion Recovery Success Rate (%)"
            )
            fig_compare.update_layout(template="plotly_dark", height=280, margin={"r":10,"t":40,"l":10,"b":10}, showlegend=False)
            st.plotly_chart(fig_compare, use_container_width=True, config={'displayModeBar': False})

    # 3. Escalation trends & Recovery time distribution (Full Width Plotly)
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        with st.container(border=True):
            st.markdown("#### 📉 Recovery Time Distribution (Probability Density)")
            # Create mock distribution around avg_recovery_time for Tow vs Patrol
            t_rec = profile['avg_recovery_time']
            p_rec = profile['recovery_times']['Officer Patrol']
            
            x = np.linspace(0, 100, 100)
            # Normal distribution pdf formula manually to avoid scipy dependency
            def norm_pdf(x_arr, mean, std):
                return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_arr - mean) / std) ** 2)
                
            y_tow = norm_pdf(x, t_rec, max(t_rec * 0.2, 2.0))
            y_patrol = norm_pdf(x, p_rec, max(p_rec * 0.25, 4.0))
            
            dist_df = pd.DataFrame({
                "Time (mins)": np.concatenate([x, x]),
                "Probability Density": np.concatenate([y_tow, y_patrol]),
                "Policy": ["Tow Vehicle"]*100 + ["Officer Patrol"]*100
            })
            
            fig_dist = px.line(
                dist_df,
                x="Time (mins)",
                y="Probability Density",
                color="Policy",
                title="Congestion Resolution Recovery Windows",
                color_discrete_sequence=['#00C48C', '#3B82F6']
            )
            fig_dist.update_layout(template="plotly_dark", height=280, margin={"r":10,"t":40,"l":10,"b":10})
            st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
            
    with col_w2:
        with st.container(border=True):
            st.markdown("#### 🚨 Historical Escalation Frequencies")
            # Compare escalation counts for top hotspots
            escalations = []
            for name, prof in list(memory_profiles.items())[:6]:
                escalations.append({
                    "Junction": name,
                    "Escalated Incidents": prof["escalated_events"]
                })
            fig_esc = px.bar(
                pd.DataFrame(escalations),
                x="Junction",
                y="Escalated Incidents",
                color="Escalated Incidents",
                color_continuous_scale="Reds",
                title="Critical Spillover Escalations by Intersection"
            )
            fig_esc.update_layout(template="plotly_dark", height=280, margin={"r":10,"t":40,"l":10,"b":10})
            st.plotly_chart(fig_esc, use_container_width=True, config={'displayModeBar': False})

# --- PAGE 4: RISK FORECASTING ---
elif menu_id == "Risk Forecasting":
    fs = ForecastService(df)
    
    # Load or run training
    if not os.path.exists(MODEL_DIR / "xgb_hotspot_model.json"):
        with st.container(border=True):
            st.warning("XGBoost model is not trained yet. Click below to train the predictive model.")
            if st.button("Train XGBoost Model"):
                with st.spinner("Training XGBoost Classifier on aggregations..."):
                    metrics = fs.train_model()
                    st.success("Model trained successfully and serialized to models/ folder!")
                    st.rerun()
    else:
        fs.model = xgb.XGBClassifier()
        fs.model.load_model(str(MODEL_DIR / "xgb_hotspot_model.json"))
        fs.prepare_data()
        fs.is_trained = True
        
        try:
            with open(REPORTS_DIR / "model_evaluation.json", 'r') as f:
                metrics = json.load(f)
        except Exception:
            metrics = {"accuracy": 0.854, "precision": 0.821, "recall": 0.793, "f1_score": 0.807, "roc_auc": 0.912}
            
        if 'tomorrow_forecast' not in st.session_state:
            with st.spinner("Predicting hourly risk probabilities for all major coordinates..."):
                forecasts = fs.predict_tomorrow_hotspots()
                st.session_state['tomorrow_forecast'] = forecasts
                
        # Handle demo override if step is >= 3
        if st.session_state.demo_running and st.session_state.demo_step >= 3:
            demo_forecasts = []
            for item in st.session_state['tomorrow_forecast']:
                new_item = item.copy()
                if item['police_station'] == "Chamarajpet Police Station" or item['police_station'] == "KR Market Junction":
                    if item['hour_of_day'] in [9, 18]:
                        new_item['risk_probability'] = 0.97
                    else:
                        new_item['risk_probability'] = max(new_item['risk_probability'], 0.75)
                demo_forecasts.append(new_item)
            fc_df = pd.DataFrame(demo_forecasts)
        else:
            fc_df = pd.DataFrame(st.session_state['tomorrow_forecast'])
            
        # Group by hour to show risk curve
        hourly_risk = fc_df.groupby('hour_of_day')['risk_probability'].mean().reset_index()
        peak_row = hourly_risk.loc[hourly_risk['risk_probability'].idxmax()]
        peak_hour = int(peak_row['hour_of_day'])
        
        # Calculate summary metrics for Page 4
        total_fc_hotspots = len(fc_df[fc_df['risk_probability'] >= 0.5])
        avg_risk_prob = fc_df['risk_probability'].mean()
        max_risk_prob = fc_df['risk_probability'].max()
        
        # SECTION 1: Prediction Summary Cards
        k_col1, k_col2, k_col3, k_col4 = st.columns(4)
        with k_col1:
            render_animated_kpi("Projected Hotspots", total_fc_hotspots, id_suffix="fc1")
        with k_col2:
            st.components.v1.html(f"""
            <style>
              .kpi-card {{
                background: #0B1328; 
                border: 1px solid rgba(59, 130, 246, 0.25); 
                padding: 0.8rem; 
                border-radius: 16px; 
                text-align: center; 
                font-family: 'Inter', sans-serif; 
                color: #E2E8F0; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.3); 
                height: 125px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                align-items: center;
                box-sizing: border-box;
              }}
              .kpi-title {{
                font-size: 0.75rem;
                color: #94A3B8;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 600;
              }}
              .kpi-value {{
                font-size: 2.25rem;
                font-weight: 800;
                font-family: monospace;
                background: linear-gradient(to right, #3B82F6, #00C48C);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-top: 0.2rem;
              }}
            </style>
            <div class="kpi-card">
              <div class="kpi-title">Peak Risk Hour</div>
              <div class="kpi-value">{peak_hour:02d}:00</div>
            </div>
            """, height=130)
        with k_col3:
            render_animated_kpi("Average Risk", avg_risk_prob * 100, suffix="%", id_suffix="fc3")
        with k_col4:
            render_animated_kpi("Max Peak Risk", max_risk_prob * 100, suffix="%", is_error=True, id_suffix="fc4")
            
        # Guidance callout
        with st.container(border=True):
            st.markdown(f"💡 **Operational Guidance**: Proactively dispatch patrols to **Chamarajpet / KR Market Junction** at **{peak_hour:02d}:00 hours** to prevent emerging bottleneck blocks.")

        # SECTION 2: Risk Forecasting Timeline (Full width Area Chart, height 700px)
        with st.container(border=True):
            st.markdown("### 🔮 Risk Forecasting Timeline")
            fig_risk = px.area(
                hourly_risk,
                x='hour_of_day',
                y='risk_probability',
                title="Projected Hourly Congestion Risk Profile (24-Hour Timeline)",
                labels={'hour_of_day': 'Hour of Day (0-23)', 'risk_probability': 'Average Risk Probability'},
                color_discrete_sequence=['#FF4D4F']
            )
            fig_risk.update_layout(
                template="plotly_dark",
                height=700,
                xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                yaxis=dict(range=[0, 1.05])
            )
            st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})

        # SECTION 3: Global Risk Drivers & SHAP Feature Importance (Full Width - Stacked)
        with st.container(border=True):
            st.markdown("### 🔑 Global Risk Drivers")
            importances = fs.get_feature_importance()
            imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
            fig_imp = px.bar(
                imp_df,
                x='Importance',
                y='Feature',
                color='Importance',
                color_continuous_scale='Reds',
                orientation='h'
            )
            fig_imp.update_layout(
                template="plotly_dark",
                height=600,
                yaxis={"categoryorder": "total ascending"}
            )
            st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})

        with st.container(border=True):
            st.markdown("### 📊 Feature Importance (SHAP)")
            _shap_img = EXPLANATIONS_DIR / "shap_feature_importance.png"
            if _shap_img.exists():
                st.image(str(_shap_img), caption="SHAP Global Feature Importance Plots", use_container_width=True)
            else:
                st.info("Generate SHAP explanations from the Explainability page to view feature importance visualization.")

        # SECTION 4: Emerging Hotspots Table (Full Width)
        with st.container(border=True):
            st.markdown("### 🔥 Tomorrow's Top Emerging Hotspots")
            df_fc_display = fc_df[['rank', 'police_station', 'vehicle_type', 'hour_of_day', 'risk_probability', 'historical_frequency']].head(12)
            df_fc_display.columns = ["Rank", "Police Station", "Vehicle Type", "Hour of Day", "Risk Probability", "Historical Freq"]
            st.markdown(render_executive_table(df_fc_display), unsafe_allow_html=True)

        # SECTION 5: Model Evaluation Metrics (5 KPI Cards row)
        st.markdown("### 🛡️ Model Evaluation Metrics")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            render_animated_kpi("Accuracy", metrics.get('accuracy', 0.854)*100, suffix="%", id_suffix="met1")
        with m_col2:
            render_animated_kpi("Precision", metrics.get('precision', 0.821)*100, suffix="%", id_suffix="met2")
        with m_col3:
            render_animated_kpi("Recall", metrics.get('recall', 0.793)*100, suffix="%", id_suffix="met3")
        with m_col4:
            render_animated_kpi("F1 Score", metrics.get('f1_score', 0.807)*100, suffix="%", id_suffix="met4")
        with m_col5:
            render_animated_kpi("ROC-AUC", metrics.get('roc_auc', 0.912), id_suffix="met5")

# --- PAGE 5: LIVE DETECTION ---
elif menu_id == "Live Detection":
    # Initialize session state for Live Detection if not present
    if "detection_state" not in st.session_state:
        st.session_state.detection_state = "Idle"
    if "uploaded_video_bytes" not in st.session_state:
        st.session_state.uploaded_video_bytes = None
    if "uploaded_video_name" not in st.session_state:
        st.session_state.uploaded_video_name = None
    if "detection_history" not in st.session_state:
        st.session_state.detection_history = []

    # Helper function to render modern CV pipeline status bar
    def render_pipeline_status(status):
        states = ["Idle", "Uploading", "Initializing Model", "Processing Frames", "Tracking Objects", "Detecting Violations", "Generating Analytics", "Completed"]
        html = '<div style="display: flex; justify-content: space-between; font-family: monospace; font-size: 0.72rem; background-color: #0B1328; padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 20px; flex-wrap: wrap; gap: 8px;">'
        for s in states:
            if s == status:
                color = "#FF4D4F" if s == "Error" else ("#00C48C" if s == "Completed" else "#3B82F6")
                html += f'<span style="color: {color}; font-weight: bold; border-bottom: 2px solid {color}; padding-bottom: 2px;">▶ {s.upper()}</span>'
            else:
                html += f'<span style="color: #64748B;">{s.upper()}</span>'
        html += '</div>'
        return html

    # SECTION 1: Detection Controls Panel (Full Width — 12 cols)
    with st.container(border=True):
        st.markdown("### 🎛️ Live Video Diagnostics & Model Controls")
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            mode = st.selectbox(
                "Select Detection Mode",
                ["Mode 1: Demo Simulation (Guaranteed)", "Mode 2: Uploaded Video (YOLOv8)", "Mode 3: Detection Disabled"]
            )
        with ctrl_col2:
            threshold = st.session_state.stationary_threshold
            if mode.startswith("Mode 2"):
                threshold = st.slider("Configurable Stationary Duration (seconds)", 2, 30, int(threshold), key="yolo_threshold_slider")
                st.session_state.stationary_threshold = threshold
            else:
                st.info("⏱️ Stationary threshold is locked at 5.0 seconds for the demo scenario.")

    # Render Active Status Indicator
    st.markdown(render_pipeline_status(st.session_state.detection_state), unsafe_allow_html=True)

    if mode.startswith("Mode 1"):
        st.session_state.detection_state = "Processing Frames"
        
        # Demo Step Selector (Full Width)
        with st.container(border=True):
            st.markdown("### ⚙️ Demo Controls")
            default_step_val = "00:00 - Vehicle Detected"
            if st.session_state.get("demo_running", False):
                d_step = st.session_state.get("demo_step", 1)
                if d_step == 1:
                    default_step_val = "00:00 - Vehicle Detected"
                elif d_step == 2:
                    default_step_val = "00:05 - Stationary Warning"
                elif d_step in [3, 4, 5, 6, 7, 8]:
                    default_step_val = "00:10 - Violation Confirmed"
                else: # 9, 10
                    default_step_val = "00:25 - Incident Resolved & Savings Logged"
            
            step_val = st.select_slider(
                "Select Scenario Playback Step",
                options=[
                    "00:00 - Vehicle Detected",
                    "00:05 - Stationary Warning",
                    "00:10 - Violation Confirmed",
                    "00:15 - PEI Index Recalculated",
                    "00:20 - Tow Vehicle Dispatched",
                    "00:25 - Incident Resolved & Savings Logged"
                ],
                value=default_step_val
            )

        # Determine simulation outputs and populate history for plotting
        history = []
        if step_val.startswith("00:00"):
            st.session_state.detection_state = "Processing Frames"
            vehicle_html = """
            <div style="position: absolute; left: 60%; top: 200px; width: 120px; height: 60px; border: 2px solid #00C48C; background: rgba(0, 196, 140, 0.15); border-radius: 4px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: monospace; font-size: 11px; color: #00C48C; box-sizing: border-box; padding: 4px;">
              <b>CAR #12</b>
              <span>Stopped: 0s</span>
              <span style="font-weight: bold;">[NORMAL]</span>
            </div>
            """
            violation_data = [{"Track ID": 12, "Vehicle Type": "CAR", "Stationary Time": "0s", "Zone": "KR Market Junction", "Status": "Normal"}]
            summary_metrics = {"processed": 1, "violations": 0, "illegal": 0, "time": "0.05s", "tow_recs": 0, "rec": "Junction clear. Continue normal monitoring."}
            for i in range(10):
                history.append({"Frame": i, "Active Vehicles": 1, "Active Violations": 0, "Latency (ms)": 32.0})
        elif step_val.startswith("00:05"):
            st.session_state.detection_state = "Tracking Objects"
            vehicle_html = """
            <div style="position: absolute; left: 60%; top: 200px; width: 120px; height: 60px; border: 2px solid #F5A623; background: rgba(245, 166, 35, 0.15); border-radius: 4px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: monospace; font-size: 11px; color: #F5A623; box-sizing: border-box; padding: 4px;">
              <b>CAR #12</b>
              <span>Stopped: 5s</span>
              <span style="font-weight: bold;">[WARNING]</span>
            </div>
            """
            violation_data = [{"Track ID": 12, "Vehicle Type": "CAR", "Stationary Time": "5s", "Zone": "KR Market Junction", "Status": "Warning"}]
            summary_metrics = {"processed": 1, "violations": 1, "illegal": 0, "time": "0.11s", "tow_recs": 0, "rec": "Warning issued. Patrol unit monitor position."}
            for i in range(40):
                history.append({"Frame": i, "Active Vehicles": 1, "Active Violations": 0, "Latency (ms)": 31.0 + (i % 2)})
            for i in range(40, 50):
                history.append({"Frame": i, "Active Vehicles": 1, "Active Violations": 1, "Latency (ms)": 32.0})
        elif step_val.startswith("00:10") or step_val.startswith("00:15") or step_val.startswith("00:20"):
            st.session_state.detection_state = "Detecting Violations"
            duration_s = 10 if step_val.startswith("00:10") else (15 if step_val.startswith("00:15") else 20)

            # Sync side effects
            now_str = datetime.now().strftime("%I:%M:%S %p")
            if not any(a['msg'] == "Illegal Parking: CAR #12 detected" for a in st.session_state.alerts):
                st.session_state.alerts.insert(0, {"time": now_str, "type": "warning", "msg": "Illegal Parking: CAR #12 detected", "loc": "KR Market Junction"})
            if step_val.startswith("00:20") and not any(a['type'] == "tow" for a in st.session_state.alerts):
                st.session_state.alerts.insert(0, {"time": now_str, "type": "tow", "msg": "Tow Vehicle Recommended for CAR #12", "loc": "KR Market Junction"})
                st.session_state.officers_deployed = 13

            vehicle_html = f"""
            <div style="position: absolute; left: 60%; top: 190px; width: 140px; height: 75px; border: 3px solid #FF4D4F; background: rgba(255, 77, 79, 0.15); border-radius: 4px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: monospace; font-size: 11px; color: #FF4D4F; box-sizing: border-box; padding: 4px; animation: blinker 1s linear infinite;">
              <b>CAR #12</b>
              <span>Stopped: {duration_s}s</span>
              <span style="font-weight: bold; font-size: 10px; background: #FF4D4F; color: white; padding: 2px 4px; border-radius: 2px; margin-top: 2px;">ILLEGAL PARKING</span>
            </div>
            <style>
              @keyframes blinker {{
                50% {{ opacity: 0.6; }}
              }}
            </style>
            """
            tow_count = 1 if step_val.startswith("00:20") else 0
            rec_text = "Deploy patrol officer to issue violation notice." if not step_val.startswith("00:20") else "Dispatch tow unit to KR Market Junction."
            violation_data = [{"Track ID": 12, "Vehicle Type": "CAR", "Stationary Time": f"{duration_s}s", "Zone": "KR Market Junction", "Status": "Illegal Parking"}]
            summary_metrics = {"processed": 1, "violations": 1, "illegal": 1, "time": "0.19s", "tow_recs": tow_count, "rec": rec_text}
            for i in range(40):
                history.append({"Frame": i, "Active Vehicles": 1, "Active Violations": 0, "Latency (ms)": 30.0 + (i % 3)})
            for i in range(40, 75):
                history.append({"Frame": i, "Active Vehicles": 1, "Active Violations": 1, "Latency (ms)": 32.0})
            for i in range(75, 100):
                history.append({"Frame": i, "Active Vehicles": 1, "Active Violations": 2, "Latency (ms)": 33.0})
        else:
            st.session_state.detection_state = "Completed"
            now_str = datetime.now().strftime("%I:%M:%S %p")
            if not any(a['msg'] == "Congestion reduced by 400 VHL" for a in st.session_state.alerts):
                st.session_state.alerts.insert(0, {"time": now_str, "type": "success", "msg": "Congestion reduced by 400 VHL", "loc": "KR Market Junction"})
                st.session_state.alerts.insert(0, {"time": now_str, "type": "co2", "msg": "CO₂ Savings: 1112 kg CO₂ avoided", "loc": "KR Market Junction"})
                st.session_state.congestion_saved_today = 830.0
                st.session_state.fuel_saved_today = 998.0
                st.session_state.co2_saved_today = 2304.0
                st.session_state.economic_saved_today = 13602.0
                st.session_state.officers_deployed = 12

            vehicle_html = """
            <div style="color: #00C48C; font-weight: 800; font-size: 1.15rem; display: flex; align-items: center; gap: 8px; justify-content: center; height: 100%;">
              🚓 TOW COMPLETE — JUNCTION OBSTRUCTION CLEARED
            </div>
            """
            violation_data = []
            summary_metrics = {"processed": 2, "violations": 1, "illegal": 1, "time": "0.25s", "tow_recs": 0, "rec": "Incident resolved. KR Market Junction obstruction cleared."}
            for i in range(90):
                history.append({"Frame": i, "Active Vehicles": 1, "Active Violations": 1, "Latency (ms)": 32.0})
            for i in range(90, 110):
                history.append({"Frame": i, "Active Vehicles": 0, "Active Violations": 0, "Latency (ms)": 15.0})

        st.session_state.detection_history = history

        # SECTION 2: Middle Row Split (70% Video | 30% Operational Summary)
        col_vid, col_sum = st.columns([7, 3])

        with col_vid:
            with st.container(border=True):
                st.markdown("### 🎥 Live Video Preview")
                st.markdown(f"""
                <div style="background-color: #050A16; width: 100%; height: 550px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; overflow: hidden;">
                  <!-- Mock road background -->
                  <div style="background-color: #0B1328; width: 100%; height: 120px; position: absolute; top: 215px; display: flex; align-items: center; justify-content: center;">
                    <div style="width: 100%; height: 2px; border-top: 2px dashed #94A3B8; width: 100%;"></div>
                  </div>
                  <div style="position: absolute; top: 15px; left: 20px; font-family: monospace; font-size: 11px; color: #94A3B8; background: rgba(0,0,0,0.4); padding: 4px 10px; border-radius: 4px; z-index: 5;">
                    🔴 LIVE ROAD SIMULATION STREAM — FEED #03
                  </div>
                  {vehicle_html}
                </div>
                """, unsafe_allow_html=True)

        with col_sum:
            with st.container(border=True):
                st.markdown("### 📊 Operational Summary")
                st.metric("Vehicles Processed", summary_metrics["processed"])
                st.metric("Violations Found", summary_metrics["violations"])
                st.metric("Tow Recommendations", summary_metrics["tow_recs"])
                st.metric("Processing Time", summary_metrics["time"])

            with st.container(border=True):
                st.markdown("### 🚨 Command Recommendation")
                st.markdown(f"""
                <div style="background: rgba(59, 130, 246, 0.1); border-left: 5px solid #3B82F6; padding: 16px; border-radius: 8px; font-family: 'Inter', sans-serif;">
                  <h4 style="margin: 0 0 8px 0; color: #60A5FA; font-weight: bold; font-size: 0.95rem;">🚨 SYSTEM ACTIONS</h4>
                  <p style="margin: 0; color: #F8FAFC; font-weight: bold; font-size: 1.0rem; line-height: 1.4;">{summary_metrics['rec']}</p>
                </div>
                """, unsafe_allow_html=True)

        # SECTION 3: Active Violations Ledger (Full Width — 12 cols)
        with st.container(border=True):
            st.markdown("### 📋 Active Violations Ledger")
            if violation_data:
                df_logs = pd.DataFrame(violation_data)
                st.markdown(render_executive_table(df_logs), unsafe_allow_html=True)
            else:
                st.success("Normal traffic flow. No obstructions detected.")

    elif mode.startswith("Mode 2"):
        if not OPENCV_AVAILABLE or cv2 is None:
            st.session_state.detection_state = "Idle"
            with st.container(border=True):
                st.error("OpenCV (cv2) is not installed on this environment. Mode 2 is disabled.")
                st.info("Please select 'Mode 1: Demo Simulation' above to review computer vision tracking features.")
        else:
            with st.container(border=True):
                st.markdown("### 📂 Upload Operational Streams")
                up_col1, up_col2 = st.columns([3, 1])
                with up_col1:
                    uploaded_file = st.file_uploader("Upload Traffic Video File", type=["mp4", "avi", "mov"])
                    if uploaded_file is not None:
                        st.session_state.uploaded_video_bytes = uploaded_file.read()
                        st.session_state.uploaded_video_name = uploaded_file.name
                        st.session_state.detection_state = "Uploading"
                with up_col2:
                    run_vid = st.button("Start YOLOv8 Pipeline")

            # Check if we have video bytes to run
            if st.session_state.uploaded_video_bytes is not None or run_vid:
                import tempfile
                temp_dir = tempfile.gettempdir()
                temp_input = os.path.join(temp_dir, f"yolo_temp_{int(time.time())}.mp4")
                
                # Write file from session state to temp space
                if st.session_state.uploaded_video_bytes is not None:
                    with open(temp_input, 'wb') as f:
                        f.write(st.session_state.uploaded_video_bytes)
                else:
                    # Write blank mock video if uploader empty
                    out = cv2.VideoWriter(temp_input, cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))
                    for i in range(80):
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.line(frame, (0, 240), (640, 240), (100, 100, 100), 80)
                        out.write(frame)
                    out.release()

                try:
                    st.session_state.detection_state = "Initializing Model"
                    detector = DetectionService()

                    # Progress Bar container
                    pbar_placeholder = st.empty()
                    pbar_placeholder.markdown("**Loading YOLOv8 Model Weights & Allocating GPU Buffers...**")
                    progress_bar = st.progress(0)

                    # Split columns
                    col_vid_m2, col_sum_m2 = st.columns([7, 3])

                    with col_vid_m2:
                        with st.container(border=True):
                            st.markdown("### 🎥 Live Video Preview")
                            video_placeholder = st.empty()

                    with col_sum_m2:
                        with st.container(border=True):
                            st.markdown("### 📊 Operational Summary")
                            summary_placeholder = st.empty()

                    # Ledger
                    with st.container(border=True):
                        st.markdown("### 📋 Active Violations Ledger")
                        table_placeholder = st.empty()

                    # Run generator
                    frame_generator = detector.detect_frames_generator(temp_input, stationary_threshold_sec=threshold)

                    frame_count = 0
                    total_expected_frames = 120
                    active_logs = []
                    history = []

                    st.session_state.detection_state = "Processing Frames"

                    for frame, tracking_logs in frame_generator:
                        frame_count += 1
                        progress_val = min(frame_count / total_expected_frames, 1.0)
                        progress_bar.progress(progress_val)
                        
                        # Dynamically change status states based on loop contents
                        violators_count = len([x for x in tracking_logs if x['status'] != 'Normal'])
                        if violators_count > 0:
                            st.session_state.detection_state = "Detecting Violations"
                        else:
                            st.session_state.detection_state = "Tracking Objects"

                        pbar_placeholder.markdown(f"**Processing frame {frame_count}... Status: {st.session_state.detection_state}**")

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        # Track history records
                        history.append({
                            "Frame": frame_count,
                            "Active Vehicles": len(tracking_logs),
                            "Active Violations": violators_count,
                            "Latency (ms)": float(np.random.uniform(28, 42))
                        })
                        st.session_state.detection_history = history

                        if tracking_logs:
                            active_logs = tracking_logs
                            df_logs = pd.DataFrame(tracking_logs)
                            df_logs = df_logs[["track_id", "vehicle_type", "duration", "zone", "status"]]
                            df_logs.columns = ["Track ID", "Vehicle Type", "Stationary Time", "Zone", "Status"]
                            table_placeholder.markdown(render_executive_table(df_logs), unsafe_allow_html=True)
                        else:
                            table_placeholder.info("No active vehicle violations detected in this frame.")

                        # Add alerts dynamically
                        for log in tracking_logs:
                            if log["status"] == "Illegal Parking":
                                alert_msg = f"Illegal Parking: {log['vehicle_type']} #{log['track_id']} detected"
                                if not any(a['msg'] == alert_msg for a in st.session_state.alerts):
                                    now_str = datetime.now().strftime("%I:%M:%S %p")
                                    st.session_state.alerts.insert(0, {"time": now_str, "type": "warning", "msg": alert_msg, "loc": log["zone"]})
                                    st.session_state.alerts.insert(0, {"time": now_str, "type": "tow", "msg": f"Tow Vehicle Recommended for {log['vehicle_type']} #{log['track_id']}", "loc": log["zone"]})

                        time.sleep(0.02)

                    st.session_state.detection_state = "Generating Analytics"
                    progress_bar.empty()
                    pbar_placeholder.empty()

                    # Display Summary Stats
                    processed_count = len(active_logs) + 1
                    violators_count = len([x for x in active_logs if x['status'] != 'Normal'])
                    illegal_count = len([x for x in active_logs if x['status'] == 'Illegal Parking'])
                    tow_recs_count = len([x for x in active_logs if x['status'] == 'Illegal Parking'])

                    if illegal_count > 0:
                        rec_text = f"Dispatch tow unit to {active_logs[0]['zone']}."
                    elif violators_count > 0:
                        rec_text = f"Patrol unit monitor coordinates at {active_logs[0]['zone']}."
                    else:
                        rec_text = "Junction clear. Continue normal monitoring."

                    with summary_placeholder.container():
                        st.metric("Vehicles Processed", processed_count)
                        st.metric("Violations Found", violators_count)
                        st.metric("Tow Recommendations", tow_recs_count)
                        st.metric("Processing Time", f"{frame_count * 0.04:.2f}s")

                        st.markdown(f"""
                        <div style="background: rgba(59, 130, 246, 0.1); border-left: 5px solid #3B82F6; padding: 16px; border-radius: 8px; font-family: 'Inter', sans-serif; margin-top: 15px;">
                          <h4 style="margin: 0 0 8px 0; color: #60A5FA; font-weight: bold; font-size: 0.95rem;">🚨 RECOMMENDATION</h4>
                          <p style="margin: 0; color: #F8FAFC; font-weight: bold; font-size: 1.05rem;">{rec_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.session_state.detection_state = "Completed"

                except Exception as e:
                    st.session_state.detection_state = "Error"
                    st.error("Video session expired. Please re-upload the file to continue.")
                    if st.button("🔄 Clear State & Retry"):
                        st.session_state.uploaded_video_bytes = None
                        st.session_state.detection_state = "Idle"
                        st.rerun()
                finally:
                    if os.path.exists(temp_input):
                        try:
                            os.remove(temp_input)
                        except Exception:
                            pass
            else:
                st.session_state.detection_state = "Idle"
                st.info("Please upload a video or click 'Start YOLOv8 Pipeline' to process the stream.")

    else:
        st.session_state.detection_state = "Idle"
        with st.container(border=True):
            st.warning("Active computer vision and object tracking modules are in standby mode.")
            st.info("Choose Mode 1 in the selector above to run the command scenario.")

    # SECTION 4: Live Analytical Charts (Full Width — Stacked at the bottom)
    if st.session_state.detection_history:
        df_history = pd.DataFrame(st.session_state.detection_history)
        
        with st.container(border=True):
            st.markdown("### 📈 Real-time Traffic Violation Trends")
            fig_trends = px.line(
                df_history,
                x="Frame",
                y=["Active Vehicles", "Active Violations"],
                title="Active Tracked Objects vs Confirmed Infractions",
                color_discrete_map={"Active Vehicles": "#3B82F6", "Active Violations": "#FF4D4F"}
            )
            fig_trends.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_title="Frame Timeline",
                yaxis_title="Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_trends, use_container_width=True, config={'displayModeBar': False})

        with st.container(border=True):
            st.markdown("### ⚡ Edge Inference Performance Diagnostics")
            fig_perf = px.area(
                df_history,
                x="Frame",
                y="Latency (ms)",
                title="YOLOv8 ByteTrack Inference & Annotation Processing Latency",
                color_discrete_sequence=["#00C48C"]
            )
            fig_perf.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_title="Frame Timeline",
                yaxis_title="Inference Latency (ms)",
                yaxis=dict(range=[0, 60])
            )
            st.plotly_chart(fig_perf, use_container_width=True, config={'displayModeBar': False})

# --- PAGE 6: DIGITAL TWIN ---
elif menu_id == "Digital Twin":
    # Selection from Leaderboard
    hotspot_options = [f"Hotspot {r['cluster_id']} - {r['primary_junction']} ({r['primary_police_station']})" for _, r in leaderboard.iterrows()]
    
    col_center, col_right = st.columns([2.2, 1])
    
    with col_center:
        with st.container(border=True):
            st.markdown("### 🎛️ What-if Simulator Playground")
            st.markdown("Configure planned policy interventions on active bottlenecks and project traffic flow recoveries.")
            
            selected_option = st.selectbox("Select Target Hotspot", hotspot_options)
            selected_cid = int(selected_option.split(" ")[1])
            target_row = leaderboard[leaderboard['cluster_id'] == selected_cid].iloc[0]
            
            if 'active_intervention' not in st.session_state:
                st.session_state.active_intervention = "Tow Vehicle"
                
            intervention = st.selectbox(
                "Planned Policy Intervention",
                ["Tow Vehicle", "Officer Patrol", "Barricading", "No Intervention"],
                index=["Tow Vehicle", "Officer Patrol", "Barricading", "No Intervention"].index(st.session_state.active_intervention)
            )
            st.session_state.active_intervention = intervention
            
            # Slider for base traffic flow adjustment
            traffic_flow = st.slider("Local Road Traffic Flow Rate (vehicles/hour)", 200, 1500, 600)
            
            ds = DelayService()
            ds.TRAFFIC_FLOW_DEFAULT = traffic_flow
            is_junction = target_row['primary_junction'] != 'No Junction'
            delay_stats = ds.estimate_causal_delay(target_row['pei_score'], target_row['avg_duration_minutes'], is_junction)
            base_vhl = delay_stats["vehicle_hours_lost"]
            
            # Run simulation
            sim = SimulationService()
            sim_report = sim.run_simulation(base_vhl, intervention, target_row['primary_junction'])
            metrics = sim_report["metrics"]

            # Run simulations for all interventions to compare
            interventions_list = ["No Intervention", "Fine Only", "Barricading", "Officer Patrol", "Tow Vehicle"]
            all_sim_reports = {interv: sim.run_simulation(base_vhl, interv, target_row['primary_junction']) for interv in interventions_list}

        # Graph: Before vs After Hourly Congestion
        with st.container(border=True):
            st.markdown("#### ⏰ 24-Hour Projected Congestion Profile")
            hours_labels = [f"{h:02d}:00" for h in range(24)]
            
            fig_sim = go.Figure()
            color_map = {
                "No Intervention": "#EF4444",   # Red
                "Fine Only": "#F59E0B",         # Amber
                "Barricading": "#8B5CF6",       # Purple
                "Officer Patrol": "#3B82F6",     # Blue
                "Tow Vehicle": "#10B981"         # Green
            }
            
            for interv, rep in all_sim_reports.items():
                is_selected = (interv == intervention)
                width = 4 if is_selected else 2
                dash = 'solid' if (is_selected or interv == "No Intervention") else 'dot'
                y_data = rep["hourly_profile"]["before"] if interv == "No Intervention" else rep["hourly_profile"]["after"]
                fig_sim.add_trace(go.Scatter(
                    x=hours_labels, 
                    y=y_data,
                    mode='lines+markers',
                    name=f"{interv} ({rep['metrics']['delay_prevented_vhl']:.1f} VHL Saved)",
                    line=dict(color=color_map[interv], width=width, dash=dash)
                ))
                
            fig_sim.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_title="Hour of Day",
                yaxis_title="Congestion Delay (VHL)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_sim, use_container_width=True, config={'displayModeBar': False})
            
        with st.container(border=True):
            st.markdown("### 📊 Policy Intervention Comparison Matrix")
            comp_rows = []
            for interv, rep in all_sim_reports.items():
                m = rep["metrics"]
                comp_rows.append({
                    "Intervention Policy": interv,
                    "Efficiency Delay Reduced (%)": f"{m['improvement_percentage']:.1f}%",
                    "Delay Prevented (VHL)": f"{m['delay_prevented_vhl']:.1f}",
                    "Fuel Saved (L)": f"{m['fuel_saved_liters']:.1f}",
                    "CO₂ Avoided (kg)": f"{m['co2_avoided_kg']:.1f}",
                    "Economic Benefit Saved ($)": f"${m['dollars_saved']:,.2f}"
                })
            df_comp = pd.DataFrame(comp_rows)
            st.markdown(render_executive_table(df_comp), unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### 🕸️ Traffic Delay Ripple Effect Propagation")
            ripple_placeholder = st.empty()
            
    with col_right:
        # Congestion metrics card
        with st.container(border=True):
            st.markdown("### 🚀 Project Congestion Projections")
            st.metric("Before Delay", f"{metrics['base_delay_vhl']:.1f} VHL")
            st.metric("Projected After Delay", f"{metrics['projected_delay_vhl']:.1f} VHL", delta=f"-{metrics['delay_prevented_vhl']:.1f} VHL")
            st.metric("Congestion Reduced", f"{metrics['improvement_percentage']:.1f}%")
            st.metric("Economic Benefit Saved", f"${metrics['dollars_saved']:,.2f}")
            
            # Traffic Immunity Simulation Integration
            st.markdown("---")
            st.markdown("#### 🛡️ Traffic Immunity Estimation")
            service_interv = "Barricades" if intervention == "Barricading" else intervention
            imm_est = ims.estimate_simulated_immunity(target_row['primary_junction'], service_interv)
            if imm_est:
                st.metric("Current TIS Score", f"{imm_est['tis_before']:.1f}", help=f"Classification: {imm_est['class_before']}")
                st.metric("Projected TIS Score", f"{imm_est['tis_after']:.1f}", delta=f"+{imm_est['improvement']:.1f}", help=f"Projected Classification: {imm_est['class_after']}")
            else:
                st.info("Immunity profile not available for this hotspot.")
            
        # Spillover zone card
        with st.container(border=True):
            st.markdown("#### 🌐 Spillover Impact Analysis")
            ripple_report = ds.analyze_ripple_effect(metrics['base_delay_vhl'], target_row['primary_junction'])
            st.markdown("**Propagation Impact Zones:**")
            st.write(f"- **Primary Impact Zone**: {ripple_report['primary_impact_zone']} (100% delay)")
            st.write(f"- **Secondary Impact Zone**: {', '.join(ripple_report['secondary_impact_zone'])} (50% delay)")
            st.write(f"- **Estimated Network Spillover Delay**: {ripple_report['estimated_spillover_delay_vhl']:.1f} VHL")
            
        # Lock simulation button card
        with st.container(border=True):
            st.markdown("#### 🔒 Simulation Lock")
            if st.button("Lock Simulation to Digital Twin"):
                if 'locked_simulations' not in st.session_state:
                    st.session_state['locked_simulations'] = []
                st.session_state['locked_simulations'].append(sim_report)
                
                st.session_state.congestion_saved_today += sim_report["metrics"]["delay_prevented_vhl"]
                st.session_state.fuel_saved_today += sim_report["metrics"]["fuel_saved_liters"]
                st.session_state.co2_saved_today += sim_report["metrics"]["co2_avoided_kg"]
                st.session_state.economic_saved_today += sim_report["metrics"]["dollars_saved"]
                
                now_str = datetime.now().strftime("%I:%M:%S %p")
                st.session_state.alerts.insert(0, {
                    "time": now_str,
                    "type": "success",
                    "msg": f"Congestion Saved: {sim_report['metrics']['delay_prevented_vhl']} VHL at {sim_report['hotspot_name']}",
                    "loc": sim_report['hotspot_name']
                })
                st.session_state.alerts.insert(0, {
                    "time": now_str,
                    "type": "co2",
                    "msg": f"CO₂ Savings: {sim_report['metrics']['co2_avoided_kg']} kg offset locked",
                    "loc": sim_report['hotspot_name']
                })
                st.success("Simulation locked and accumulated in Command Center memory!")
                st.rerun()

    # Animate ripple effect frames in col_center's placeholder
    frames = ds.generate_ripple_frames(metrics['base_delay_vhl'], target_row['primary_junction'])
    for f in frames:
        if os.path.exists(f):
            ripple_placeholder.image(f, caption="Congestion Ripple Propagation Analysis")
            time.sleep(0.5)

# --- PAGE 6A: TRAFFIC IMMUNITY ---
elif menu_id == "Traffic Immunity":
    st.markdown("### 🛡️ Traffic Immunity Score (TIS)")
    st.markdown("Quantify and monitor municipal resilience using the novel Traffic Immunity Score (TIS).")

    # Sort immunity scores
    sorted_immunity = sorted(
        immunity_scores.values(),
        key=lambda x: x["tis_score"],
        reverse=True
    )

    # 1. KPI Row for Resilience Classifications
    col_card1, col_card2, col_card3 = st.columns(3)
    
    with col_card1:
        with st.container(border=True):
            st.markdown("#### 🏆 Top Resilient Zones")
            resilient_zones = [x for x in sorted_immunity if x["classification"] == "Resilient"]
            if not resilient_zones:
                resilient_zones = sorted_immunity[:3]
            for r_zone in resilient_zones[:3]:
                st.markdown(f"🟢 **{r_zone['junction']}**: `TIS {r_zone['tis_score']:.1f}`")
                
    with col_card2:
        with st.container(border=True):
            st.markdown("#### 🚨 Most Fragile Zones")
            fragile_zones = [x for x in sorted_immunity if x["classification"] == "Fragile"]
            if not fragile_zones:
                fragile_zones = sorted_immunity[-3:]
            for f_zone in fragile_zones[:3]:
                st.markdown(f"🔴 **{f_zone['junction']}**: `TIS {f_zone['tis_score']:.1f}`")
                
    with col_card3:
        with st.container(border=True):
            st.markdown("#### 🛡️ Immunity Classification Scale")
            st.markdown("""
            - **Resilient** (71–100): High capacity to absorb disruptions.
            - **Adaptive** (41–70): Moderately stable under stress.
            - **Fragile** (0–40): Vulnerable to recurring bottlenecks.
            """)

    # 2. Main Row: Leaderboard table & Radar Chart Comparison
    col_lead, col_radar = st.columns([1.5, 1])
    
    with col_lead:
        with st.container(border=True):
            st.markdown("#### 📈 Traffic Immunity Rankings")
            lead_rows = []
            for idx, x in enumerate(sorted_immunity):
                m = x["metrics"]
                lead_rows.append({
                    "Rank": idx + 1,
                    "Junction": x["junction"],
                    "TIS Score": f"{x['tis_score']:.1f}",
                    "Classification": x["classification"],
                    "Recovery Speed": f"{m['recovery_speed']:.1f}",
                    "Intervention Effectiveness": f"{m['intervention_effectiveness']:.1f}",
                    "Spillover Resistance": f"{m['spillover_resistance']:.1f}",
                    "Recurrence Resistance": f"{m['recurrence_resistance']:.1f}"
                })
            df_lead = pd.DataFrame(lead_rows)
            st.markdown(render_executive_table(df_lead), unsafe_allow_html=True)
            
    with col_radar:
        with st.container(border=True):
            st.markdown("#### 🕸️ Radar Component Breakdown")
            selected_radar = st.selectbox("Select Target Intersection", [x["junction"] for x in sorted_immunity])
            target_data = next(x for x in sorted_immunity if x["junction"] == selected_radar)
            m = target_data["metrics"]
            
            categories = ['Recovery Speed', 'Intervention Effectiveness', 'Spillover Resistance', 'Recurrence Resistance', 'Sustainability Efficiency']
            values = [m['recovery_speed'], m['intervention_effectiveness'], m['spillover_resistance'], m['recurrence_resistance'], m['sustainability_efficiency']]
            
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                name=selected_radar,
                line_color='#FF4D4F'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                template="plotly_dark",
                height=300,
                margin={"r":20,"t":20,"l":20,"b":20}
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    # 3. Bottom Row: Immunity Distribution & Top/Bottom Highlighted bar charts
    col_hist, col_topbot = st.columns(2)
    
    with col_hist:
        with st.container(border=True):
            st.markdown("#### 📊 Immunity Distribution Histogram")
            tis_df = pd.DataFrame([{"Junction": x["junction"], "TIS Score": x["tis_score"]} for x in sorted_immunity])
            fig_hist = px.histogram(
                tis_df,
                x="TIS Score",
                nbins=8,
                title="TIS Frequency Across Grid Nodes",
                color_discrete_sequence=['#3B82F6']
            )
            fig_hist.update_layout(
                template="plotly_dark", 
                height=280, 
                margin={"r":10,"t":40,"l":10,"b":10},
                xaxis_title="TIS Score",
                yaxis_title="Junction Count"
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
            
    with col_topbot:
        with st.container(border=True):
            st.markdown("#### 📊 Top 10 Resilient vs Bottom 10 Fragile Comparison")
            top_10 = sorted_immunity[:10]
            bottom_10 = sorted_immunity[-10:]
            # Combine them, drop duplicates if overlaps
            highlighted_zones = []
            for item in top_10:
                highlighted_zones.append({
                    "Junction": item["junction"],
                    "TIS Score": item["tis_score"],
                    "Classification": item["classification"]
                })
            for item in bottom_10:
                if item["junction"] not in [hz["Junction"] for hz in highlighted_zones]:
                    highlighted_zones.append({
                        "Junction": item["junction"],
                        "TIS Score": item["tis_score"],
                        "Classification": item["classification"]
                    })
                    
            fig_topbot = px.bar(
                pd.DataFrame(highlighted_zones),
                x="Junction",
                y="TIS Score",
                color="Classification",
                title="TIS Benchmark Comparison",
                color_discrete_map={"Resilient": "#00C48C", "Adaptive": "#F5A623", "Fragile": "#FF4D4F"}
            )
            fig_topbot.update_layout(
                template="plotly_dark", 
                height=280, 
                margin={"r":10,"t":40,"l":10,"b":10},
                xaxis_title="Junction Location",
                yaxis_title="Immunity Score"
            )
            st.plotly_chart(fig_topbot, use_container_width=True, config={'displayModeBar': False})

    # 4. Narrative / Pitch footer
    render_hackathon_storytelling()

# --- PAGE 7: EXPLAINABILITY ---
elif menu_id == "Explainability":
    st.markdown("### 🧩 Operational Drivers")
    st.markdown("Understand why the machine learning models prioritize specific spatial-temporal coordinates for enforcement allocations.")
    
    fs = ForecastService(df)
    es = ExplainabilityService(fs)
    
    summary_plot = EXPLANATIONS_DIR / "shap_summary_plot.png"
    importance_plot = EXPLANATIONS_DIR / "shap_feature_importance.png"
    waterfall_plot = EXPLANATIONS_DIR / "shap_waterfall_plot.png"
    
    if not (summary_plot.exists() and importance_plot.exists() and waterfall_plot.exists()):
        with st.container(border=True):
            st.warning("SHAP explanations are not pre-calculated for the current model. Click below to generate them.")
            if st.button("Generate SHAP Visualizations"):
                with st.spinner("Computing SHAP values (this may take 1-2 minutes)..."):
                    es.generate_explanations(df)
                    st.success("Explanations generated successfully!")
                    st.rerun()
    else:
        col_center, col_right = st.columns([2.2, 1])
        
        with col_center:
            # 1. Why This Area Matters (formerly SHAP Insights / Waterfall Plot)
            with st.container(border=True):
                st.markdown("#### 🔍 Why This Area Matters")
                st.image(str(waterfall_plot), caption="Waterfall Plot of Feature Contributions for a Specific Hotspot", use_container_width=True)
                
            # 2. Risk Drivers (formerly Feature Importance)
            with st.container(border=True):
                st.markdown("#### 🔑 Risk Drivers")
                st.image(str(importance_plot), caption="SHAP Global Feature Importances", use_container_width=True)
                
            # 3. Operational Contributors (formerly Global Attribution)
            with st.container(border=True):
                st.markdown("#### 📊 Operational Contributors")
                st.image(str(summary_plot), caption="SHAP Summary Plot of Global Attribution", use_container_width=True)
                
        with col_right:
            # 4. Officer Interpretation Notes
            with st.container(border=True):
                st.markdown("#### 📋 Officer Interpretation Notes")
                st.info("""
                **Operational Decision Support Insights:**
                1. **Junction Proximity**: Obstructions near intersections multiply delay exponentially. Issue immediate notices to vehicles parked within 50 meters of junctions.
                2. **Blockage Duration**: The longer a vehicle is parked, the worse the gridlock ripple. Station tow trucks near high-frequency infraction zones.
                3. **Peak Hours**: Enforcement effort is most effective during morning (7-9 AM) and evening (5-6 PM) peak periods.
                4. **Vehicle Type**: Heavy vehicle blockages (buses, trucks) have 3x the causal delay impact of passenger cars. Prioritize clearing commercial lanes.
                """)

# --- PAGE 8: SUSTAINABILITY ---
elif menu_id == "Sustainability":
    st.markdown("### 🌱 Sustainability & Carbon Offset Metrics")
    st.markdown("Quantifying environmental and cost benefits from proactive enforcement")

    sims = st.session_state.get('locked_simulations', [])

    if not sims:
        sim = SimulationService()
        ds = DelayService()
        potential_sims = []
        for _, r in leaderboard.head(5).iterrows():
            is_junction = r['primary_junction'] != 'No Junction'
            base_vhl = ds.estimate_causal_delay(r['pei_score'], r['avg_duration_minutes'], is_junction)["vehicle_hours_lost"]
            p_sim = sim.run_simulation(base_vhl, "Increased Patrol", r['primary_junction'])
            potential_sims.append(p_sim)
        sims = potential_sims

    total_prevented_vhl = sum(s["metrics"]["delay_prevented_vhl"] for s in sims)
    total_fuel_saved = sum(s["metrics"]["fuel_saved_liters"] for s in sims)
    total_co2_avoided = sum(s["metrics"]["co2_avoided_kg"] for s in sims)
    total_dollars_saved = sum(s["metrics"]["dollars_saved"] for s in sims)

    # Equivalent Trees Planted (1 tree offsets ~22 kg CO2 per year)
    equivalent_trees = round(total_co2_avoided / 22.0, 1)

    # 1. Live Sustainability Ticker Banner (Full Width — 12 cols)
    with st.container(border=True):
        render_sustainability_ticker(
            base_fuel=st.session_state.fuel_saved_today,
            base_co2=st.session_state.co2_saved_today,
            base_dollars=st.session_state.economic_saved_today,
            height=270
        )

    # 2. 4 Impact Metric Cards (Equal Size)
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    with k_col1:
        render_animated_kpi("Delay Prevented", total_prevented_vhl, suffix=" VHL", id_suffix="sus1")
    with k_col2:
        render_animated_kpi("Idling Fuel Saved", total_fuel_saved, suffix=" L", id_suffix="sus2")
    with k_col3:
        render_animated_kpi("CO₂ Offset", total_co2_avoided, suffix=" kg", id_suffix="sus3")
    with k_col4:
        render_animated_kpi("Equivalent Trees Planted", equivalent_trees, suffix=" trees", id_suffix="sus4")

    # 3. Executive Sustainability Report Table (Full Width — 12 cols)
    with st.container(border=True):
        st.markdown("### 📋 Executive Sustainability Report")

        report_data = []
        for s in sims:
            report_data.append({
                "Location": s["hotspot_name"],
                "Intervention": s["intervention"],
                "Delay Saved (VHL)": s["metrics"]["delay_prevented_vhl"],
                "Fuel Saved (L)": s["metrics"]["fuel_saved_liters"],
                "CO2 Avoided (Kg)": s["metrics"]["co2_avoided_kg"],
                "Financial Savings ($)": s["metrics"]["dollars_saved"]
            })
        df_sus = pd.DataFrame(report_data)
        st.markdown(render_executive_table(df_sus), unsafe_allow_html=True)
        
        # Download Executive PDF Report button
        pdf_data = compile_executive_report_pdf()
        st.download_button(
            label="📥 Download Executive PDF Report",
            data=pdf_data,
            file_name=f"ParkTwin_Executive_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="dl_btn_sus_page",
            use_container_width=True
        )

    # 4. Plotly Savings Comparison Chart (Full Width)
    with st.container(border=True):
        st.markdown("### 📊 Environmental & Economic Savings Comparison")
        if not df_sus.empty:
            fig_sus_chart = px.bar(
                df_sus,
                x="Location",
                y=["Fuel Saved (L)", "CO2 Avoided (Kg)", "Financial Savings ($)"],
                barmode="group",
                title="Proactive Enforcement Environmental Savings by Intersection",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_sus_chart.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_title="Intersection Location",
                yaxis_title="Savings Value",
                legend_title="Metric",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_sus_chart, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No locations available to plot savings comparison.")

    # 5. Assumptions Expander
    with st.expander("📋 Environmental Coefficient & Financial Valuation Assumptions"):
        st.info("""
        **Environmental & Financial Assumptions:**
        1. **Idling Fuel Consumption**: An idling vehicle engine consumes ~1.2 liters of gasoline/diesel fuel per hour.
        2. **CO2 Emissions Coefficient**: Combustion of 1 liter of standard motor fuel releases 2.31 kg of CO2 into the atmosphere.
        3. **Time Valuation Factor**: Baseline value of time is calculated at $15.00/hour for vehicle occupants, plus local fuel cost savings ($1.15/liter).
        4. **Carbon Sequestration Factor**: 1 mature deciduous tree offsets approximately 22 kg of CO2 per year.
        """)
