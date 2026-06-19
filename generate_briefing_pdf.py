import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime

def generate_pdf():
    pdf_path = "ParkTwin_AI_Implementation_Briefing.pdf"
    doc = SimpleDocTemplate(
        pdf_path, 
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#FF4D4F'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=15,
        fontName='Helvetica'
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )
    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0F172A'),
        fontName='Courier',
        spaceAfter=6,
        leftIndent=15
    )
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0F172A'),
        fontName='Helvetica'
    )
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    # Title & Metadata
    story.append(Paragraph("PARKTWIN AI: COMPLETE SYSTEM IMPLEMENTATION SPECIFICATION", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')} | Smart City Command Center", subtitle_style))
    story.append(Spacer(1, 10))
    
    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Core Paradigm Shift", h2_style))
    story.append(Paragraph(
        "ParkTwin AI is a decision intelligence platform designed to transition municipal congestion management from a reactive "
        "enforcement model to a proactive, resilience-focused command system. By combining geospatial Haversine DBSCAN clustering, "
        "a multi-criteria Parking Externality Index (PEI), XGBoost temporal forecasts, game-theoretic SHAP explainability, and real-time "
        "YOLOv8 + ByteTrack edge computer vision pipelines, ParkTwin AI provides comprehensive decision support. "
        "Crucially, the platform incorporates two novel components: <b>Traffic Memory</b>, which enables the system to learn from historical "
        "intervention success rates to build institutional memory, and the <b>Traffic Immunity Score (TIS)</b>, a research-grade metric "
        "quantifying the systemic resilience of urban nodes.",
        body_style
    ))
    
    # 2. Complete Modules List (Table)
    story.append(Paragraph("2. Core Implementation Modules and Files", h2_style))
    modules_data = [
        [Paragraph("Module / Component", table_header_style), Paragraph("Target Files", table_header_style), Paragraph("Key Features Implemented", table_header_style)]
    ]
    
    modules_list = [
        ("Data Pipeline & Ingestion", "app/data_pipeline.py, app/config.py", "Auto-encoding search, column normalization, temporal feature engineering, rush hour flag computations."),
        ("DBSCAN Spatial Clustering", "services/hotspot_service.py", "Haversine geospatial clustering, epsilon constraint calibration (50m radius), temporal breakdowns."),
        ("PEI Ranking Engine", "services/pei_service.py", "Multi-criteria score (Frequency, Duration, Peak severity, Junction criticality, Density), severity labeling."),
        ("Delay & Network Ripple", "services/delay_service.py", "Vehicle-Hours Lost (VHL) calculations, NetworkX directed graph congestion propagation modeling (1st & 2nd hop)."),
        ("XGBoost Risk Forecasting", "services/forecast_service.py", "Predictive temporal model classifier training, hourly risk timeline forecasting (24-hour ahead predictions)."),
        ("SHAP Explainability", "services/explainability_service.py", "Tree SHAP explainer implementation, automated generation of global feature importance and waterfall plots."),
        ("YOLOv8 ByteTrack CV", "services/detection_service.py", "YOLOv8 vehicle detection + ByteTrack tracking, stationary duration triggers, edge diagnostics."),
        ("Digital Twin Simulator", "services/simulation_service.py", "Intervention scenario mapping (Patrol, Tow, Barricades, Fine), 24h recovery profiles, economic benefits."),
        ("Traffic Memory Engine", "services/traffic_memory_service.py", "Institutional learning profiles, historical success rate evaluations, recovery time averages, recurrence checks."),
        ("Traffic Immunity Score (TIS)", "services/immunity_service.py", "TIS weighted calculations (0-100), resilience classifications, simulation projection updates."),
        ("FastAPI Gateway Server", "api/main.py", "FastAPI endpoints for hotspots, leaderboard, forecasting simulation, and CV telemetry."),
        ("Command UI & Demo", "dashboard/dashboard.py", "10-step autoplay Smart City Story Mode, Plotly dashboards, PDF exporter, storytelling blocks.")
    ]
    
    for mod, files, features in modules_list:
        modules_data.append([
            Paragraph(mod, table_text_style),
            Paragraph(files, table_text_style),
            Paragraph(features, table_text_style)
        ])
        
    t_mods = Table(modules_data, colWidths=[130, 150, 250])
    t_mods.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.HexColor('#F1F5F9')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_mods)
    story.append(Spacer(1, 10))
    story.append(PageBreak())
    
    # 3. New Innovative Frameworks
    story.append(Paragraph("3. Details of Highlighted Components", h2_style))
    
    story.append(Paragraph("3.1. Traffic Memory Engine", ParagraphStyle('SubSub', parent=h2_style, fontSize=10)))
    story.append(Paragraph(
        "Instead of treating resolved violations as closed history, the Traffic Memory Engine continuously builds "
        "institutional profiles for active junctions. It dynamically correlates historical interventions "
        "(Tow Vehicle, Officer Patrol, Barricades, and Fine Only) to evaluate actual outcomes. It computes "
        "recurrence ratios, average blockage durations, policy effectiveness percentages, recovery times, and "
        "critical congestion spillover escalation counts. The generated learning profiles advise the dispatcher "
        "about which intervention historically resolved the gridlock fastest.",
        body_style
    ))
    
    story.append(Paragraph("3.2. Traffic Immunity Score (TIS) & Resilience", ParagraphStyle('SubSub', parent=h2_style, fontSize=10)))
    story.append(Paragraph(
        "The Traffic Immunity Score (TIS) is a weighted, normalized index (0-100) representing systemic resilience: "
        "<br/><i>TIS = 0.30 x Recovery Speed + 0.25 x Intervention Effectiveness + 0.20 x Spillover Resistance + 0.15 x Recurrence Resistance + 0.10 x Sustainability Efficiency</i>"
        "<br/>It classifies grid nodes into three categories: "
        "<br/>• <b>Fragile</b> (0–40): Prone to immediate gridlocks and ripple propagation. "
        "<br/>• <b>Adaptive</b> (41–70): Stable under normal flows but sensitive to high-severity events. "
        "<br/>• <b>Resilient</b> (71–100): High capacity to absorb disruptions and clear blockages rapidly.",
        body_style
    ))
    
    story.append(Paragraph("3.3. 10-Step Decision Support Story Mode", ParagraphStyle('SubSub', parent=h2_style, fontSize=10)))
    story.append(Paragraph(
        "The Smart City Demo HUD takes judges step-by-step through a unified decision pipeline: "
        "<br/><b>1. Live CV Detection</b> ➔ <b>2. Spatial DBSCAN Hotspots</b> ➔ <b>3. PEI Severity Scoring</b> ➔ "
        "<b>4. XGBoost Risk Forecasting</b> ➔ <b>5. AI Traffic Commander Recommendation</b> ➔ "
        "<b>6. Traffic Memory Profile</b> ➔ <b>7. Digital Twin Simulation</b> ➔ <b>8. Traffic Immunity Evaluation</b> ➔ "
        "<b>9. Sustainability Analytics</b> ➔ <b>10. Executive PDF Briefing compile.</b>",
        body_style
    ))

    # 4. APIs and Test Suite Validation
    story.append(Paragraph("4. Backend APIs & Test Suite Validations", h2_style))
    story.append(Paragraph(
        "<b>REST API Endpoints:</b> FastAPI handles all client requests: "
        "<br/>• <code>GET /</code>: Health Check "
        "<br/>• <code>GET /api/hotspots</code>: Spatial clusters "
        "<br/>• <code>GET /api/leaderboard</code>: PEI Ranking leaderboard "
        "<br/>• <code>POST /api/forecast</code>: XGBoost temporal forecast probability "
        "<br/>• <code>POST /api/simulate</code>: Run causal what-if matrix math "
        "<br/>• <code>POST /api/detect</code>: Live CV object tracking output",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>Pytest Test Suite Verification:</b> A complete test suite containing 10 automated test cases "
        "validates all models, pipelines, services, and REST APIs: "
        "<br/>• <code>test_data_pipeline</code>, <code>test_hotspot_dbscan</code>, <code>test_pei_calculation</code>: Data and Priority calculations. "
        "<br/>• <code>test_forecast_xgboost</code>, <code>test_delay_estimation</code>, <code>test_simulation_intervention</code>: Models and Projections. "
        "<br/>• <code>test_detection_pipeline</code>, <code>test_api_endpoints</code>: CV object tracking and backend health. "
        "<br/>• <code>test_traffic_memory_service</code>, <code>test_traffic_immunity_service</code>: Institutional memory profiles, TIS mathematical bounds, and simulated improvements.",
        body_style
    ))
    
    doc.build(story)
    print("PDF Briefing compiled successfully.")

if __name__ == '__main__':
    generate_pdf()
