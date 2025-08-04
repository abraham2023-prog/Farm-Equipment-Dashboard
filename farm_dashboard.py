import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px

# Set page config FIRST
st.set_page_config(
    page_title="🚜 Farm Equipment Dashboard",
    page_icon="🚜",
    layout="wide"
)

# Initialize session state with realistic farm data
if 'machines' not in st.session_state:
    st.session_state.machines = {
        'JD_Tractor_8245R': {
            'last_service': (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d'),
            'hours_used': 287,
            'status': 'Active',
            'fuel_level': 78,
            'location': 'North Field',
            'maintenance_history': [
                {'date': '2023-01-15', 'type': 'Oil Change', 'tech': 'Smith'},
                {'date': '2023-04-02', 'type': 'Tire Replacement', 'tech': 'Johnson'}
            ]
        },
        'CIH_Combine_8250': {
            'last_service': (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d'),
            'hours_used': 412,
            'status': 'Maintenance Needed',
            'fuel_level': 32,
            'location': 'Grain Silos',
            'maintenance_history': [
                {'date': '2023-03-22', 'type': 'Engine Service', 'tech': 'Williams'}
            ]
        },
        'SP_Sprayer_4430': {
            'last_service': (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d'),
            'hours_used': 156,
            'status': 'Active',
            'fuel_level': 91,
            'location': 'South Field',
            'maintenance_history': [
                {'date': '2023-05-10', 'type': 'Nozzle Replacement', 'tech': 'Brown'}
            ]
        }
    }

# Dashboard Header
st.title("🚜 Farm Machinery Monitoring Dashboard")
st.markdown("### Real-time Equipment Tracking and Maintenance Scheduling")

# Sidebar Controls
with st.sidebar:
    st.header("Equipment Manager")
    selected_machine = st.selectbox(
        "Select Equipment", 
        options=list(st.session_state.machines.keys()),
        format_func=lambda x: x.replace('_', ' '),
        key="machine_selector"
    )
    
    # Service record form with UNIQUE key
    with st.expander("Service Records"):
        machine_data = st.session_state.machines[selected_machine]
        
        with st.form("service_update_form"):  # Changed key
            st.write(f"### {selected_machine.replace('_', ' ')}")
            service_date = st.date_input(
                "Last Service Date", 
                value=datetime.strptime(machine_data['last_service'], '%Y-%m-%d'),
                key="service_date"
            )
            hours_used = st.number_input(
                "Operating Hours", 
                min_value=0, 
                value=machine_data['hours_used'],
                key="hours_used"
            )
            fuel_level = st.slider(
                "Fuel Level (%)", 
                min_value=0, 
                max_value=100, 
                value=machine_data['fuel_level'],
                key="fuel_level"
            )
            status = st.selectbox(
                "Status", 
                ["Active", "Maintenance Needed", "Out of Service"],
                index=["Active", "Maintenance Needed", "Out of Service"].index(machine_data['status']),
                key="status_select"
            )
            
            if st.form_submit_button("Update Equipment"):
                st.session_state.machines[selected_machine].update({
                    'last_service': service_date.strftime('%Y-%m-%d'),
                    'hours_used': hours_used,
                    'fuel_level': fuel_level,
                    'status': status
                })
                st.success("Equipment record updated!")

    # Add new equipment form with UNIQUE key
    with st.expander("Add New Equipment"):
        with st.form("new_equipment_form"):  # Changed key
            st.write("### Register New Equipment")
            machine_name = st.text_input("Equipment Name", key="equip_name")
            machine_type = st.selectbox("Type", ["Tractor", "Combine", "Sprayer", "Planter", "Other"], key="equip_type")
            initial_hours = st.number_input("Initial Hours", min_value=0, value=0, key="init_hours")
            
            if st.form_submit_button("Add to Fleet"):
                if machine_name:
                    key = f"{machine_type}_{machine_name}".upper().replace(' ', '_')
                    st.session_state.machines[key] = {
                        'last_service': datetime.now().strftime('%Y-%m-%d'),
                        'hours_used': initial_hours,
                        'status': 'Active',
                        'fuel_level': 100,
                        'location': 'Yard',
                        'maintenance_history': []
                    }
                    st.success(f"{machine_name} added to fleet!")
                else:
                    st.error("Please enter an equipment name")

    # Data management section
    with st.expander("Data Management"):
        st.download_button(
            "📥 Export Equipment Data",
            pd.DataFrame.from_dict(st.session_state.machines, orient='index').to_csv(),
            file_name="farm_equipment_data.csv",
            key="export_data"
        )
        
        uploaded_file = st.file_uploader("Import Equipment Data", type=['csv'], key="import_data")
        if uploaded_file:
            try:
                new_data = pd.read_csv(uploaded_file).to_dict(orient='index')
                st.session_state.machines = new_data
                st.success("Equipment data imported successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error importing data: {e}")

# Main Dashboard Layout
tab1, tab2, tab3 = st.tabs(["Overview", "Maintenance", "Telemetry"])

with tab1:
    st.header("Equipment Overview")
    
    col1, col2, col3 = st.columns(3)
    active_machines = sum(1 for m in st.session_state.machines.values() if m['status'] == 'Active')
    maintenance_needed = sum(1 for m in st.session_state.machines.values() if m['status'] == 'Maintenance Needed')
    
    col1.metric("Total Equipment", len(st.session_state.machines))
    col2.metric("Active", active_machines, f"{active_machines/len(st.session_state.machines)*100:.0f}%")
    col3.metric("Needs Maintenance", maintenance_needed, delta_color="inverse")
    
    st.subheader("Current Status")
    machine_df = pd.DataFrame.from_dict(st.session_state.machines, orient='index')
    st.dataframe(
        machine_df[['last_service', 'hours_used', 'status', 'location', 'fuel_level']]
        .rename(columns={
            'last_service': 'Last Service', 
            'hours_used': 'Hours', 
            'fuel_level': 'Fuel %'
        })
        .style.applymap(
            lambda x: 'color: red' if x == 'Maintenance Needed' else 'color: green', 
            subset=['status']
        )
        .background_gradient(subset=['Fuel %'], cmap='YlOrRd')
        .format({'Fuel %': '{:.0f}%'}),
        height=400
    )

with tab2:
    st.header("Maintenance Management")
    
    with st.expander("🔴 Critical Alerts", expanded=True):
        for machine, data in st.session_state.machines.items():
            if data['status'] == 'Maintenance Needed':
                days_since_service = (datetime.now() - datetime.strptime(data['last_service'], '%Y-%m-%d')).days
                st.warning(f"""
                **{machine.replace('_', ' ')}**  
                🕒 {days_since_service} days since last service | ⏱️ {data['hours_used']} operating hours  
                📍 {data['location']} | ⛽ {data['fuel_level']}% fuel remaining
                """)
    
    st.subheader("Service History")
    selected_history = st.selectbox(
        "View maintenance history for:", 
        options=list(st.session_state.machines.keys()),
        format_func=lambda x: x.replace('_', ' '),
        key="history_selector"
    )
    history_df = pd.DataFrame(st.session_state.machines[selected_history]['maintenance_history'])
    if not history_df.empty:
        st.dataframe(history_df.sort_values('date', ascending=False))
    else:
        st.info("No maintenance history recorded for this equipment")

with tab3:
    st.header("Equipment Telemetry")
    
    if st.checkbox("Enable live telemetry", False, key="telemetry_toggle"):
        telemetry_placeholder = st.empty()
        stop_simulation = st.button("Stop Simulation", key="stop_sim")
        
        while not stop_simulation:
            # Simulate changing values
            for machine in st.session_state.machines:
                if np.random.random() > 0.9:
                    st.session_state.machines[machine]['status'] = "Maintenance Needed"
                st.session_state.machines[machine]['fuel_level'] = max(
                    0, 
                    st.session_state.machines[machine]['fuel_level'] - np.random.randint(0, 3)
                )
            
            # Create telemetry display
            telemetry_data = []
            for machine, data in st.session_state.machines.items():
                telemetry_data.append({
                    'Equipment': machine.replace('_', ' '),
                    'Status': data['status'],
                    'Hours': data['hours_used'],
                    'Fuel': data['fuel_level'],
                    'Location': data['location']
                })
            
            df = pd.DataFrame(telemetry_data)
            fig = px.bar(
                df, 
                x='Equipment', 
                y='Fuel',
                color='Status',
                color_discrete_map={
                    'Active': 'green',
                    'Maintenance Needed': 'red',
                    'Out of Service': 'gray'
                },
                title='Current Fuel Levels'
            )
            telemetry_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(3)
            
            if stop_simulation:
                break
    else:
        st.info("Enable live telemetry to see real-time equipment data")