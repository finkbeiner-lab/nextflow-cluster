#!/usr/bin/env python3
"""
Streamlit App for Nextflow Pipeline Configuration
Clean and simple UI for setting pipeline parameters
"""

import streamlit as st
import re
import subprocess
import os
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple

# Page configuration
st.set_page_config(
    page_title="Nextflow: Finkbeiner Configuration Settings",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def parse_config_file(config_path: str) -> Dict[str, Any]:
    """Parse the finkbeiner.config file and extract parameters with their types and options."""
    
    config_data = {}
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Find all parameter definitions
    param_pattern = r'params\.(\w+)\s*=\s*(.+?)(?://.*)?$'
    matches = re.findall(param_pattern, content, re.MULTILINE)
    
    for param_name, param_value in matches:
        # Clean up the value
        value = param_value.strip().rstrip(',')
        
        # Determine parameter type and options
        param_info = {
            'value': value,
            'type': 'text',
            'options': None,
            'min_value': None,
            'max_value': None,
            'help_text': ''
        }
        
        # Extract help text from comments
        help_match = re.search(rf'params\.{param_name}\s*=\s*.*?//\s*(.+?)$', content, re.MULTILINE)
        if help_match:
            param_info['help_text'] = help_match.group(1).strip()
        
        # Determine parameter type based on value and context
        if value.lower() in ['true', 'false']:
            param_info['type'] = 'boolean'
        elif param_name == 'use_aligned_tiles' and value == "''":
            # Special case for use_aligned_tiles - treat empty string as boolean false
            param_info['type'] = 'boolean'
            param_info['value'] = False
        elif value.startswith("'") and value.endswith("'"):
            param_info['type'] = 'text'
            param_info['value'] = value.strip("'")
        elif value.startswith('[') and value.endswith(']'):
            param_info['type'] = 'list'
            # Parse list values
            list_content = value[1:-1]
            if list_content:
                param_info['value'] = [item.strip().strip("'\"") for item in list_content.split(',')]
            else:
                param_info['value'] = []
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            param_info['type'] = 'integer'
            param_info['value'] = int(value)
        elif re.match(r'^\d+\.\d+$', value):
            param_info['type'] = 'float'
            param_info['value'] = float(value)
        elif value == 'null':
            param_info['type'] = 'text'
            param_info['value'] = ''
        else:
            param_info['type'] = 'text'
        
        # Check for specific options in comments
        options_match = re.search(rf'params\.{param_name}\s*=\s*.*?//\s*\[(.*?)\]', content, re.MULTILINE)
        if options_match:
            options_str = options_match.group(1)
            param_info['options'] = [opt.strip().strip("'\"") for opt in options_str.split(',')]
            param_info['type'] = 'select'
        
        # Set reasonable ranges for numeric parameters
        if param_info['type'] in ['integer', 'float']:
            if 'threshold' in param_name.lower() or 'area' in param_name.lower():
                param_info['min_value'] = 0
                param_info['max_value'] = 10000
            elif 'diameter' in param_name.lower():
                param_info['min_value'] = 1
                param_info['max_value'] = 100
            elif 'batch' in param_name.lower():
                param_info['min_value'] = 1
                param_info['max_value'] = 128
            elif 'epochs' in param_name.lower():
                param_info['min_value'] = 1
                param_info['max_value'] = 1000
            elif 'learning_rate' in param_name.lower():
                param_info['min_value'] = 1e-8
                param_info['max_value'] = 1e-2
                param_info['type'] = 'float'
                param_info['format'] = '%.2e'
        
        config_data[param_name] = param_info
    
    return config_data

def create_config_file(config_data: Dict[str, Any], output_path: str) -> None:
    """Create a new config file with the provided parameters."""
    
    with open(output_path, 'w') as f:
        f.write("/*\n")
        f.write(" //* pipeline input parameters\n")
        f.write("// https://nextflow-io.github.io/patterns/conditional-process/\n")
        f.write(" //*/\n")
        f.write("// SELECT MODULES\n\n")
        
        # Group parameters by category
        categories = {
            'Workflow Selection': ['DO_STD_WORKFLOW', 'DO_STD_WORKFLOW_IXM'],
            'Module Selection': [key for key in config_data.keys() if key.startswith('DO_') and key not in ['DO_STD_WORKFLOW', 'DO_STD_WORKFLOW_IXM']],
            'Shared Variables': ['experiment', 'morphology_channel', 'chosen_wells', 'wells_toggle', 'timepoints_toggle', 'chosen_timepoints', 'channels_toggle', 'chosen_channels', 'use_aligned_tiles', 'tile', 'analysis_version', 'img_norm_name', 'dir_structure'],
            'Register Experiment': ['input_path', 'output_path', 'template_path', 'robo_file', 'ixm_hts_file', 'platemap_path', 'illumination_file', 'robo_num', 'chosen_channels_for_register_exp', 'overwrite_experiment'],
            'Segmentation': ['segmentation_method', 'lower_area_thresh', 'upper_area_thresh', 'sd_scale_factor'],
            'Cellpose Segmentation': ['model_type', 'batch_size_cellpose', 'cell_diameter', 'flow_threshold', 'cell_probability'],
            'Puncta Segmentation': ['puncta_target_channel', 'puncta_segmentation_method', 'sigma1', 'sigma2', 'puncta_manual_thresh'],
            'Tracking': ['distance_threshold', 'voronoi_bool', 'USE_PROXIMITY', 'USE_OVERLAP', 'track_type'],
            'Intensity': ['target_channel'],
            'Crop': ['crop_size', 'target_channel_crop'],
            'Montage and Platemontage': ['tiletype', 'montage_pattern', 'well_size_for_platemontage', 'norm_intensity', 'image_overlap'],
            'Alignment': ['aligntiletype', 'dir_structure', 'alignment_algorithm', 'imaging_mode', 'shift_dict'],
            'CNN': ['cnn_model_type', 'label_type', 'label_name', 'classes', 'img_norn_name_cnn', 'filters', 'chosen_channels_for_cnn', 'num_channels', 'n_samples', 'epochs', 'batch_size', 'learning_rate', 'momentum', 'optimizer'],
            'Overlay': ['shift', 'contrast', 'tile'],
            'Overlay Montage': ['wells_toggle', 'timepoints_toggle']
        }
        
        for category, params in categories.items():
            f.write(f"\n// {category.upper()}\n")
            for param in params:
                if param in config_data:
                    value = config_data[param]['value']
                    help_text = config_data[param]['help_text']
                    
                    if config_data[param]['type'] == 'boolean':
                        if param == 'use_aligned_tiles' and not value:
                            f.write(f"params.{param} = ''")
                        else:
                            f.write(f"params.{param} = {str(value).lower()}")
                    elif config_data[param]['type'] == 'list':
                        if isinstance(value, list) and value:
                            f.write(f"params.{param} = {value}")
                        else:
                            f.write(f"params.{param} = []")
                    elif config_data[param]['type'] in ['integer', 'float']:
                        f.write(f"params.{param} = {value}")
                    else:
                        f.write(f"params.{param} = '{value}'")
                    
                    if help_text:
                        f.write(f"  // {help_text}")
                    f.write("\n")
        
        # Write any remaining parameters not in categories
        categorized_params = set()
        for params in categories.values():
            categorized_params.update(params)
        
        remaining_params = set(config_data.keys()) - categorized_params
        if remaining_params:
            f.write("\n// OTHER PARAMETERS\n")
            for param in sorted(remaining_params):
                value = config_data[param]['value']
                help_text = config_data[param]['help_text']
                
                if config_data[param]['type'] == 'boolean':
                    if param == 'use_aligned_tiles' and not value:
                        f.write(f"params.{param} = ''")
                    else:
                        f.write(f"params.{param} = {str(value).lower()}")
                elif config_data[param]['type'] == 'list':
                    if isinstance(value, list) and value:
                        f.write(f"params.{param} = {value}")
                    else:
                        f.write(f"params.{param} = []")
                elif config_data[param]['type'] in ['integer', 'float']:
                    f.write(f"params.{param} = {value}")
                else:
                    f.write(f"params.{param} = '{value}'")
                
                if help_text:
                    f.write(f"  // {help_text}")
                f.write("\n")

def run_pipeline() -> Tuple[bool, str, str]:
    """Execute the run.sh script using sbatch and return job ID if successful."""
    try:
        # Make sure run.sh is executable
        os.chmod('run.sh', 0o755)
        
        # Submit the job using sbatch
        result = subprocess.run(['sbatch', 'run.sh'], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            # Extract job ID from sbatch output
            job_id = extract_job_id(result.stdout)
            return True, result.stdout, job_id
        else:
            return False, result.stderr, ""
            
    except Exception as e:
        return False, str(e), ""

def extract_job_id(output: str) -> str:
    """Extract SLURM job ID from output."""
    import re
    # Look for patterns like "Submitted batch job 12345"
    match = re.search(r'Submitted batch job (\d+)', output)
    if match:
        return match.group(1)
    
    # Look for patterns like "Job 12345 submitted"
    match = re.search(r'Job (\d+) submitted', output)
    if match:
        return match.group(1)
    
    return ""


def check_job_status(job_id: str) -> str:
    """Check the status of a SLURM job."""
    if not job_id:
        return "No job ID"
    
    try:
        result = subprocess.run(['squeue', '-j', job_id, '--noheader'], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            # Job is still in queue
            parts = result.stdout.strip().split()
            if len(parts) >= 5:
                return parts[4]  # Status column
            return "Running"
        else:
            # Job not in queue, check if completed
            result = subprocess.run(['sacct', '-j', job_id, '--format=State', '--noheader'], 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
            return "Completed"
            
    except Exception as e:
        return f"Error: {str(e)}"

def get_queue_status() -> Tuple[bool, str]:
    """Get SLURM queue status using squeue."""
    try:
        result = subprocess.run(['squeue'], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
            
    except Exception as e:
        return False, f"Error getting queue status: {str(e)}"


def main():
    """Main Streamlit application."""
    
    # Custom CSS for button styling
    st.markdown("""
    <style>
    /* Green Save & Run button */
    button[data-testid="baseButton-secondary"][aria-label*="save_run_btn"],
    button[data-testid="baseButton-primary"][aria-label*="save_run_btn"] {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        height: 50px !important;
    }
    
    button[data-testid="baseButton-secondary"][aria-label*="save_run_btn"]:hover,
    button[data-testid="baseButton-primary"][aria-label*="save_run_btn"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    
    /* Red Cancel button */
    button[data-testid="baseButton-primary"][aria-label*="cancel_btn"] {
        background-color: #dc3545 !important;
        border-color: #dc3545 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        height: 50px !important;
    }
    
    button[data-testid="baseButton-primary"][aria-label*="cancel_btn"]:hover {
        background-color: #c82333 !important;
        border-color: #bd2130 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("🧬 Nextflow: Finkbeiner Configurator")
    st.markdown("Configure and run your Nextflow pipeline with a clean, simple interface.")
    
    # Load configuration (always read current file)
    config_path = "finkbeiner.config"
    
    if not os.path.exists(config_path):
        st.error(f"Configuration file '{config_path}' not found!")
        return
    
    # Parse configuration (read fresh every time)
    try:
        config_data = parse_config_file(config_path)
        st.info("📄 Loaded current configuration from finkbeiner.config")
    except Exception as e:
        st.error(f"Error parsing configuration file: {e}")
        return
    
    # Group parameters by category
    categories = {
        'Workflow Selection': ['DO_STD_WORKFLOW', 'DO_STD_WORKFLOW_IXM'],
        'Module Selection': [key for key in config_data.keys() if key.startswith('DO_') and key not in ['DO_STD_WORKFLOW', 'DO_STD_WORKFLOW_IXM']],
        'Shared Variables': ['experiment', 'morphology_channel', 'chosen_wells', 'wells_toggle', 'timepoints_toggle', 'chosen_timepoints', 'channels_toggle', 'chosen_channels', 'use_aligned_tiles', 'tile', 'analysis_version', 'img_norm_name', 'dir_structure'],
        'Register Experiment': ['input_path', 'output_path', 'template_path', 'robo_file', 'ixm_hts_file', 'platemap_path', 'illumination_file', 'robo_num', 'chosen_channels_for_register_exp', 'overwrite_experiment'],
        'Segmentation': ['segmentation_method', 'lower_area_thresh', 'upper_area_thresh', 'sd_scale_factor'],
        'Cellpose Segmentation': ['model_type', 'batch_size_cellpose', 'cell_diameter', 'flow_threshold', 'cell_probability'],
        'Puncta Segmentation': ['puncta_target_channel', 'puncta_segmentation_method', 'sigma1', 'sigma2', 'puncta_manual_thresh'],
        'Tracking': ['distance_threshold', 'voronoi_bool', 'USE_PROXIMITY', 'USE_OVERLAP', 'track_type'],
        'Intensity': ['target_channel'],
        'Crop': ['crop_size', 'target_channel_crop'],
        'Montage and Platemontage': ['tiletype', 'montage_pattern', 'well_size_for_platemontage', 'norm_intensity', 'image_overlap'],
        'Alignment': ['aligntiletype', 'dir_structure', 'alignment_algorithm', 'imaging_mode', 'shift_dict'],
        'CNN': ['cnn_model_type', 'label_type', 'label_name', 'classes', 'img_norn_name_cnn', 'filters', 'chosen_channels_for_cnn', 'num_channels', 'n_samples', 'epochs', 'batch_size', 'learning_rate', 'momentum', 'optimizer'],
        'Overlay': ['shift', 'contrast', 'tile'],
        'Overlay Montage': ['wells_toggle', 'timepoints_toggle']
    }
    
    # Special handling for Workflow Selection with radio buttons (outside form for dynamic updates)
    st.header("📝 Workflow Selection")
    
    # Determine current workflow selection
    current_workflow = "custom"
    
    # Get the actual boolean values
    std_workflow = config_data.get('DO_STD_WORKFLOW', {}).get('value', False)
    ixm_workflow = config_data.get('DO_STD_WORKFLOW_IXM', {}).get('value', False)
    
    # Convert string "true"/"false" to boolean if needed
    if isinstance(std_workflow, str):
        std_workflow = std_workflow.lower() == 'true'
    if isinstance(ixm_workflow, str):
        ixm_workflow = ixm_workflow.lower() == 'true'
    
    if std_workflow:
        current_workflow = "standard"
    elif ixm_workflow:
        current_workflow = "ixm"
    
    # Radio button selection
    workflow_choice = st.radio(
        "Choose Workflow Type:",
        options=["standard", "ixm", "custom"],
        index=["standard", "ixm", "custom"].index(current_workflow),
        format_func=lambda x: {
            "standard": "🔄 Standard Workflow",
            "ixm": "⚡ IXM Workflow", 
            "custom": "⚙️ Custom (Manual Module Selection)"
        }[x],
        help="Select the workflow type. Custom allows you to manually select individual modules.",
        key="workflow_choice_radio"
    )
    
    # Update workflow parameters based on selection
    if workflow_choice == "standard":
        config_data['DO_STD_WORKFLOW']['value'] = True
        config_data['DO_STD_WORKFLOW_IXM']['value'] = False
        # Set all module parameters to false for predefined workflows
        for param_name in config_data:
            if param_name.startswith('DO_') and param_name not in ['DO_STD_WORKFLOW', 'DO_STD_WORKFLOW_IXM']:
                config_data[param_name]['value'] = False
    elif workflow_choice == "ixm":
        config_data['DO_STD_WORKFLOW']['value'] = False
        config_data['DO_STD_WORKFLOW_IXM']['value'] = True
        # Set all module parameters to false for predefined workflows
        for param_name in config_data:
            if param_name.startswith('DO_') and param_name not in ['DO_STD_WORKFLOW', 'DO_STD_WORKFLOW_IXM']:
                config_data[param_name]['value'] = False
    else:  # custom
        config_data['DO_STD_WORKFLOW']['value'] = False
        config_data['DO_STD_WORKFLOW_IXM']['value'] = False
        # Keep existing module values for custom workflow
    
    st.markdown("---")
    
    # Create form for the configuration parameters
    with st.form("pipeline_config"):
        
        # Display all parameters organized by category
        for category_name, param_list in categories.items():
            # Skip Workflow Selection as it's handled above
            if category_name == 'Workflow Selection':
                continue
            
            # Handle Module Selection visibility based on workflow choice
            if category_name == 'Module Selection':
                st.header(f"📝 {category_name}")
                if workflow_choice != "custom":
                    st.info("🔒 Module Selection is disabled when using predefined workflows. Switch to 'Custom' to manually select modules.")
                    st.markdown("---")
                    continue
                else:
                    st.info("✅ Module Selection is enabled. You can manually select which modules to run.")
                    # Continue to show the module selection parameters
            else:
                st.header(f"📝 {category_name}")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            for i, param_name in enumerate(param_list):
                if param_name in config_data:
                    param_info = config_data[param_name]
                    
                    # Alternate between columns
                    current_col = col1 if i % 2 == 0 else col2
                    
                    with current_col:
                        # Create appropriate UI element based on parameter type
                        # Create unique key that includes category and parameter name
                        unique_key = f"{category_name}_{param_name}".replace(' ', '_').replace('-', '_').lower()
                        
                        if param_info['type'] == 'boolean':
                            value = st.checkbox(
                                param_name.replace('_', ' ').title(),
                                value=param_info['value'],
                                help=param_info['help_text'],
                                key=f"checkbox_{unique_key}"
                            )
                        elif param_info['type'] == 'select' and param_info['options']:
                            value = st.selectbox(
                                param_name.replace('_', ' ').title(),
                                options=param_info['options'],
                                index=param_info['options'].index(param_info['value']) if param_info['value'] in param_info['options'] else 0,
                                help=param_info['help_text'],
                                key=f"selectbox_{unique_key}"
                            )
                        elif param_info['type'] == 'list':
                            if param_info['value']:
                                value = st.text_area(
                                    param_name.replace('_', ' ').title(),
                                    value=', '.join(param_info['value']) if isinstance(param_info['value'], list) else str(param_info['value']),
                                    help=f"{param_info['help_text']} (comma-separated list)",
                                    key=f"textarea_{unique_key}"
                                )
                                # Convert back to list
                                value = [item.strip() for item in value.split(',') if item.strip()]
                            else:
                                value = st.text_area(
                                    param_name.replace('_', ' ').title(),
                                    value='',
                                    help=f"{param_info['help_text']} (comma-separated list)",
                                    key=f"textarea_{unique_key}"
                                )
                                value = [item.strip() for item in value.split(',') if item.strip()]
                        elif param_info['type'] == 'integer':
                            value = st.number_input(
                                param_name.replace('_', ' ').title(),
                                value=int(param_info['value']),
                                min_value=int(param_info['min_value'] or 0),
                                max_value=int(param_info['max_value'] or 1000000),
                                step=1,
                                help=param_info['help_text'],
                                key=f"number_int_{unique_key}"
                            )
                        elif param_info['type'] == 'float':
                            if 'learning_rate' in param_name.lower():
                                # Special handling for learning rate with scientific notation
                                value = st.number_input(
                                    param_name.replace('_', ' ').title(),
                                    value=float(param_info['value']),
                                    min_value=float(param_info['min_value'] or 1e-8),
                                    max_value=float(param_info['max_value'] or 1e-2),
                                    format="%.2e",
                                    help=param_info['help_text'],
                                    key=f"number_float_{unique_key}"
                                )
                            else:
                                value = st.number_input(
                                    param_name.replace('_', ' ').title(),
                                    value=float(param_info['value']),
                                    min_value=float(param_info['min_value'] or 0.0),
                                    max_value=float(param_info['max_value'] or 1000000.0),
                                    step=0.1,
                                    help=param_info['help_text'],
                                    key=f"number_float_{unique_key}"
                                )
                        else:  # text
                            value = st.text_input(
                                param_name.replace('_', ' ').title(),
                                value=param_info['value'],
                                help=param_info['help_text'],
                                key=f"text_{unique_key}"
                            )
                        
                        # Update the config data
                        config_data[param_name]['value'] = value
            
            # Add some spacing between sections
            st.markdown("---")
        
        # Save and Run buttons at the bottom
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            save_only = st.form_submit_button("💾 Save Config", type="secondary")
        
        with col3:
            save_and_run = st.form_submit_button("🚀 Save & Run Pipeline", type="primary", use_container_width=True, key="save_run_btn")
        
        # Handle form submission
        if save_only or save_and_run:
            # Overwrite the original config file
            create_config_file(config_data, config_path)
            
            st.success("✅ Configuration saved to finkbeiner.config!")
            
            # Show generated config
            with st.expander("📄 Updated Configuration"):
                st.code(open(config_path).read(), language="groovy")
            
            # Run pipeline if requested
            if save_and_run:
                st.info("🔄 Submitting pipeline to SLURM queue...")
                
                # Submit job (non-blocking)
                success, output, job_id = run_pipeline()
                
                if success:
                    st.markdown("""
                    <div style="text-align: center; padding: 20px; background-color: #d4edda; border: 2px solid #28a745; border-radius: 10px; margin: 20px 0;">
                        <h1 style="color: #155724; margin: 0; font-size: 2.5em;">✅ Pipeline submitted successfully!</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("🎯 Your pipeline has been submitted to the SLURM queue. You can now safely exit this application.")
                    
                    # Terminal-like display for squeue
                    st.markdown("---")
                    st.subheader("🖥️ SLURM Queue Status")
                    
                    # Create a terminal-like container
                    with st.container():
                       
                        # Wait 10 seconds for job to appear in queue
                        with st.spinner("Waiting 10 seconds for job to appear in queue..."):
                            import time
                            time.sleep(20)
                        
                        # Get and display queue status
                        queue_success, queue_output = get_queue_status()
                        if queue_success:
                            if queue_output.strip():
                                # Display in terminal-like format
                                st.text(queue_output)
                            else:
                                st.text("No jobs currently in queue")
                               
                        else:
                            st.text(f"Error: {queue_output}")
                      
                else:
                    st.error("❌ Pipeline submission failed!")
                    st.text("Error:")
                    st.text(output)
    
    # Job cancellation section - always available
    st.markdown("---")
    st.subheader("🛑 Job Cancellation")
    
    # Get current username
    current_user = os.getenv('USER', 'vgramas')
    cancel_command = f"scancel -u {current_user}"
    
    # Display the exact command that will be executed
    st.code(cancel_command, language="bash")
    
    # Initialize session state for cancel message
    if 'cancel_message' not in st.session_state:
        st.session_state.cancel_message = None
    if 'cancel_message_type' not in st.session_state:
        st.session_state.cancel_message_type = None
    
    # Display any existing cancel message
    if st.session_state.cancel_message:
        if st.session_state.cancel_message_type == 'success':
            st.success(st.session_state.cancel_message)
        elif st.session_state.cancel_message_type == 'error':
            st.error(st.session_state.cancel_message)
        # Clear the message after displaying
        st.session_state.cancel_message = None
        st.session_state.cancel_message_type = None
    
    # Cancel button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚫 Cancel All Jobs", type="primary", use_container_width=True, key="cancel_btn"):
            with st.spinner(f"Executing: {cancel_command}"):
                try:
                    # Execute scancel command
                    result = subprocess.run(['scancel', '-u', current_user], 
                                          capture_output=True, 
                                          text=True,
                                          timeout=30)
                    
                    if result.returncode == 0:
                        st.session_state.cancel_message = "✅ All jobs cancelled successfully!"
                        st.session_state.cancel_message_type = 'success'
                        st.rerun()  # Refresh to show updated queue
                    else:
                        st.session_state.cancel_message = f"❌ Failed to cancel jobs: {result.stderr}"
                        st.session_state.cancel_message_type = 'error'
                        st.rerun()
                except subprocess.TimeoutExpired:
                    st.session_state.cancel_message = "❌ Command timed out. Please try again."
                    st.session_state.cancel_message_type = 'error'
                    st.rerun()
                except Exception as e:
                    st.session_state.cancel_message = f"❌ Error cancelling jobs: {str(e)}"
                    st.session_state.cancel_message_type = 'error'
                    st.rerun()
    
   
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip**: Scroll through all sections to configure your pipeline parameters.")
    st.markdown("---")
    st.markdown("**Author**: Vivek Gopal Ramaswamy")
    st.markdown("**ID**: vgramas")

if __name__ == "__main__":
    main()
