# Nextflow Pipeline Streamlit UI

A clean and simple web interface for configuring and running your Nextflow pipeline.

## Features

- **Clean UI**: Organized by configuration sections with intuitive form elements
- **Auto-detection**: Automatically detects parameter types (text, numbers, booleans, lists, dropdowns)
- **Default values**: Loads default values from your `finkbeiner.config` file
- **Validation**: Built-in parameter validation and help text
- **Real-time execution**: Run your pipeline directly from the web interface

## Installation

1. Install Streamlit:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your browser to the URL shown (usually `http://localhost:8501`)

3. Navigate through the configuration sections using the sidebar

4. Modify parameters as needed

5. Click "🚀 Run Pipeline" to execute your pipeline

## Configuration Sections

- **Workflow Selection**: Choose between standard and IXM workflows
- **Module Selection**: Enable/disable specific pipeline modules
- **Shared Variables**: Basic experiment settings
- **Register Experiment**: Input/output paths and experiment setup
- **Segmentation**: Cell segmentation parameters
- **Cellpose Segmentation**: AI-based segmentation settings
- **Puncta Segmentation**: Puncta detection parameters
- **Tracking**: Cell tracking configuration
- **Intensity**: Intensity measurement settings
- **Crop**: Image cropping parameters
- **Montage and Platemontage**: Image montage settings
- **Alignment**: Image alignment parameters
- **CNN**: Deep learning model settings
- **Overlay**: Overlay visualization settings

## How It Works

1. The app parses your `finkbeiner.config` file
2. Creates appropriate UI elements based on parameter types:
   - Checkboxes for boolean values
   - Dropdowns for parameters with predefined options
   - Number inputs for numeric values
   - Text areas for lists
   - Text inputs for strings
3. Generates a new config file when you submit
4. Executes `run.sh` to start your pipeline

## Notes

- The app preserves all comments and structure from your original config file
- Generated config files are saved as `finkbeiner_ui.config`
- Pipeline execution runs in the background with real-time output
- All parameters include helpful tooltips from your original config comments
