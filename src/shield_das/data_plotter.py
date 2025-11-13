"""Data plotter module for SHIELD Data Acquisition System.

This module provides the main DataPlotter class which creates an interactive
Dash web application for visualizing pressure, temperature, and permeability
data from multiple experimental datasets.
"""

import io
import os
import threading
import webbrowser
import zipfile
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import html

from . import layout_components as lc
from .callbacks import register_all_callbacks
from .dataset import Dataset
from .dataset_table import DatasetTable
from .figures import PermeabilityPlot, PressurePlot, TemperaturePlot


class DataPlotter:
    """Plotter UI for pressure gauge datasets using Dash.

    Provides a Dash app that displays upstream and downstream pressure
    plots for multiple datasets. Datasets are stored in `self.datasets` as a list.

    Args:
        dataset_paths: List of folder paths containing datasets to load on
        dataset_names: List of names corresponding to each dataset path
        port: Port number for Dash app (default: 8050)

    Attributes:
        dataset_paths: List of folder paths containing datasets to load on
        dataset_names: List of names corresponding to each dataset path
        port: Port number for Dash app (default: 8050)
        app: Dash app instance
        datasets: List of Dataset instances for plotting
        figure_resamplers: Dictionary of FigureResampler instances for each plot
    """

    # Type hints / attributes
    dataset_paths: list[str]
    dataset_names: list[str]
    port: int

    app: dash.Dash
    datasets: list[Dataset]
    upstream_datasets: list[dict]
    downstream_datasets: list[dict]

    def __init__(
        self,
        dataset_paths: list[str] | None = None,
        dataset_names: list[str] | None = None,
        port: int = 8050,
    ):
        """Initialize the DataPlotter with datasets and configuration.

        Args:
            dataset_paths: List of folder paths containing datasets to load.
                Each path should contain run_metadata.json and data files.
            dataset_names: List of display names for each dataset.
                Must match length of dataset_paths if provided.
            port: Port number for the Dash web server (default: 8050).
        """
        # Initialize dataset configuration
        self.dataset_paths = dataset_paths or []
        self.dataset_names = dataset_names or []
        self.port = port

        # Initialize Dash app with Bootstrap theme and Font Awesome icons
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
            ],
        )
        # Set the browser tab title
        self.app.title = "SHIELD Data Visualisation"

        # Initialize empty datasets list (populated via load_data())
        self.datasets: list[Dataset] = []

        # Store FigureResampler instances for interactive zoom/pan callbacks
        self.figure_resamplers: dict[str, Any] = {}

        # Initialize plot generators (created after datasets are loaded)
        self.upstream_plot: PressurePlot | None = None
        self.downstream_plot: PressurePlot | None = None
        self.temperature_plot: TemperaturePlot | None = None
        self.permeability_plot: PermeabilityPlot | None = None

    @property
    def dataset_paths(self) -> list[str]:
        """Get the list of dataset folder paths.

        Returns:
            list[str]: List of absolute paths to dataset folders
        """
        return self._dataset_paths

    @dataset_paths.setter
    def dataset_paths(self, value: list[str]) -> None:
        """Set and validate dataset folder paths.

        Validates that:
        - Value is a list of strings
        - All paths exist on filesystem
        - All paths are unique
        - Each path contains CSV data files
        - Each path contains run_metadata.json

        Args:
            value: List of folder paths to validate

        Raises:
            ValueError: If validation fails
            FileNotFoundError: If required files are missing
        """
        # Validate input type
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError("dataset_paths must be a list of strings")

        # Validate all paths exist
        for dataset_path in value:
            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset path does not exist: {dataset_path}")

        # Ensure all paths are unique
        if len(value) != len(set(value)):
            raise ValueError("dataset_paths must contain unique paths")

        # Verify CSV files exist in each path
        for dataset_path in value:
            csv_files = [
                f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")
            ]
            if not csv_files:
                raise FileNotFoundError(
                    f"No data CSV files found in dataset path: {dataset_path}"
                )

        # Verify metadata file exists in each path
        for dataset_path in value:
            metadata_file = os.path.join(dataset_path, "run_metadata.json")
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(
                    f"No run_metadata.json file found in dataset path: {dataset_path}"
                )

        self._dataset_paths = value

    @property
    def dataset_names(self) -> list[str]:
        """Get the list of dataset display names.

        Returns:
            list[str]: List of names for each dataset
        """
        return self._dataset_names

    @dataset_names.setter
    def dataset_names(self, value: list[str]) -> None:
        """Set and validate dataset display names.

        Validates that:
        - Value is a list of strings
        - Length matches dataset_paths length
        - All names are unique

        Args:
            value: List of display names for datasets

        Raises:
            ValueError: If validation fails
        """
        # Validate input type
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError("dataset_names must be a list of strings")

        # Ensure length matches paths
        if len(value) != len(self.dataset_paths):
            raise ValueError(
                f"dataset_names length ({len(value)}) must match dataset_paths "
                f"length ({len(self.dataset_paths)})"
            )

        # Ensure all names are unique
        if len(value) != len(set(value)):
            raise ValueError("dataset_names must contain unique names")

        self._dataset_names = value

    def load_data(self, dataset_path: str, dataset_name: str) -> None:
        """Load and process data from a dataset folder.

        Creates a Dataset instance, assigns a color, processes the data files,
        and adds it to the internal datasets list.

        Only supports metadata version 1.3. Will raise ValueError for other versions.

        Args:
            dataset_path: Absolute path to folder containing run_metadata.json
                and data CSV files
            dataset_name: Display name to assign to this dataset

        Raises:
            ValueError: If metadata version is not 1.3
            FileNotFoundError: If metadata file or required data files are missing
        """
        # Create Dataset instance with path and name
        dataset = Dataset(path=dataset_path, name=dataset_name)

        # Assign color based on dataset index for consistent visualization
        dataset.colour = self.get_next_color(len(self.datasets))

        # Load and process data from CSV files
        dataset.process_data()

        # Add to internal datasets list
        self.datasets.append(dataset)

    def get_next_color(self, index: int) -> str:
        """Get a color for a dataset based on its index.

        Cycles through a predefined color palette to ensure visual distinction
        between datasets in plots.

        Args:
            index: Zero-based index of the dataset in the datasets list

        Returns:
            str: Hex color code (e.g., "#000000" for black)
        """
        colors = [
            "#000000",  # Black
            "#DF1AD2",  # Magenta
            "#779BE7",  # Light Blue
            "#49B6FF",  # Blue
            "#254E70",  # Dark Blue
            "#0CCA4A",  # Green
            "#929487",  # Gray
            "#A1B0AB",  # Light Gray
        ]
        return colors[index % len(colors)]

    def create_layout(self) -> dbc.Container:
        """Create the main Dash application layout.

        Builds the complete web interface layout by assembling all UI components
        including header, dataset management, plots, controls, and hidden data stores.
        Uses layout component builders from layout_components module.

        Returns:
            dbc.Container: Bootstrap container with all dashboard components arranged
                in fluid layout for responsive design

        Layout Structure:
            1. Header with SHIELD logo and title
            2. Dataset management card with table and controls
            3. Hidden data stores for plot settings
            4. Pressure plots (upstream/downstream) in row
            5. Pressure plot controls
            6. Temperature plot with controls
            7. Permeability plot with controls
            8. Bottom spacing
            9. Download components
            10. Live data update interval component
        """
        from .layout_components import create_download_components, create_hidden_stores

        dataset_table = DatasetTable(self.datasets)

        return dbc.Container(
            [
                # Dashboard header with title and branding
                lc.create_header(),
                # Dataset management: table with add/edit/delete controls
                lc.create_dataset_management_card(dataset_table.create()),
                # Hidden stores for plot settings (upstream, downstream, temp, perm)
                *create_hidden_stores(),
                # Pressure plots side-by-side in a row
                lc.create_pressure_plots_row(
                    self._generate_upstream_plot(),
                    self._generate_downstream_plot(),
                ),
                # Pressure plot controls (apply to both upstream and downstream)
                lc.create_plot_controls_row(),
                # Temperature plot with dedicated controls below
                lc.create_temperature_plot_card(self._generate_temperature_plot()),
                # Permeability (Arrhenius) plot with dedicated controls below
                lc.create_permeability_plot_card(self._generate_permeability_plot()),
                # Spacing at bottom for better layout
                lc.create_bottom_spacing(),
                # Download modals for dataset and plot exports
                *create_download_components(),
                # Interval component for live data refresh
                lc.create_live_data_interval(),
            ],
            fluid=True,  # Use full width for responsive design
        )

    def _create_dataset_download(self, dataset_path: str) -> dict[str, Any]:
        """Package original dataset files for download as a ZIP archive.

        Creates an in-memory ZIP file containing all files from the dataset
        folder, preserving the relative directory structure.

        Args:
            dataset_path: Absolute path to the dataset folder to package

        Returns:
            dict: Download specification with keys:
                - content: ZIP file bytes
                - filename: Suggested filename (e.g., "dataset.zip")
                - type: MIME type ("application/zip")
        """
        # Create in-memory ZIP archive preserving directory structure
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Walk through all files and subdirectories
            for root, _dirs, files in os.walk(dataset_path):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    try:
                        # Store relative path to maintain structure in archive
                        arcname = os.path.relpath(file_path, dataset_path)
                        zf.write(file_path, arcname)
                    except Exception:
                        # Skip files we cannot read (permissions, etc.)
                        continue

        # Rewind buffer for reading
        mem_zip.seek(0)

        # Use folder name as ZIP filename
        base = os.path.basename(os.path.normpath(dataset_path))
        return dict(
            content=mem_zip.getvalue(),
            filename=f"{base}.zip",
            type="application/zip",
        )

    def _initialize_figure_generators(self) -> None:
        """Initialize plot generator instances with loaded datasets.

        Creates instances of PressurePlot, TemperaturePlot, and PermeabilityPlot,
        passing the loaded datasets to each. These generators are used to create
        the initial plots and regenerate them when settings change.
        """
        # Create upstream pressure plot generator
        self.upstream_plot = PressurePlot(
            datasets=self.datasets, plot_id="upstream-plot", plot_type="upstream"
        )
        # Create downstream pressure plot generator
        self.downstream_plot = PressurePlot(
            datasets=self.datasets, plot_id="downstream-plot", plot_type="downstream"
        )
        # Create temperature plot generator
        self.temperature_plot = TemperaturePlot(
            datasets=self.datasets, plot_id="temperature-plot"
        )
        # Create permeability (Arrhenius) plot generator
        self.permeability_plot = PermeabilityPlot(
            datasets=self.datasets, plot_id="permeability-plot"
        )

    def _generate_plot(
        self,
        plot_type: str,
        show_error_bars: bool = False,
        show_valve_times: bool = False,
        x_scale: str = "linear",
        y_scale: str = "log",
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> Any:
        """Generic plot generation method for all plot types.

        This method provides a unified interface for generating any plot type.
        It sets the appropriate plot parameters, generates the plot, and stores
        the FigureResampler instance for interactive callbacks.

        Args:
            plot_type: Type of plot to generate. Must be one of:
                - 'upstream': Upstream pressure vs time
                - 'downstream': Downstream pressure vs time
                - 'temperature': Temperature vs time
                - 'permeability': Permeability vs inverse temperature (Arrhenius)
            show_error_bars: Whether to display error bars on data points
            show_valve_times: Whether to display valve event markers.
                Only applies to pressure plots (upstream/downstream).
            x_scale: X-axis scale type ('linear' or 'log')
            y_scale: Y-axis scale type ('linear' or 'log')
            x_min: Minimum x-axis value (None for auto-scaling)
            x_max: Maximum x-axis value (None for auto-scaling)
            y_min: Minimum y-axis value (None for auto-scaling)
            y_max: Maximum y-axis value (None for auto-scaling)

        Returns:
            Plotly Figure object (FigureResampler) for the specified plot type

        Raises:
            ValueError: If plot_type is not recognized
        """
        # Map plot type to corresponding plot generator and callback key
        plot_config: dict[str, tuple[Any, str]] = {
            "upstream": (self.upstream_plot, "upstream-plot"),
            "downstream": (self.downstream_plot, "downstream-plot"),
            "temperature": (self.temperature_plot, "temperature-plot"),
            "permeability": (self.permeability_plot, "permeability-plot"),
        }

        # Validate plot type
        if plot_type not in plot_config:
            raise ValueError(f"Unknown plot type: {plot_type}")

        # Get plot generator and resampler key
        plot_obj, resampler_key = plot_config[plot_type]

        # Determine if valve times should be shown (only for pressure plots)
        use_valve_times = (
            show_valve_times if plot_type in ["upstream", "downstream"] else False
        )

        plot_obj.plot_parameters = {
            "show_error_bars": show_error_bars,
            "show_valve_times": use_valve_times,
            "x_scale": x_scale,
            "y_scale": y_scale,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }

        # Generate the plot
        fig = plot_obj.generate()

        # Store the FigureResampler instance for callback registration
        self.figure_resamplers[resampler_key] = plot_obj.figure_resampler

        return fig

    def _generate_upstream_plot(self, **kwargs) -> Any:
        """Generate upstream pressure vs time plot.

        This is a convenience wrapper around _generate_plot that automatically
        sets the plot type to 'upstream'.

        Args:
            **kwargs: All keyword arguments are passed to _generate_plot.
                See _generate_plot() for available parameters.

        Returns:
            Plotly Figure object (FigureResampler) for upstream pressure plot
        """
        return self._generate_plot("upstream", **kwargs)

    def _generate_downstream_plot(self, **kwargs) -> Any:
        """Generate downstream pressure vs time plot.

        This is a convenience wrapper around _generate_plot that automatically
        sets the plot type to 'downstream'.

        Args:
            **kwargs: All keyword arguments are passed to _generate_plot.
                See _generate_plot() for available parameters.

        Returns:
            Plotly Figure object (FigureResampler) for downstream pressure plot
        """
        return self._generate_plot("downstream", **kwargs)

    def _generate_temperature_plot(self, **kwargs) -> Any:
        """Generate temperature vs time plot.

        This is a convenience wrapper around _generate_plot that automatically
        sets the plot type to 'temperature'.

        Args:
            **kwargs: All keyword arguments are passed to _generate_plot.
                See _generate_plot() for available parameters.

        Returns:
            Plotly Figure object (FigureResampler) for temperature plot
        """
        return self._generate_plot("temperature", **kwargs)

    def _generate_permeability_plot(self, **kwargs) -> Any:
        """Generate permeability (Arrhenius) plot.

        This is a convenience wrapper around _generate_plot that automatically
        sets the plot type to 'permeability'. The permeability plot shows
        permeability vs inverse temperature with reference HTM data.

        Args:
            **kwargs: All keyword arguments are passed to _generate_plot.
                See _generate_plot() for available parameters.

        Returns:
            Plotly Figure object (FigureResampler) for permeability plot
        """
        return self._generate_plot("permeability", **kwargs)

    def start(self) -> None:
        """Initialize the dashboard and start the Dash web server.

        This method orchestrates the complete dashboard startup process:
        1. Loads all dataset data from configured paths
        2. Initializes plot generators with loaded data
        3. Creates the Dash application layout
        4. Registers all callbacks for interactivity
        5. Opens the dashboard in the default web browser
        6. Starts the Flask development server (blocking)

        The server runs on localhost:{self.port} and automatically opens
        in a browser after a short delay to ensure the server is ready.

        Note:
            This method blocks until the server is shut down (Ctrl+C).
            It should be the last call in your application startup code.

        Raises:
            Various exceptions from data loading, layout creation, or
            server startup may be raised if issues occur.
        """
        # Load and process all dataset files
        for dataset_path, dataset_name in zip(self.dataset_paths, self.dataset_names):
            self.load_data(dataset_path, dataset_name)

        # Initialize all plot generators after datasets are loaded
        self._initialize_figure_generators()

        # Create the Dash application layout
        self.app.layout = self.create_layout()

        # Apply custom CSS for hover effects and styling
        self.app.index_string = hover_css

        # Add custom favicon (SHIELD logo)
        custom_favicon_link = (
            '<link rel="icon" href="/assets/shield.svg" type="image/svg+xml">'
        )
        self.app.index_string = hover_css.replace("{%favicon%}", custom_favicon_link)

        # Register all interactive callbacks
        register_all_callbacks(self)

        print(f"Starting dashboard on http://localhost:{self.port}")

        # Open default web browser after short delay to ensure server is ready
        threading.Timer(
            0.1, lambda: webbrowser.open(f"http://127.0.0.1:{self.port}")
        ).start()

        # Start the Flask development server (blocking call)
        self.app.run(debug=False, host="127.0.0.1", port=self.port)


hover_css = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                .dataset-name-input:hover {
                    border-color: #007bff !important;
                    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
                    transform: scale(1.01) !important;
                }

                .dataset-name-input:focus {
                    border-color: #007bff !important;
                    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
                    outline: 0 !important;
                }

                .color-picker-input:hover {
                    border-color: #007bff !important;
                    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.4) !important;
                    transform: none !important;
                }

                .color-picker-input:focus {
                    border-color: #007bff !important;
                    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.4) !important;
                    outline: 0 !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """
