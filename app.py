#!/usr/bin/env python3
"""
ORCA Hyperspectral Viewer Application

Purpose: Interactive web application for viewing and analysing hyperspectral image data
stored in nc file format. Allows RGB visualisation and spectral analysis of individual pixels.

Run command: python app.py
Access via: http://127.0.0.1:8050/

Author: Michael Venz
"""

import os
import dash
from dash import dcc, html, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset

# =============================================================================
# 1. APP INITIALISATION & BANNER
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=["assets/ORCA_stylesheet.css", dbc.themes.SPACELAB],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/chroma-js/2.1.0/chroma.min.js"
    ],
    suppress_callback_exceptions=True,
)
server = app.server  # Expose server for deployment


def build_banner():
    """
    Builds the top banner with logo and title.

    Returns:
        html.Div: Banner component with logo and text
    """
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-info",
                children=[
                    html.Img(id="logo", src=app.get_asset_url("ORCAlogo.png")),
                ],
            ),
            html.Div(
                id="banner-text",
                children=[
                    html.H1("OPEN ROBOTICS CONSERVATION AUSTRALIA"),
                    html.H2("LOCAL HYPERSPECTRAL VIEWER"),
                ],
            ),
        ],
    )


# =============================================================================
# 2. HELPER FUNCTIONS FOR DATA HANDLING
# =============================================================================


def get_wavelengths(filename, fileFolder):
    """
    Reads the wavelength data from a NetCDF file.

    Args:
        filename (str): Name of the NetCDF file
        fileFolder (str): Path to the folder containing the file

    Returns:
        numpy.ndarray: Array of wavelength values
    """
    filepath = os.path.join(fileFolder, filename)
    with Dataset(filepath, "r") as nc_file:
        wavelengths = nc_file.variables["wavelength"][:]
    return wavelengths


def get_hypercube_data(filename, fileFolder):
    """
    Reads the full datacube and wavelengths from a NetCDF file.

    Args:
        filename (str): Name of the NetCDF file
        fileFolder (str): Path to the folder containing the file

    Returns:
        tuple: (datacube, wavelengths) arrays
    """
    filepath = os.path.join(fileFolder, filename)
    with Dataset(filepath, "r") as nc_file:
        wavelengths = nc_file.variables["wavelength"][:]
        datacube = nc_file.variables["datacube"][:]
    return datacube, wavelengths


def get_rgb_image_figure(red_idx, green_idx, blue_idx, filename, fileFolder):
    """
    Creates a Plotly figure for the RGB image from specified bands.

    Args:
        red_idx (int): Index for red channel wavelength
        green_idx (int): Index for green channel wavelength
        blue_idx (int): Index for blue channel wavelength
        filename (str): Name of the NetCDF file
        fileFolder (str): Path to the folder containing the file

    Returns:
        tuple: (plotly.graph_objects.Figure, tuple of RGB wavelengths)
    """
    filepath = os.path.join(fileFolder, filename)
    with Dataset(filepath, "r") as nc_file:
        red_array = nc_file["datacube"][red_idx, :, :]
        green_array = nc_file["datacube"][green_idx, :, :]
        blue_array = nc_file["datacube"][blue_idx, :, :]
        rgb_wavelengths = (
            nc_file["wavelength"][red_idx],
            nc_file["wavelength"][green_idx],
            nc_file["wavelength"][blue_idx],
        )

    # Stack the bands into RGB format
    img = np.stack([red_array, green_array, blue_array], axis=-1)

    # Handle masked arrays properly to avoid NumPy warnings
    if ma.is_masked(img):
        # Convert masked array to regular array with NaN for masked values
        img = ma.filled(img, np.nan)

    # Remove any infinite values that could cause issues
    img = np.where(np.isfinite(img), img, np.nan)

    # Calculate percentiles on valid (non-NaN) data only
    valid_data = img[~np.isnan(img)]
    if len(valid_data) > 0:
        vmin, vmax = np.percentile(valid_data, [1, 99])
    else:
        # Fallback if no valid data
        vmin, vmax = 0, 1

    # Ensure we don't divide by zero
    if vmax == vmin:
        vmax = vmin + 1

    # Clip outliers and normalise
    img_clipped = np.clip(img, vmin, vmax)
    img_normalised = (img_clipped - vmin) / (vmax - vmin)

    # Handle any remaining NaN values
    img_normalised = np.nan_to_num(img_normalised, nan=0.0)

    # Scale to 0-255 for display
    rgb_img_rescaled = (img_normalised * 255).astype(np.uint8)

    # Create the figure
    fig = px.imshow(rgb_img_rescaled)
    fig.update_layout(
        title="Hypercube RGB Image (Click a pixel to plot spectrum)",
        dragmode="pan",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig, rgb_wavelengths


def find_optimal_rgb_bands(datacube, wavelengths):
    """
    Finds the optimal R, G, and B bands by maximising variance within defined
    spectral windows. This creates a data-driven, contrast-enhanced RGB image.

    This is especially useful for environments like underwater where standard
    RGB wavelengths may not be effective.

    Args:
        datacube (numpy.ndarray): The hyperspectral data cube.
        wavelengths (numpy.ndarray): The array of wavelengths.

    Returns:
        tuple: A tuple containing the (red_idx, green_idx, blue_idx) of the
               optimal bands.
    """
    # Define spectral windows (in nm) to search for the best R, G, and B bands
    # These windows are chosen to be broad enough to adapt to different water types.
    search_windows_nm = {"red": (560, 660), "green": (500, 560), "blue": (430, 500)}

    optimal_indices = {}

    for color, (min_wl, max_wl) in search_windows_nm.items():
        # Find the indices of the bands within the current search window
        search_indices = np.where((wavelengths >= min_wl) & (wavelengths <= max_wl))[0]

        if len(search_indices) == 0:
            # Fallback: if no bands are in the window, find the closest band to the window's center
            center_wl = (min_wl + max_wl) / 2
            best_idx = np.abs(wavelengths - center_wl).argmin()
            optimal_indices[color] = best_idx
            continue

        max_std = -1
        best_idx = search_indices[0]

        # Iterate through the bands in the window to find the one with max variance
        for idx in search_indices:
            band_data = datacube[idx, :, :]
            # Use nanstd to safely calculate standard deviation, ignoring NaNs
            current_std = np.nanstd(band_data)

            if current_std > max_std:
                max_std = current_std
                best_idx = idx

        optimal_indices[color] = best_idx

    return (optimal_indices["red"], optimal_indices["green"], optimal_indices["blue"])


# =============================================================================
# 3. APP LAYOUT DEFINITION
# =============================================================================


def spectra_display_settings():
    """
    Defines the settings panel for the app.

    Returns:
        html.Div: Settings panel component
    """
    return html.Div(
        [
            html.H5("SPECTRA DISPLAY PLATFORM"),
            html.Hr(),
            html.Label("1. Enter folder path:", className="input-path-title"),
            dcc.Input(
                id="spectra-folder-input",
                type="text",
                placeholder="e.g., /home/user/data",
                style={"width": "100%", "marginBottom": "15px"},
            ),
            html.Hr(),
            html.Label("2. Select hypercube file:", className="input-path-title"),
            dcc.Dropdown(
                id="spectra-file-dropdown",
                options=[],
                value=None,
                placeholder="Select a .nc file...",
                style={"marginBottom": "15px"},
            ),
            html.Hr(),
            html.Label(
                "3. Set wavelengths for RGB visualisation:",
                className="input-path-title",
            ),
            html.Label("Red Channel:", className="input-path-tip"),
            dcc.Slider(
                id="slider-wl-red",
                min=0,
                max=1,
                step=1,
                value=1,
                marks=None,
                tooltip={"placement": "bottom"},
            ),
            html.Label("Green Channel:", className="input-path-tip"),
            dcc.Slider(
                id="slider-wl-green",
                min=0,
                max=1,
                step=1,
                value=1,
                marks=None,
                tooltip={"placement": "bottom"},
            ),
            html.Label("Blue Channel:", className="input-path-tip"),
            dcc.Slider(
                id="slider-wl-blue",
                min=0,
                max=1,
                step=1,
                value=1,
                marks=None,
                tooltip={"placement": "bottom"},
            ),
            html.Hr(),
            html.Div(id="rgb-wavelength-values", style={"marginTop": "10px"}),
        ]
    )


def create_initial_figures():
    """
    Creates empty figures for initialisation to avoid errors.

    Returns:
        tuple: (initial_image_fig, initial_spectrum_fig)
    """
    initial_image_fig = go.Figure()
    initial_image_fig.update_layout(
        template="plotly_dark",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "Please select a file to display the image",
                "showarrow": False,
                "font": {"color": "white"},
            }
        ],
    )

    initial_spectrum_fig = go.Figure()
    initial_spectrum_fig.update_layout(
        template="plotly_dark",
        title="Spectrum Plot",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        annotations=[
            {
                "text": "Click on the image to plot a spectrum",
                "showarrow": False,
                "font": {"color": "white"},
            }
        ],
    )
    return initial_image_fig, initial_spectrum_fig


# Initialise figures
initial_image_fig, initial_spectrum_fig = create_initial_figures()

# Main application layout
app.layout = html.Div(
    [
        build_banner(),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            spectra_display_settings(),
                            id="left-dash-sidebar",
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="hypercube-image-combined",
                                    figure=initial_image_fig,
                                    style={"height": "45vh"},
                                ),
                                html.Hr(),
                                dcc.Graph(
                                    id="pixel-spectrum-combined",
                                    figure=initial_spectrum_fig,
                                    style={"height": "35vh"},
                                ),
                            ],
                            id="right-dash-column",
                            width=9,
                        ),
                    ],
                ),
            ],
            fluid=True,
            className="dbc",
        ),
    ]
)


# =============================================================================
# 4. CALLBACKS
# =============================================================================


@callback(
    Output("spectra-file-dropdown", "options"), Input("spectra-folder-input", "value")
)
def update_file_dropdown(folder_path):
    """
    Updates the file dropdown options based on the selected folder.

    Args:
        folder_path (str): Path to the folder containing NetCDF files

    Returns:
        list: List of dropdown options for .nc files
    """
    if not folder_path or not os.path.isdir(folder_path):
        return []
    try:
        nc_files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]
        return [{"label": f, "value": f} for f in nc_files]
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing folder: {e}")
        return []


@callback(
    [
        Output("hypercube-image-combined", "figure"),
        Output("rgb-wavelength-values", "children"),
    ],
    [
        Input("slider-wl-red", "value"),
        Input("slider-wl-green", "value"),
        Input("slider-wl-blue", "value"),
    ],
    [State("spectra-file-dropdown", "value"), State("spectra-folder-input", "value")],
    prevent_initial_call=True,
)
def update_rgb_image(red_idx, green_idx, blue_idx, filename, folder_path):
    """
    Updates the RGB image based on slider values.

    Args:
        red_idx (int): Red channel wavelength index
        green_idx (int): Green channel wavelength index
        blue_idx (int): Blue channel wavelength index
        filename (str): Selected NetCDF filename
        folder_path (str): Path to the folder containing the file

    Returns:
        tuple: (plotly figure, RGB wavelength text display)
    """
    if not all(
        [
            red_idx is not None,
            green_idx is not None,
            blue_idx is not None,
            filename,
            folder_path,
        ]
    ):
        raise PreventUpdate

    fig, rgb_wls = get_rgb_image_figure(
        red_idx, green_idx, blue_idx, filename, folder_path
    )

    rgb_text = html.Div(
        [
            html.Strong("Selected Wavelengths (nm): "),
            f"R: {rgb_wls[0]:.1f}, G: {rgb_wls[1]:.1f}, B: {rgb_wls[2]:.1f}",
        ]
    )

    return fig, rgb_text


@callback(
    Output("pixel-spectrum-combined", "figure"),
    Input("hypercube-image-combined", "clickData"),
    [State("spectra-file-dropdown", "value"), State("spectra-folder-input", "value")],
    prevent_initial_call=True,
)
def display_pixel_spectrum(click_data, filename, folder_path):
    """
    Displays spectrum plot for clicked pixel coordinates.

    Args:
        click_data (dict): Click event data from image
        filename (str): Selected NetCDF filename
        folder_path (str): Path to the folder containing the file

    Returns:
        plotly.graph_objects.Figure: Spectrum plot figure
    """
    if click_data is None or not filename or not folder_path:
        raise PreventUpdate

    # Extract pixel coordinates from click event
    point = click_data["points"][0]
    y_coord, x_coord = point["y"], point["x"]

    # Load hypercube data
    datacube, wavelengths = get_hypercube_data(filename, folder_path)

    # Extract spectrum for the selected pixel
    spectrum = datacube[:, y_coord, x_coord]

    # Handle masked array data
    if ma.is_masked(spectrum):
        spectrum = ma.filled(spectrum, np.nan)

    # Create spectrum plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wavelengths,
            y=spectrum,
            mode="lines",
            line=dict(color="#3399f3", width=2),
            name=f"Pixel ({x_coord}, {y_coord})",
        )
    )

    fig.update_layout(
        title=f"Spectrum for Pixel (x={x_coord}, y={y_coord})",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="'Chakra Petch', sans-serif"),
    )

    return fig


@callback(
    [
        Output("slider-wl-red", "max"),
        Output("slider-wl-green", "max"),
        Output("slider-wl-blue", "max"),
        Output("slider-wl-red", "value"),
        Output("slider-wl-green", "value"),
        Output("slider-wl-blue", "value"),
        Output("slider-wl-red", "marks"),
        Output("slider-wl-green", "marks"),
        Output("slider-wl-blue", "marks"),
    ],
    Input("spectra-file-dropdown", "value"),
    State("spectra-folder-input", "value"),
    prevent_initial_call=True,
)
def update_slider_properties(filename, folder_path):
    """
    Updates slider properties when a new file is selected.
    to automatically select the optimal R, G, and B channels
    based on variance within spectral windows.
    """
    if not filename or not folder_path:
        raise PreventUpdate

    # We need the full datacube to calculate variance
    datacube, wavelengths = get_hypercube_data(filename, folder_path)
    max_idx = len(wavelengths) - 1

    # Create marks for sliders showing wavelength values
    marks = {
        int(0): f"{wavelengths[0]:.0f}nm",
        int(max_idx): f"{wavelengths[-1]:.0f}nm",
    }

    # Find the best R, G, and B bands based on data
    red_idx, green_idx, blue_idx = find_optimal_rgb_bands(datacube, wavelengths)

    return (
        max_idx,
        max_idx,
        max_idx,
        red_idx,  # Default Red (Optimised)
        green_idx,  # Default Green (Optimised)
        blue_idx,  # Default Blue (Optimised)
        marks,
        marks,
        marks,
    )


# =============================================================================
# 5. RUN THE APP
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True)
