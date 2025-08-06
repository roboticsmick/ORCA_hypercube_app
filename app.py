import os
import dash
from dash import dcc, html, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from netCDF4 import Dataset

# =============================================================================
# 1. APP INITIALIZATION & BANNER
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
    """Builds the top banner with logo and title."""
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
    """Reads the wavelength data from a NetCDF file."""
    filepath = os.path.join(fileFolder, filename)
    with Dataset(filepath, "r") as nc_file:
        wavelengths = nc_file.variables["wavelength"][:]
    return wavelengths


def get_hypercube_data(filename, fileFolder):
    """Reads the full datacube and wavelengths from a NetCDF file."""
    filepath = os.path.join(fileFolder, filename)
    with Dataset(filepath, "r") as nc_file:
        wavelengths = nc_file.variables["wavelength"][:]
        datacube = nc_file.variables["datacube"][:]
    return datacube, wavelengths


def get_rgb_image_figure(red_idx, green_idx, blue_idx, filename, fileFolder):
    """Creates a Plotly figure for the RGB image from specified bands."""
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

    # Stack the bands and normalize for better visualization
    img = np.stack([red_array, green_array, blue_array], axis=-1)

    # Clip outliers and scale to 0-255 for display
    vmin, vmax = np.percentile(img, [1, 99])
    img_clipped = np.clip(img, vmin, vmax)
    img_normalized = (img_clipped - vmin) / (vmax - vmin)
    rgb_img_rescaled = (img_normalized * 255).astype(np.uint8)

    # Create the figure
    fig = px.imshow(rgb_img_rescaled)
    fig.update_layout(
        title="Hypercube RGB Image (Click a pixel to plot spectrum)",
        dragmode="pan",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig, rgb_wavelengths


# =============================================================================
# 3. APP LAYOUT DEFINITION
# =============================================================================


def spectra_display_settings():
    """Defines the settings panel for the app."""
    return html.Div(
        [
            html.H5("SPECTRA DISPLAY PLATFORM"),
            html.Hr(),
            html.Label("1. Enter folder path:", className="input-path-title"),
            dcc.Input(
                id="spectra-folder-input",
                type="text",
                placeholder="e.g., C:/Users/YourUser/Desktop/data",
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
                "3. Set wavelengths for RGB visualization:",
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
    """Creates empty figures for initialization to avoid errors."""
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


initial_image_fig, initial_spectrum_fig = create_initial_figures()

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


# Callback 1: Update file dropdown from folder path
@callback(
    Output("spectra-file-dropdown", "options"), Input("spectra-folder-input", "value")
)
def update_file_dropdown(folder_path):
    if not folder_path or not os.path.isdir(folder_path):
        return []
    try:
        nc_files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]
        return [{"label": f, "value": f} for f in nc_files]
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing folder: {e}")
        return []


# Callback 2: Update sliders when a new file is selected
@callback(
    Output("slider-wl-red", "max"),
    Output("slider-wl-green", "max"),
    Output("slider-wl-blue", "max"),
    Output("slider-wl-red", "value"),
    Output("slider-wl-green", "value"),
    Output("slider-wl-blue", "value"),
    Output("slider-wl-red", "marks"),
    Output("slider-wl-green", "marks"),
    Output("slider-wl-blue", "marks"),
    Input("spectra-file-dropdown", "value"),
    State("spectra-folder-input", "value"),
    prevent_initial_call=True,
)
def update_slider_properties(filename, folder_path):
    if not filename or not folder_path:
        raise PreventUpdate

    wavelengths = get_wavelengths(filename, folder_path)
    max_idx = len(wavelengths) - 1
    marks = {0: f"{wavelengths[0]:.0f}nm", max_idx: f"{wavelengths[-1]:.0f}nm"}
    target_rgb = [640, 530, 445]  # R, G, B nm
    default_indices = [np.abs(np.array(wavelengths) - wl).argmin() for wl in target_rgb]

    return (
        max_idx,
        max_idx,
        max_idx,
        default_indices[0],
        default_indices[1],
        default_indices[2],
        marks,
        marks,
        marks,
    )


# Callback 3: Update RGB image based on slider values
@callback(
    Output("hypercube-image-combined", "figure"),
    Output("rgb-wavelength-values", "children"),
    Input("slider-wl-red", "value"),
    Input("slider-wl-green", "value"),
    Input("slider-wl-blue", "value"),
    State("spectra-file-dropdown", "value"),
    State("spectra-folder-input", "value"),
    prevent_initial_call=True,
)
def update_rgb_image(red_idx, green_idx, blue_idx, filename, folder_path):
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


# Callback 4: Display spectrum on click
@callback(
    Output("pixel-spectrum-combined", "figure"),
    Input("hypercube-image-combined", "clickData"),
    State("spectra-file-dropdown", "value"),
    State("spectra-folder-input", "value"),
    prevent_initial_call=True,
)
def display_pixel_spectrum(click_data, filename, folder_path):
    if click_data is None or not filename or not folder_path:
        raise PreventUpdate

    point = click_data["points"][0]
    y_coord, x_coord = point["y"], point["x"]
    datacube, wavelengths = get_hypercube_data(filename, folder_path)
    spectrum = datacube[:, y_coord, x_coord]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wavelengths, y=spectrum, mode="lines", line=dict(color="#3399f3", width=2)
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


# =============================================================================
# 5. RUN THE APP
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True)

# --- END OF FILE app.py ---# --- START OF FILE app.py ---

import os
import dash
from dash import dcc, html, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from netCDF4 import Dataset

# =============================================================================
# 1. APP INITIALIZATION & BANNER
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
    """Builds the top banner with logo and title."""
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
    """Reads the wavelength data from a NetCDF file."""
    filepath = os.path.join(fileFolder, filename)
    with Dataset(filepath, "r") as nc_file:
        wavelengths = nc_file.variables["wavelength"][:]
    return wavelengths


def get_hypercube_data(filename, fileFolder):
    """Reads the full datacube and wavelengths from a NetCDF file."""
    filepath = os.path.join(fileFolder, filename)
    with Dataset(filepath, "r") as nc_file:
        wavelengths = nc_file.variables["wavelength"][:]
        datacube = nc_file.variables["datacube"][:]
    return datacube, wavelengths


def get_rgb_image_figure(red_idx, green_idx, blue_idx, filename, fileFolder):
    """Creates a Plotly figure for the RGB image from specified bands."""
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

    # Stack the bands and normalize for better visualization
    img = np.stack([red_array, green_array, blue_array], axis=-1)

    # Clip outliers and scale to 0-255 for display
    vmin, vmax = np.percentile(img, [1, 99])
    img_clipped = np.clip(img, vmin, vmax)
    img_normalized = (img_clipped - vmin) / (vmax - vmin)
    rgb_img_rescaled = (img_normalized * 255).astype(np.uint8)

    # Create the figure
    fig = px.imshow(rgb_img_rescaled)
    fig.update_layout(
        title="Hypercube RGB Image (Click a pixel to plot spectrum)",
        dragmode="pan",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig, rgb_wavelengths


# =============================================================================
# 3. APP LAYOUT DEFINITION
# =============================================================================


def spectra_display_settings():
    """Defines the settings panel for the app."""
    return html.Div(
        [
            html.H5("SPECTRA DISPLAY PLATFORM"),
            html.Hr(),
            html.Label("1. Enter folder path:", className="input-path-title"),
            dcc.Input(
                id="spectra-folder-input",
                type="text",
                placeholder="e.g., C:/Users/YourUser/Desktop/data",
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
                "3. Set wavelengths for RGB visualization:",
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
    """Creates empty figures for initialization to avoid errors."""
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


initial_image_fig, initial_spectrum_fig = create_initial_figures()

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


# Callback 1: Update file dropdown from folder path
@callback(
    Output("spectra-file-dropdown", "options"), Input("spectra-folder-input", "value")
)
def update_file_dropdown(folder_path):
    if not folder_path or not os.path.isdir(folder_path):
        return []
    try:
        nc_files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]
        return [{"label": f, "value": f} for f in nc_files]
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing folder: {e}")
        return []


# Callback 2: Update sliders when a new file is selected
@callback(
    Output("slider-wl-red", "max"),
    Output("slider-wl-green", "max"),
    Output("slider-wl-blue", "max"),
    Output("slider-wl-red", "value"),
    Output("slider-wl-green", "value"),
    Output("slider-wl-blue", "value"),
    Output("slider-wl-red", "marks"),
    Output("slider-wl-green", "marks"),
    Output("slider-wl-blue", "marks"),
    Input("spectra-file-dropdown", "value"),
    State("spectra-folder-input", "value"),
    prevent_initial_call=True,
)
def update_slider_properties(filename, folder_path):
    if not filename or not folder_path:
        raise PreventUpdate

    wavelengths = get_wavelengths(filename, folder_path)
    max_idx = len(wavelengths) - 1
    marks = {0: f"{wavelengths[0]:.0f}nm", max_idx: f"{wavelengths[-1]:.0f}nm"}
    target_rgb = [640, 530, 445]  # R, G, B nm
    default_indices = [np.abs(np.array(wavelengths) - wl).argmin() for wl in target_rgb]

    return (
        max_idx,
        max_idx,
        max_idx,
        default_indices[0],
        default_indices[1],
        default_indices[2],
        marks,
        marks,
        marks,
    )


# Callback 3: Update RGB image based on slider values
@callback(
    Output("hypercube-image-combined", "figure"),
    Output("rgb-wavelength-values", "children"),
    Input("slider-wl-red", "value"),
    Input("slider-wl-green", "value"),
    Input("slider-wl-blue", "value"),
    State("spectra-file-dropdown", "value"),
    State("spectra-folder-input", "value"),
    prevent_initial_call=True,
)
def update_rgb_image(red_idx, green_idx, blue_idx, filename, folder_path):
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


# Callback 4: Display spectrum on click
@callback(
    Output("pixel-spectrum-combined", "figure"),
    Input("hypercube-image-combined", "clickData"),
    State("spectra-file-dropdown", "value"),
    State("spectra-folder-input", "value"),
    prevent_initial_call=True,
)
def display_pixel_spectrum(click_data, filename, folder_path):
    if click_data is None or not filename or not folder_path:
        raise PreventUpdate

    point = click_data["points"][0]
    y_coord, x_coord = point["y"], point["x"]
    datacube, wavelengths = get_hypercube_data(filename, folder_path)
    spectrum = datacube[:, y_coord, x_coord]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wavelengths, y=spectrum, mode="lines", line=dict(color="#3399f3", width=2)
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


# =============================================================================
# 5. RUN THE APP
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True)
