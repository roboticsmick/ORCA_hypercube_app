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
import io
import csv
from datetime import datetime, timedelta
import dash
from dash import dcc, html, callback, Output, Input, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import pandas as pd

# =============================================================================
# 1. APP INITIALISATION & BANNER (No Changes)
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=["assets/ORCA_stylesheet.css", dbc.themes.SPACELAB],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/chroma-js/2.1.0/chroma.min.js"
    ],
    suppress_callback_exceptions=True,
)
server = app.server


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-info",
                children=[html.Img(id="logo", src=app.get_asset_url("ORCAlogo.png"))],
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
# 2. HELPER FUNCTIONS FOR DATA HANDLING (No Changes)
# =============================================================================
def get_hypercube_data(filename, fileFolder):
    filepath = os.path.join(fileFolder, filename)
    try:
        with Dataset(filepath, "r") as nc_file:
            wavelengths = nc_file.variables["wavelength"][:]
            datacube = nc_file.variables["datacube"][:]
        return datacube, wavelengths
    except OSError as e:
        print(f"Error reading NetCDF file {filepath}: {e}")
        return None, None


def get_rgb_image_figure(red_idx, green_idx, blue_idx, filename, fileFolder):
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
    img = np.stack([red_array, green_array, blue_array], axis=-1)
    if ma.is_masked(img):
        img = ma.filled(img, np.nan)
    img = np.where(np.isfinite(img), img, np.nan)
    valid_data = img[~np.isnan(img)]
    if len(valid_data) > 0:
        vmin, vmax = np.percentile(valid_data, [1, 99])
    else:
        vmin, vmax = 0, 1
    if vmax == vmin:
        vmax = vmin + 1
    img_clipped = np.clip(img, vmin, vmax)
    img_normalised = (img_clipped - vmin) / (vmax - vmin)
    img_normalised = np.nan_to_num(img_normalised, nan=0.0)
    rgb_img_rescaled = (img_normalised * 255).astype(np.uint8)
    fig = px.imshow(rgb_img_rescaled)
    fig.update_layout(
        title="Hypercube RGB Image (Click a pixel to plot spectrum)",
        dragmode="pan",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig, rgb_wavelengths


def get_single_channel_image_figure(band_idx, filename, fileFolder):
    filepath = os.path.join(fileFolder, filename)
    with Dataset(filepath, "r") as nc_file:
        band_array = nc_file["datacube"][band_idx, :, :]
        wavelength = nc_file["wavelength"][band_idx]
    if ma.is_masked(band_array):
        band_array = ma.filled(band_array, np.nan)
    band_array = np.where(np.isfinite(band_array), band_array, np.nan)
    fig = px.imshow(band_array, color_continuous_scale="gray", aspect="equal")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        title=f"Single Channel View: {wavelength:.1f} nm (Click to plot spectrum)",
        dragmode="pan",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig, wavelength


def find_optimal_rgb_bands(datacube, wavelengths):
    search_windows_nm = {
        "red_wide": (560, 660),
        "red_priority": (620, 660),
        "green": (500, 560),
        "blue": (430, 500),
    }
    RED_STRENGTH_THRESHOLD_RATIO = 0.50

    def get_best_band_in_window(window):
        search_indices = np.where(
            (wavelengths >= window[0]) & (wavelengths <= window[1])
        )[0]
        if len(search_indices) == 0:
            center_wl = (window[0] + window[1]) / 2
            fallback_idx = np.abs(wavelengths - center_wl).argmin()
            return fallback_idx, np.nanstd(datacube[fallback_idx, :, :])
        max_std = -1
        best_idx = search_indices[0]
        for idx in search_indices:
            current_std = np.nanstd(datacube[idx, :, :])
            if current_std > max_std:
                max_std = current_std
                best_idx = idx
        return best_idx, max_std

    green_idx, green_variance_benchmark = get_best_band_in_window(
        search_windows_nm["green"]
    )
    blue_idx, _ = get_best_band_in_window(search_windows_nm["blue"])
    priority_red_idx, priority_red_variance = get_best_band_in_window(
        search_windows_nm["red_priority"]
    )
    if priority_red_variance >= (
        green_variance_benchmark * RED_STRENGTH_THRESHOLD_RATIO
    ):
        final_red_idx = priority_red_idx
    else:
        final_red_idx, _ = get_best_band_in_window(search_windows_nm["red_wide"])
    return (final_red_idx, green_idx, blue_idx)

# =============================================================================
# 3. APP LAYOUT DEFINITION (No Changes)
# =============================================================================
def spectra_display_settings():
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
            html.Label("3. Select Visualisation Mode:", className="input-path-title"),
            dbc.Switch(
                id="mode-toggle-switch",
                label="Single Channel Mode",
                value=False,
                style={"marginTop": "5px", "marginBottom": "15px"},
            ),
            html.Hr(),
            html.Div(
                id="rgb-sliders-container",
                children=[
                    html.Label(
                        "Set wavelengths for RGB visualisation:",
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
                ],
            ),
            html.Div(
                id="single-channel-slider-container",
                style={"display": "none"},
                children=[
                    html.Label(
                        "Set wavelength for single channel view:",
                        className="input-path-title",
                    ),
                    html.Label("Wavelength:", className="input-path-tip"),
                    dcc.Slider(
                        id="slider-wl-single",
                        min=0,
                        max=1,
                        step=1,
                        value=1,
                        marks=None,
                        tooltip={"placement": "bottom"},
                    ),
                ],
            ),
            html.Div(id="rgb-wavelength-values", style={"marginTop": "10px"}),
            html.Hr(),
            dbc.Button(
                "Export Spectrum to CSV",
                id="export-csv-button",
                color="primary",
                className="mt-3",
                style={"width": "100%"},
                disabled=True,
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Export Spectrum")),
                    dbc.ModalBody(
                        [
                            html.P("Enter a filename for the CSV export."),
                            dbc.Input(
                                id="csv-filename-input",
                                placeholder="e.g., my_spectrum",
                                type="text",
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Save",
                            id="save-csv-button",
                            className="ms-auto",
                            n_clicks=0,
                        )
                    ),
                ],
                id="csv-modal",
                is_open=False,
            ),
        ]
    )


def create_initial_figures():
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
                            spectra_display_settings(), id="left-dash-sidebar", width=3
                        ),
                        dbc.Col(
                            [
                                html.Div(id="file-error-alert"),
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
                    ]
                ),
            ],
            fluid=True,
            className="dbc",
        ),
        dcc.Store(id="spectrum-data-store"),
        dcc.Download(id="download-csv"),
    ]
)

# =============================================================================
# 4. CALLBACKS
# =============================================================================
@callback(
    [
        Output("spectra-file-dropdown", "options"),
        Output("pixel-spectrum-combined", "figure", allow_duplicate=True),
    ],
    Input("spectra-folder-input", "value"),
    prevent_initial_call=True,
)
def update_file_dropdown(folder_path):
    if not folder_path or not os.path.isdir(folder_path):
        return [], initial_spectrum_fig
    try:
        nc_files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]
        return [{"label": f, "value": f} for f in nc_files], initial_spectrum_fig
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing folder: {e}")
        return [], initial_spectrum_fig


@callback(
    Output("rgb-sliders-container", "style"),
    Output("single-channel-slider-container", "style"),
    Input("mode-toggle-switch", "value"),
)
def toggle_slider_mode(is_single_channel):
    if is_single_channel:
        return {"display": "none"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "none"}


@callback(
    Output("hypercube-image-combined", "figure"),
    Output("rgb-wavelength-values", "children"),
    [
        Input("slider-wl-red", "value"),
        Input("slider-wl-green", "value"),
        Input("slider-wl-blue", "value"),
        Input("slider-wl-single", "value"),
        Input("mode-toggle-switch", "value"),
        Input("spectra-file-dropdown", "value"),
    ],
    [State("spectra-folder-input", "value")],
    prevent_initial_call=True,
)
def update_main_image(
    red_idx, green_idx, blue_idx, single_idx, is_single_channel, filename, folder_path
):
    trigger_id = ctx.triggered_id
    if not filename or not folder_path or not os.path.isdir(folder_path):
        raise PreventUpdate
    if (
        is_single_channel
        and trigger_id in ["slider-wl-red", "slider-wl-green", "slider-wl-blue"]
    ) or (not is_single_channel and trigger_id == "slider-wl-single"):
        raise PreventUpdate
    if is_single_channel:
        if single_idx is None:
            raise PreventUpdate
        fig, wavelength = get_single_channel_image_figure(
            single_idx, filename, folder_path
        )
        display_text = html.Div(
            [html.Strong("Selected Wavelength (nm): "), f"{wavelength:.1f}"]
        )
        return fig, display_text
    else:
        if not all([red_idx is not None, green_idx is not None, blue_idx is not None]):
            raise PreventUpdate
        fig, rgb_wls = get_rgb_image_figure(
            red_idx, green_idx, blue_idx, filename, folder_path
        )
        display_text = html.Div(
            [
                html.Strong("Selected Wavelengths (nm): "),
                f"R: {rgb_wls[0]:.1f}, G: {rgb_wls[1]:.1f}, B: {rgb_wls[2]:.1f}",
            ]
        )
        return fig, display_text


@callback(
    [
        Output("pixel-spectrum-combined", "figure", allow_duplicate=True),
        Output("spectrum-data-store", "data"),
    ],
    Input("hypercube-image-combined", "clickData"),
    [State("spectra-file-dropdown", "value"), State("spectra-folder-input", "value")],
    prevent_initial_call=True,
)
def display_pixel_spectrum(click_data, filename, folder_path):
    if click_data is None or not filename or not folder_path:
        raise PreventUpdate
    point = click_data["points"][0]
    y_coord, x_coord = point["y"], point["x"]
    filepath = os.path.join(folder_path, filename)
    try:
        with Dataset(filepath, "r") as nc_file:
            datacube = nc_file.variables["datacube"][:]
            wavelengths = nc_file.variables["wavelength"][:]
            exposure_time = (
                nc_file.getncattr("setting_exposure_ms")
                if "setting_exposure_ms" in nc_file.ncattrs()
                else "N/A"
            )
            timestamp_for_row = "N/A (Parsing Failed)"
            if "time" in nc_file.variables:
                time_var = nc_file.variables["time"]
                if y_coord < time_var.shape[0]:
                    try:
                        pd_timestamp = pd.to_datetime(time_var[y_coord])
                        if pd_timestamp.year < 2000:
                            raise ValueError(
                                "Timestamp appears to be a relative offset."
                            )
                        timestamp_for_row = (
                            pd_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                        )
                    except (ValueError, TypeError, AttributeError):
                        try:
                            base_filename = os.path.splitext(filename)[0]
                            start_time = datetime.strptime(
                                base_filename, "%Y_%m_%d-%H_%M_%S"
                            )
                            offset_ms = float(time_var[y_coord])
                            time_delta = timedelta(milliseconds=offset_ms)
                            final_time = start_time + time_delta
                            timestamp_for_row = (
                                final_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                            )
                        except Exception as e:
                            print(f"Timestamp fallback strategy failed: {e}")
                else:
                    timestamp_for_row = "N/A (Index out of bounds)"
    except (OSError, KeyError) as e:
        print(f"Could not read file for spectrum export: {e}")
        raise PreventUpdate
    spectrum = datacube[:, y_coord, x_coord]
    if ma.is_masked(spectrum):
        spectrum = ma.filled(spectrum, np.nan).tolist()
    else:
        spectrum = spectrum.tolist()
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
    stored_data = {
        "wavelengths": wavelengths.tolist(),
        "intensities": spectrum,
        "exposure_time": exposure_time,
        "timestamp": timestamp_for_row,
        "nc_filename": filename,
        "coordinates": f"{y_coord},{x_coord}",
    }
    return fig, stored_data


@callback(
    Output("csv-modal", "is_open"),
    [Input("export-csv-button", "n_clicks"), Input("save-csv-button", "n_clicks")],
    [State("csv-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(n_export, n_save, is_open):
    if n_export or n_save:
        return not is_open
    return is_open


@callback(
    Output("download-csv", "data"),
    Input("save-csv-button", "n_clicks"),
    [State("csv-filename-input", "value"), State("spectrum-data-store", "data")],
    prevent_initial_call=True,
)
def generate_and_download_csv(n_clicks, filename_input, stored_data):
    if not n_clicks or not stored_data:
        raise PreventUpdate
    filename = filename_input.strip() if filename_input else "spectra"
    if not filename.lower().endswith(".csv"):
        filename += ".csv"
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["wavelengths_nm", *stored_data["wavelengths"]])
    writer.writerow(["intensities", *stored_data["intensities"]])
    writer.writerow(["exposure_time_ms", stored_data["exposure_time"]])
    writer.writerow(["timestamp_utc", stored_data["timestamp"]])
    writer.writerow(["source_filename", stored_data["nc_filename"]])
    y, x = stored_data["coordinates"].split(",")
    writer.writerow(["coordinates", f"y={y}, x={x}"])
    csv_string = output.getvalue()
    return dict(content=csv_string, filename=filename)


@callback(
    [
        Output("slider-wl-red", "max"),
        Output("slider-wl-green", "max"),
        Output("slider-wl-blue", "max"),
        Output("slider-wl-single", "max"),
        Output("slider-wl-red", "value"),
        Output("slider-wl-green", "value"),
        Output("slider-wl-blue", "value"),
        Output("slider-wl-single", "value"),
        Output("slider-wl-red", "marks"),
        Output("slider-wl-green", "marks"),
        Output("slider-wl-blue", "marks"),
        Output("slider-wl-single", "marks"),
        Output("file-error-alert", "children"),
        Output("pixel-spectrum-combined", "figure", allow_duplicate=True),
    ],
    Input("spectra-file-dropdown", "value"),
    State("spectra-folder-input", "value"),
    prevent_initial_call=True,
)
def update_slider_properties(filename, folder_path):
    default_slider_state = (1, 1, 1, 1, 1, 1, 1, 1, None, None, None, None)
    if not filename or not folder_path:
        raise PreventUpdate
    datacube, wavelengths = get_hypercube_data(filename, folder_path)
    if datacube is None or wavelengths is None:
        error_message = dbc.Alert(
            f"Error: Could not read '{filename}'. File may be corrupt or invalid.",
            color="danger",
            dismissable=True,
        )
        return *default_slider_state, error_message, initial_spectrum_fig
    max_idx = len(wavelengths) - 1
    marks = {
        int(0): f"{wavelengths[0]:.0f}nm",
        int(max_idx): f"{wavelengths[-1]:.0f}nm",
    }
    red_idx, green_idx, blue_idx = find_optimal_rgb_bands(datacube, wavelengths)
    single_default_idx = green_idx
    return (
        max_idx,
        max_idx,
        max_idx,
        max_idx,
        red_idx,
        green_idx,
        blue_idx,
        single_default_idx,
        marks,
        marks,
        marks,
        marks,
        None,
        initial_spectrum_fig,
    )


@callback(
    Output("export-csv-button", "disabled"),
    [Input("spectrum-data-store", "data"), Input("spectra-file-dropdown", "value")],
)
def set_export_button_state(stored_data, filename):
    trigger = ctx.triggered_id
    if trigger == "spectra-file-dropdown":
        return True
    if trigger == "spectrum-data-store":
        return stored_data is None
    return True

# =============================================================================
# 5. RUN THE APP
# =============================================================================
if __name__ == "__main__":
    app.run(debug=True)
