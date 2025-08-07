# ORCA Hypercube RGB Spectra App

This is a standalone Dash Plotly application for locally viewing and analysing hyperspectral data stored in NetCDF (`.nc`) files. Designed for marine robotics applications, it allows users to generate RGB composite images from any three spectral bands, view single greyscale bands, and inspect the spectral signature of any pixel by clicking on the image.

---

## Features

- Load `.nc` hypercube files from a local folder and handle potentially corrupt files gracefully
- **Smart RGB Selection:** Automatically selects optimal red, green, and blue bands to create a high-contrast, visually informative image by default
- **Single-Channel Mode:** Switch to view a single wavelength as a greyscale image, allowing you to see how much light is reflected at that specific wavelength across the entire scene. This is particularly useful for marine applications—for example, viewing specific wavelengths around 480-520nm can highlight coral fluorescence, making healthy corals stand out clearly against the seafloor in a black and white image
- **Interactive Channel Control:** Manually override the automatic selection using sliders to choose specific wavelengths for either RGB or single-channel mode
- Display corresponding wavelengths (in nm) for selected channels
- Click on any pixel in the image to instantly plot its full spectral signature
- **Export Spectral Data:** Export individual pixel spectra to CSV format with comprehensive metadata including timestamps, exposure settings, and pixel coordinates

### How the Automatic RGB Selection Works

The application automatically selects the best RGB bands to give you a clear, high-contrast image right from the start. This is particularly useful for specialised environments like underwater imaging, where a simple approach often fails.

The process prioritises finding a "true red" band to maintain natural colours:

1. **Find a Reliable Benchmark:** The application first finds the band with the highest variance in the green region (500-560nm). In underwater data, this channel typically provides the most stable signal and serves as a reference point for the current dataset.

2. **Look for True Red First:** The algorithm searches for the band with the highest variance within a narrow "true red" zone (620-660nm).

3. **Evaluate Signal Strength:** The variance of this red candidate is compared to the green channel's benchmark variance.
   - **If the red signal is strong enough** (at least 50% of the green variance), it's selected as the red channel. This preserves natural red colours in the image.
   - **If the red signal is too weak** (likely noisy or heavily attenuated), the algorithm moves to the fallback step.

4. **Fallback to Wider Search:** When the true red signal is insufficient, the algorithm searches a wider window (560-660nm) and selects the band with the highest variance. This might be in the orange or yellow part of the spectrum, but ensures a high-contrast image even when true red is absent.

> **Developer Note:** This behaviour is controlled by the `RED_STRENGTH_THRESHOLD_RATIO` constant in the `find_optimal_rgb_bands` function. A value of `0.5` (meaning red variance must be at least 50% of green variance) has proven to work well across different sensor types and aquatic environments.

---

## Project Structure

The repository is structured as a self-contained Dash application. The `assets` folder is automatically loaded by Dash, so the stylesheet and logo must be placed there.

```
ORCA_hypercube_app/
├── assets/
│   ├── ORCAlogo.png          # Application logo
│   └── ORCA_stylesheet.css   # Custom CSS styles
├── app.py                    # Main Dash application script
├── requirements.txt          # Python dependencies
└── README.md                 # This instruction file
```

---

## Installation and Setup

To run this application on your local machine, you'll need Python 3.8 or newer. We highly recommend using a virtual environment to manage dependencies.

**1. Clone or Download the Repository**

First, get the project files onto your computer:
```bash
git clone https://github.com/roboticsmick/ORCA_hypercube_app.git
cd ORCA_hypercube_app
```

**2. Create and Activate a Virtual Environment**

A virtual environment keeps the project's dependencies separate from your system's global Python installation.

- **On macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

- **On Windows (Command Prompt):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
    
You'll see `(venv)` at the beginning of your command prompt line, indicating the environment is active.

**3. Install Dependencies**

With the virtual environment active, install all the required Python packages:

```bash
pip install -r requirements.txt
```

---

## How to Run the App

Once setup is complete, you can start the application:

```bash
python app.py
```

After running the command, you'll see output in your terminal like this:

```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app'
 * Debug mode: on
```

Open a web browser and navigate to **`http://127.0.0.1:8050/`** to use the application.

---

## How to Use the App

1. **Enter Folder Path:** In the "1. Enter folder path" input box, type or paste the full path to the directory containing your `.nc` files.
   - **Windows:** `C:\Users\username\Documents\hypercube_data` or `C:/hypercube_data`
   - **Linux/macOS:** `~/hypercube_data` or `/home/username/hypercube_data`
2. **Select File:** The "2. Select hypercube file" dropdown will automatically populate with all `.nc` files found in that folder. Choose one to load—an optimised RGB image will be generated automatically.
3. **Select View Mode & Adjust Bands:** The app defaults to an optimised RGB view.
   - Use the **"Single Channel Mode" toggle** to switch to viewing a single wavelength as a greyscale image. This lets you isolate specific spectral features—for instance, certain wavelengths can highlight coral fluorescence, coral health indicators, or algae distribution patterns that aren't obvious in the RGB composite.
   - Use the corresponding slider(s) to manually adjust the selected wavelengths for either mode. The image will update automatically.
4. **Plot Spectrum:** Click anywhere on the image. The graph below will update to show the full reflectance spectrum for the selected pixel.
5. **Export Data:** After clicking on a pixel, the "Export Spectrum to CSV" button becomes available. Click it to:
   - Enter a custom filename for your data export
   - Download a CSV file containing the full spectral data, wavelengths, timestamps, exposure settings, and pixel coordinates
   - The exported data includes metadata from the original NetCDF file, making it suitable for further analysis or sharing

---

## CSV Export Data Format

The exported CSV files contain the following information for each selected pixel:

- **Wavelengths (nm):** Complete wavelength array from the hyperspectral sensor
- **Intensities:** Corresponding reflectance values for each wavelength
- **Exposure Time (ms):** Camera exposure setting when the data was captured
- **Timestamp (UTC):** Time when the specific pixel row was recorded
- **Source Filename:** Original NetCDF file name for data traceability  
- **Coordinates:** Pixel location (x, y) within the image

This comprehensive metadata ensures your exported spectral data can be properly referenced and integrated with other datasets or analysis workflows.
