# ORCA Hypercube Spectra Viewer

This is a standalone Dash Plotly application for locally viewing and analysing hyperspectral data stored in NetCDF (`.nc`) files. It allows a user to generate an RGB composite image from any three spectral bands and inspect the spectral signature of any pixel by clicking on the image.

---

## Features

-   Load `.nc` hypercube files from a local folder.
-   The app automatically analyses the loaded data to generate an optimal false-color image. This provides a high-contrast, visually informative starting point, especially for challenging data like underwater imagery.
-   **Interactive Channel Control:** Manually override the automatic selection by using sliders to select any three wavelengths (Red, Green, and Blue) to create a custom RGB image. Red, Green, Blue are generally defaulted to [640, 530, 445]
-   Displays the corresponding wavelengths for the selected R, G, and B channels.
-   Click on any pixel in the RGB image to instantly plot its full spectral signature.

### How the Automatic RGB Selection Works

Standard wavelengths (e.g., 640nm for red) are often ineffective in underwater environments, where red light is heavily absorbed, leading to low-contrast or color-skewed images.

The process is as follows:

1.  **Define Spectral Windows:** The code defines three broad spectral windows that correspond to the "Red-ish", "Green-ish", and "Blue-ish" parts of the spectrum (e.g., Red: 560-660nm, Green: 500-560nm, Blue: 430-500nm).

2.  **Calculate Band Variance:** For every spectral band within each of the three windows, the application calculates its statistical **variance**. Variance is a measure of the spread of pixel values and serves as an excellent proxy for the amount of "information" or "contrast" in that band.

3.  **Select the Best Band:** The band with the **highest variance** from each window is chosen as the "champion" for that color.

This method ensures that the default R, G, and B channels are the most information-rich bands from their respective spectral regions, resulting in a clear, high-contrast false-color image that is tailored to the specific data in the file.

---

## Project Structure

The repository is structured to be a self-contained Dash application. The `assets` folder is automatically loaded by Dash, so the stylesheet and logo must be placed there.

```
spectra-viewer-app/
├── assets/
│   ├── ORCAlogo.png          (Application logo)
│   └── ORCA_stylesheet.css   (Custom CSS styles)
├── app.py                    (The main Dash application script)
├── requirements.txt          (Python dependencies)
└── README.md                 (This instruction file)
```

---

## Installation and Setup

To run this application on your local machine, you will need Python 3.8 or newer. It is highly recommended to use a virtual environment to manage dependencies.

**1. Clone or Download the Repository**

First, get the project files onto your computer.
```bash
git clone <your-repository-url>
cd ORCA-spectra-app
```

**2. Create and Activate a Virtual Environment**

A virtual environment keeps the project's dependencies separate from your system's global Python installation.

*   **On macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

*   **On Windows (Command Prompt):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
    
You will see `(venv)` at the beginning of your command prompt line, indicating the environment is active.

**3. Install Dependencies**

With the virtual environment active, install all the required Python packages using pip and the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

## How to Run the App

Once the setup is complete, you can start the application by running the `app.py` script:

```bash
python app.py
```

After running the command, you will see output in your terminal like this:

```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app'
 * Debug mode: on
```

Open a web browser and navigate to the URL **`http://127.0.0.1:8050/`** to use the application.

---

## How to Use the App

1.  **Enter Folder Path:** In the "1. Enter folder path" input box, type or paste the full path to the directory on your computer that contains your `.nc` files.
2.  **Select File:** The "2. Select hypercube file" dropdown will automatically populate with all `.nc` files found in that folder. Choose one to load.
3.  **Adjust RGB Bands:** Use the three sliders to select the band indices for the Red, Green, and Blue channels of the displayed image. The image will update automatically.
4.  **Plot Spectrum:** Click anywhere on the RGB image. The graph below will update to show the full reflectance spectrum for the selected pixel.
