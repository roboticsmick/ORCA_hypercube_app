# ORCA Hypercube Spectra Viewer

This is a standalone Dash Plotly application for locally viewing and analyzing hyperspectral data stored in NetCDF (`.nc`) files. It allows a user to generate an RGB composite image from any three spectral bands and inspect the spectral signature of any pixel by clicking on the image.

---

## Features

-   Load `.nc` hypercube files from a local folder.
-   Interactively select any three wavelengths (Red, Green, Blue) using sliders to create an RGB image. Wavelengths [640, 530, 445] are chosen by default as RGB.
-   Displays the corresponding wavelengths for the selected R, G, and B channels.
-   Click on any pixel in the RGB image to instantly plot its full spectral signature.

---

## Project Structure

```
ORCA_hypercube_app/
├── assets/
│   ├── ORCAlogo.png          (Application logo)
│   └── ORCA_stylesheet.css   (Custom CSS styles)
├── app.py                    (The main Dash application script)
├── requirements.txt          (Python dependencies)
└── README.md                 (This instruction file)
```

---

## Installation and Setup

Use a virtual environment to manage dependencies.

**1. Clone or Download the Repository**

First, get the project files onto your computer.
```bash
git clone https://github.com/roboticsmick/ORCA_hypercube_app/
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
