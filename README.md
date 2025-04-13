# Langchain Multi-Agent Weather Forecast Chatbot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-parameter weather forecast chatbot built with Python using Langchain, Google Gemini, and Langchain's experimental CSV Agents. It queries local CSV files for rainfall, relative humidity, wind speed, and temperature forecasts based on user-provided locations (Village, Mandal, or District) and provides a synthesized, narrative summary.

## Features

*    Conversational Interface:  Ask for forecasts using natural language location names.
*    Multi-Parameter:  Retrieves data for Rainfall, Humidity, Wind Speed, and Temperature.
*    Multi-Agent Architecture:  Uses separate Langchain CSV Agents for each data parameter.
*    LLM Synthesis:  Leverages Google Gemini (via `ChatGoogleGenerativeAI`) to combine agent results into a concise, human-readable forecast.
*    Local Data:  Reads forecast data directly from specified CSV files.
*    Configurable:  Easily update CSV paths and toggle agent verbose mode for debugging.

## Architecture

1.   User Input:  The user provides a location name (Village, Mandal, or District).
2.   Agent Dispatch:  The main script invokes specialized Langchain CSV Agents, each configured for a specific parameter's CSV file (Rainfall, Humidity, Wind Speed, Temperature).
3.   CSV Agent Query:  Each agent uses the Gemini LLM to interpret the request, generate pandas code to query its assigned CSV file for the location, and extracts the relevant forecast data for the available dates.
4.   Data Aggregation:  The main script collects the raw data strings returned by each agent.
5.   LLM Synthesis:  A final prompt containing the aggregated raw data is sent to the Gemini LLM.
6.   Formatted Output:  The LLM synthesizes the information into a concise, narrative forecast report, which is presented to the user.

## Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   Google API Key with Gemini API enabled (See: [Google AI Studio](https://aistudio.google.com/))
*   Forecast data in separate CSV files (one for each parameter).

## Setup

1.   Clone the Repository: 
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.   Create a Virtual Environment (Recommended): 
    ```bash
    python -m venv venv
    # Activate it:
    # Windows:
    # venv\Scripts\activate
    # macOS/Linux:
    # source venv/bin/activate
    ```

3.   Install Dependencies: 
    Create a `requirements.txt` file (see below) and install packages:
    ```bash
    pip install -r requirements.txt
    ```

4.   Set Up Environment Variables: 
    Create a file named `.env` in the root project directory and add your Google API Key:
    ```
    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
    ```
    *Replace `YOUR_GOOGLE_API_KEY_HERE` with your actual key.*

5.   Prepare Data Files: 
    *   Place your CSV forecast files (e.g., `RF_day...csv`, `RH_day...csv`, `WS_day...csv`, `TEMP_day...csv`) accessible to the script.
    *   Ensure the CSV files have columns named `VILLAGE`, `MANDAL`, `DISTRICT`, and date columns (e.g., `DD-MM-YY`).
    *   Update the file path constants at the top of `multi_weather_chatbot.py` to point to the  correct absolute or relative paths  of your CSV files:
        ```python
        RAINFALL_CSV_PATH = "path/to/your/rainfall_forecast.csv"
        HUMIDITY_CSV_PATH = "path/to/your/humidity_forecast.csv"
        WINDSPEED_CSV_PATH = "path/to/your/windspeed_forecast.csv"
        TEMPERATURE_CSV_PATH = "path/to/your/temperature_forecast.csv"
        ```

## Running the Chatbot

Execute the main script from your terminal:

```bash
python multi_weather_chatbot.py
