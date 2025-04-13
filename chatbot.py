import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent # Use the experimental agent



# --- Configuration ---
CSV_FILE_PATH = "/Users/sravva/Documents/Test/aware/RF_day2025040812_UTC.csv" # Make sure this file exists


# --- Load API Key ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")

# Configure the Google Generative AI client
genai.configure(api_key=api_key)

# --- Initialize Components ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3) # Using flash for efficiency

try:
    if not os.path.exists(CSV_FILE_PATH):
         raise FileNotFoundError(f"Error: The file '{CSV_FILE_PATH}' was not found.")

    df_check = pd.read_csv(CSV_FILE_PATH)
    print(f"CSV Data loaded successfully from {CSV_FILE_PATH}. Shape: {df_check.shape}")
    # Extract date columns dynamically for the prompt
    date_columns = [col for col in df_check.columns if '-' in col and col[0].isdigit()] # Basic check for date format DD-MM-YY
    start_date = date_columns[0] if date_columns else "the first forecast date"
    end_date = date_columns[-1] if date_columns else "the last forecast date"
    print(f"Detected forecast dates: {start_date} to {end_date}")


    agent_executor = create_csv_agent(
        llm=llm,
        path=CSV_FILE_PATH,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, # Keep True for debugging, set False for cleaner output
        pandas_kwargs={'encoding': 'utf-8'},
        allow_dangerous_code=True,
    )
    print("CSV Agent created successfully.")

except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"An error occurred during agent creation: {e}")
    exit()

# --- Chatbot Interaction Logic ---

def chat_with_weather_bot():
    """Handles the conversation loop with the user."""
    print("\n--- Weather Chatbot Initialized ---")
    print("Hi! I can provide rainfall forecasts for locations in the dataset.")
    print(">>> Please enter only the Village, Mandal, or District name. <<<")
    print("Type 'quit' or 'exit' to end the chat.")

    # Get date range from the loaded data (used in prompt)
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        date_columns = [col for col in df.columns if '-' in col and col[0].isdigit()]
        forecast_start_date = date_columns[0] if date_columns else "start date"
        forecast_end_date = date_columns[-1] if date_columns else "end date"
    except Exception as e:
        print(f"Warning: Could not dynamically determine date range from CSV: {e}")
        forecast_start_date = "start date"
        forecast_end_date = "end date"


    while True:
        try:
            user_input = input("\nEnter location name (or 'quit'): ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if not user_input:
                print("Please enter a location name.")
                continue

            if ',' in user_input and user_input.count(',') > 2:
                print("Please enter only the location name (Village, Mandal, or District), not the full data row.")
                continue

            # --- REVISED PROMPT FOR SAMPLE FORMAT 1 ---
            agent_prompt = f"""
            You are an agent tasked with retrieving weather forecast data from a specific CSV file located at '{CSV_FILE_PATH}'.
            The user is asking for the forecast for the location: '{user_input}'. The forecast covers dates from {forecast_start_date} to {forecast_end_date}.

            Follow these instructions carefully to retrieve the data and format the response:

            1.  **Data Retrieval:**
                *   Assume the user input '{user_input}' is the name of a VILLAGE, MANDAL, or DISTRICT.
                *   Use the pandas DataFrame ALREADY loaded from the CSV by the agent toolkit. Do NOT create a new DataFrame or use hardcoded data.
                *   Search this DataFrame case-insensitively for rows where the 'VILLAGE', 'MANDAL', or 'DISTRICT' column contains the text '{user_input}'.

            2.  **Response Formatting (If Location Found):**
                *   If you find ONE or MORE matching rows, process EACH match as follows:
                *   Start the response for the match with the title: **Local Rainfall Forecast** (use markdown bold).
                *   On the next line, state the full location clearly: **Location:** Village Name, Mandal Name, District Name. (Extract these from the matched row, use markdown bold for "Location:").
                *   On the next line, provide the forecast summary paragraph:
                    *   Begin with: **Outlook ({forecast_start_date} - {forecast_end_date}):** (use markdown bold for "Outlook...:").
                    *   Analyze the rainfall values in the date columns (e.g., '{forecast_start_date}', '{date_columns[1] if len(date_columns) > 1 else ""}', ..., '{forecast_end_date}') for the matched row.
                    *   Describe the overall pattern using forecast language (e.g., "Rainfall is forecast on...", "Light showers are expected...", "Significant rainfall is predicted...", "No rainfall is anticipated...").
                    *   Mention the specific dates and corresponding rainfall amounts (e.g., "0.02 units on April 10th", "1.17 units for April 11th").
                    *   Group consecutive days with zero rainfall if appropriate (e.g., "No rainfall is anticipated from April 12th through 17th.").
                    *   Keep the summary concise and narrative, like a brief news report.
                *   If there are multiple matches for the user's input, present the formatted forecast for each match, separated by a line like `---`.

            3.  **Response Formatting (If Location Not Found):**
                *   If NO rows are found in the DataFrame after searching the relevant columns for '{user_input}', respond clearly: `Location '{user_input}' not found in the forecast dataset.`

            4.  **Important Constraints:**
                *   Only include the formatted forecast information as described above. Do not include Sno, SP_CODE, LON, LAT, or raw lists of dates/values unless incorporated naturally into the narrative summary.
                *   Generate the Python code to query the DataFrame as needed, but the FINAL output must be the formatted text response described in step 2 or 3.
            """


            response = agent_executor.invoke({"input": agent_prompt})

            print("\nWeather Bot:")
            # Ensure we handle potential markdown in the output if needed
            # For simple bolding, direct printing is usually fine.
            print(response["output"])

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Sorry, I encountered a problem. Please try again.")
            # Consider adding more detailed logging here if issues persist
            # import traceback
            # print(traceback.format_exc())


# --- Start the Chatbot ---
if __name__ == "__main__":
    chat_with_weather_bot()