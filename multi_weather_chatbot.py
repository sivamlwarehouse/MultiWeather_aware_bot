import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import traceback # For detailed error logging

# --- Configuration ---
# !! UPDATE THESE PATHS to your actual CSV file locations !!
# Use the full paths as you did in your working example
RAINFALL_CSV_PATH = "/Users/sravva/Documents/Test/aware/RF_day2025040812_UTC.csv" # Example Rainfall Path
HUMIDITY_CSV_PATH = "/Users/sravva/Documents/Test/aware/RH_day2025040812_UTC.csv" # Example Humidity Path - UPDATE
WINDSPEED_CSV_PATH = "/Users/sravva/Documents/Test/aware/WS_day2025040812_UTC.csv" # Example Wind Speed Path - UPDATE
TEMPERATURE_CSV_PATH = "/Users/sravva/Documents/Test/aware/T2_day2025040812_UTC.csv" # Example Temperature Path - UPDATE

# --- Debugging Flag ---
# Set to True to see the internal thoughts and actions of each agent
AGENT_VERBOSE_MODE = False # Default to False for cleaner output (Set True to debug agents)

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")

# Configure the Google Generative AI client
genai.configure(api_key=api_key)

# --- Initialize LLM ---
# Using gemini-1.5-flash for a balance of capability and speed.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# --- Helper Function to Create Agents ---
def create_agent_for_csv(csv_path, llm_instance, verbose_mode):
    """Creates a CSV agent for the given file path."""
    agent_name = os.path.basename(csv_path) # Get filename for logging
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}. Agent '{agent_name}' creation skipped.")
        return None
    try:
        agent_executor = create_csv_agent(
            llm=llm_instance,
            path=csv_path,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose_mode, # Use the global flag for debugging
            pandas_kwargs={'encoding': 'utf-8'},
            allow_dangerous_code=True, # Required for pandas execution
            handle_parsing_errors=True, # Attempt to gracefully handle pandas errors
            max_iterations=5, # Give agent slightly more steps if needed
        )
        print(f"CSV Agent created successfully for: {agent_name}")
        return agent_executor
    except Exception as e:
        print(f"Error creating agent for {agent_name}: {e}")
        print(traceback.format_exc())
        return None

# --- Create Individual Agents ---
print("Initializing agents...")
agent_rainfall = create_agent_for_csv(RAINFALL_CSV_PATH, llm, AGENT_VERBOSE_MODE)
agent_humidity = create_agent_for_csv(HUMIDITY_CSV_PATH, llm, AGENT_VERBOSE_MODE)
agent_windspeed = create_agent_for_csv(WINDSPEED_CSV_PATH, llm, AGENT_VERBOSE_MODE)
agent_temperature = create_agent_for_csv(TEMPERATURE_CSV_PATH, llm, AGENT_VERBOSE_MODE) # Added Temperature agent
print("Agent initialization complete.")

# --- Determine Date Range (Assume consistent across files) ---
forecast_start_date = "start date"
forecast_end_date = "end date"
date_columns_list = []
try:
    # Check all defined paths
    valid_paths = [p for p in [RAINFALL_CSV_PATH, HUMIDITY_CSV_PATH, WINDSPEED_CSV_PATH, TEMPERATURE_CSV_PATH] if os.path.exists(p)]
    first_valid_path = valid_paths[0] if valid_paths else None

    if first_valid_path:
        df_dates = pd.read_csv(first_valid_path, encoding='utf-8')
        # Improved date column detection (basic format DD-MM-YY)
        date_columns_list = [col for col in df_dates.columns if isinstance(col, str) and col.count('-') == 2 and all(part.isdigit() for part in col.split('-'))]

        if date_columns_list:
            try:
                 date_columns_list.sort(key=lambda date: pd.to_datetime(date, format='%d-%m-%y', errors='coerce'))
                 date_columns_list = [d for d in date_columns_list if pd.to_datetime(d, format='%d-%m-%y', errors='coerce') is not pd.NaT]
            except Exception as sort_err:
                 print(f"Warning: Could not sort date columns based on DD-MM-YY format: {sort_err}. Using CSV order.")

            if date_columns_list:
                forecast_start_date = date_columns_list[0]
                forecast_end_date = date_columns_list[-1]
                print(f"Detected forecast dates: {forecast_start_date} to {forecast_end_date}")
            else:
                print("Warning: No valid date columns found after attempting to parse/sort.")
        else:
             print("Warning: Could not automatically detect date columns with DD-MM-YY format.")
    else:
        print("Warning: No valid CSV found to determine date range.")
except Exception as e:
    print(f"Warning: Error processing CSV for date range: {e}")
    print(traceback.format_exc())

# --- Chatbot Interaction Logic ---
def chat_with_weather_bot():
    """Handles the conversation loop with the user."""
    global forecast_start_date, forecast_end_date, date_columns_list

    print("\n--- Multi-Parameter Weather Chatbot ---")
    print("Hi! I can provide forecasts for Rainfall, Humidity, Wind Speed, and Temperature.")
    if date_columns_list:
        print(f"Forecasts available from {forecast_start_date} to {forecast_end_date}.")
    else:
        print("Warning: Forecast date range could not be determined.")
    print(">>> Please enter only the Village, Mandal, or District name. <<<")
    print("Type 'quit' or 'exit' to end the chat.")

    while True:
        location_found_somewhere = False
        raw_results = {}

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

            print(f"\nFetching forecast data for '{user_input}'...")

            # --- ** BASE Agent Prompt - Focused on Data Extraction ** ---
            # This prompt directs agents to ONLY extract data and use the loaded DataFrame.
            base_data_prompt = f"""
            You are an agent designed to query a specific CSV data file provided by the toolkit.
            The user wants data for the location: '{user_input}'.

            **CRITICAL INSTRUCTIONS:**
            1.  **You MUST use the pandas DataFrame object that has ALREADY been loaded from the CSV file by the agent toolkit.** This DataFrame contains the full dataset for the specific parameter associated with you.
            2.  **DO NOT generate Python code that creates a NEW DataFrame.** DO NOT use `pd.DataFrame(data={{...}})`. Operate ONLY on the existing DataFrame provided (often `df`).
            3.  Search this existing DataFrame for rows matching '{user_input}'. The search must be case-insensitive and check 'VILLAGE', 'MANDAL', 'DISTRICT' columns. Apply `.str.strip()` before checking with `.str.contains('{user_input}', case=False, na=False)`. Combine conditions with `|`.
            4.  If matching rows are found:
                *   For EACH match, extract 'VILLAGE', 'MANDAL', 'DISTRICT'.
                *   Extract data values from all relevant date columns (like '{forecast_start_date}' to '{forecast_end_date}'). Only use columns present in the DataFrame.
                *   Format EACH match as a string: "Location: [Village], [Mandal], [District]. Data: [Date1]: [Value1], [Date2]: [Value2], ...". Include all date columns found.
                *   Combine strings for multiple matches with newlines (`\n`).
            5.  If NO matching rows are found after searching the loaded DataFrame, return the exact string: "Location '{user_input}' not found."
            6.  Your final response MUST ONLY be the formatted data string(s) from step 4, OR the "not found" message from step 5. No extra text.
            """

            # --- Invoke Agents ---
            # Extended to include Temperature
            agents = {
                "Rainfall": agent_rainfall,
                "Humidity": agent_humidity,
                "Wind Speed": agent_windspeed,
                "Temperature": agent_temperature # Added Temperature
            }

            for param, agent in agents.items():
                if agent:
                    try:
                        print(f"Querying {param} Agent...")
                        agent_specific_prompt = f"Get {param.lower()} data. {base_data_prompt}"
                        response = agent.invoke({"input": agent_specific_prompt})
                        result = response.get('output', f"{param} Agent: Error retrieving output key.")

                        if result and f"Location '{user_input}' not found." not in result and "Error" not in result:
                             raw_results[param] = result.strip()
                             if "Location:" in result and "Data:" in result:
                                 location_found_somewhere = True
                        else:
                             raw_results[param] = result # Store "not found" or error

                    except Exception as e:
                        print(f"Error invoking {param} agent: {e}")
                        print(traceback.format_exc())
                        raw_results[param] = f"{param} forecast unavailable (Agent Query Error)."
                else:
                     raw_results[param] = f"{param} forecast unavailable (Agent not initialized)."


            print("Data retrieval complete. Synthesizing report...")

            # --- Synthesize Results ---
            if not location_found_somewhere:
                print("\nWeather Bot:")
                print(f"Location '{user_input}' not found in any available forecast dataset.")
                continue

            # --- ** Synthesis Prompt - For Combining Results ** ---
            synthesis_prompt = f"""
            You are a weather report compiler. Synthesize the following forecast data fragments for the user query '{user_input}' into a single, easy-to-read report using a simple narrative style.

            Forecast Period: {forecast_start_date} to {forecast_end_date}

            Data Fragments Provided:
            - Rainfall Data String: {raw_results.get("Rainfall", "Not Available")}
            - Humidity Data String: {raw_results.get("Humidity", "Not Available")}
            - Wind Speed Data String: {raw_results.get("Wind Speed", "Not Available")}
            - Temperature Data String: {raw_results.get("Temperature", "Not Available")}

            Instructions for the Final Report:
            1.  Identify the primary location details (Village, Mandal, District) from the fragments containing valid data (look for "Location:" and "Data:"). Use the first complete location found.
            2.  If no valid data fragments exist, state forecast is unavailable.
            3.  Format the report:
                *   Title: **Local Weather Forecast**
                *   Location: **Location:** Village Name, Mandal Name, District Name.
                *   Outlook Period: **Outlook ({forecast_start_date} - {forecast_end_date}):**
                *   Summary Paragraph: Concisely interpret the data from the VALID fragments:
                    *   **Rainfall:** Describe expected rain ("Light showers...", "Dry conditions...", "Rainfall around [Value] units on [Date]..."). State if unavailable.
                    *   **Humidity:** Describe humidity levels ("Relative humidity around X-Y%...", "Higher humidity expected...", "Moderate levels..."). State if unavailable. (Assume %).
                    *   **Wind Speed:** Describe wind conditions ("Winds light/moderate/strong (X-Y units)...", "Strongest winds on [Date]..."). Specify units like km/h if known/implied. State if unavailable.
                    *   **Temperature:** Describe temperature trends ("Temperatures expected around X-Y degrees [specify C/F if known]...", "Cooler towards [Date]...", "Temperatures peaking near..."). State if unavailable.
                    *   Combine these points smoothly into one or two narrative paragraphs.
            4.  Interpret the data, don't just list it. Report ONLY based on provided fragments. Handle missing parameters gracefully. Do not mention the source strings or agents.

            Generate the final synthesized weather report now.
            """

            # Invoke the main LLM for synthesis
            print("Generating final report...")
            final_response = llm.invoke(synthesis_prompt)

            print("\nWeather Bot:")
            print(final_response.content)


        except Exception as e:
            print(f"\nAn critical error occurred in the main loop: {e}")
            print(traceback.format_exc())
            print("Sorry, I encountered a problem processing your request. Please try again.")

# --- Start the Chatbot ---
if __name__ == "__main__":
    active_agents = [agent for agent in [agent_rainfall, agent_humidity, agent_windspeed, agent_temperature] if agent is not None]
    if not active_agents:
        print("\nFatal Error: No forecast agents could be initialized.")
        print("Please check CSV file paths and ensure the files exist and are readable.")
    else:
        print(f"\n{len(active_agents)} agent(s) initialized successfully.")
        chat_with_weather_bot()