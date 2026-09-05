import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY not found. "
        "Please check your .env file."
    )
    st.stop()

client = OpenAI(api_key=api_key)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODEL = "gpt-5.4-mini"

INPUT_PRICE_PER_1M = 0.75
OUTPUT_PRICE_PER_1M = 4.50


# ============================================================
# 3. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LLM Temperature Experiment",
    page_icon="🌡️",
    layout="wide"
)


# ============================================================
# 4. TITLE
# ============================================================

st.title("🌡️ LLM Temperature Experiment")

st.caption(
    "Same Model + Same Prompt → Different Temperature"
)


# ============================================================
# 5. USER PROMPT
# ============================================================

st.subheader("📝 Enter Your Prompt")

prompt = st.text_area(
    "Prompt",
    placeholder=(
        "Enter any prompt here...\n\n"
        "Example:\n"
        "Choose exactly ONE option from A, B, C, D, or E.\n\n"
        "A = Apple\n"
        "B = Mountain\n"
        "C = Ocean\n"
        "D = Robot\n"
        "E = Pizza\n\n"
        "Return ONLY the letter."
    ),
    height=220
)


# ============================================================
# 6. TEMPERATURE SETTINGS
# ============================================================

st.subheader("🌡️ Temperature Settings")

col1, col2, col3 = st.columns(3)

with col1:

    temperature_1 = st.number_input(
        "Temperature 1",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.1
    )

with col2:

    temperature_2 = st.number_input(
        "Temperature 2",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1
    )

with col3:

    temperature_3 = st.number_input(
        "Temperature 3",
        min_value=0.0,
        max_value=2.0,
        value=1.2,
        step=0.1
    )


# ============================================================
# 7. NUMBER OF RUNS
# ============================================================

runs = st.slider(
    "Number of runs for each temperature",
    min_value=1,
    max_value=10,
    value=10
)


# ============================================================
# 8. EXPERIMENT BUTTON
# ============================================================

run_experiment = st.button(
    "🚀 Run Temperature Experiment",
    type="primary",
    use_container_width=True
)


# ============================================================
# 9. RUN EXPERIMENT
# ============================================================

if run_experiment:

    if not prompt.strip():

        st.warning(
            "⚠️ Please enter a prompt before running the experiment."
        )

    else:

        temperatures = [
            temperature_1,
            temperature_2,
            temperature_3
        ]

        # --------------------------------------------------------
        # Save prompt
        # --------------------------------------------------------

        st.session_state.experiment_prompt = prompt

        # --------------------------------------------------------
        # Store experiment results
        # --------------------------------------------------------

        experiment_results = []

        total_experiment_input_tokens = 0
        total_experiment_output_tokens = 0
        total_experiment_tokens = 0
        total_experiment_cost = 0.0

        progress = st.progress(0)

        # 3 temperatures × 10 runs = 30 API calls
        total_calls = len(temperatures) * runs

        current_call = 0


        # ========================================================
        # LOOP THROUGH TEMPERATURES
        # ========================================================

        for temperature in temperatures:

            # Run SAME prompt 10 times
            for run_number in range(1, runs + 1):

                current_call += 1

                progress.progress(
                    current_call / total_calls
                )

                try:

                    # ------------------------------------------------
                    # CALL OPENAI
                    # ------------------------------------------------

                    response = client.responses.create(
                        model=MODEL,
                        input=prompt,
                        temperature=temperature
                    )


                    # ------------------------------------------------
                    # GET RESPONSE
                    # ------------------------------------------------

                    answer = response.output_text


                    # ------------------------------------------------
                    # GET TOKEN USAGE
                    # ------------------------------------------------

                    input_tokens = response.usage.input_tokens

                    output_tokens = response.usage.output_tokens

                    total_tokens = response.usage.total_tokens


                    # ------------------------------------------------
                    # COST CALCULATION
                    # ------------------------------------------------

                    input_cost = (
                        input_tokens / 1_000_000
                    ) * INPUT_PRICE_PER_1M

                    output_cost = (
                        output_tokens / 1_000_000
                    ) * OUTPUT_PRICE_PER_1M

                    total_cost = (
                        input_cost + output_cost
                    )


                    # ------------------------------------------------
                    # ADD TO TOTAL EXPERIMENT
                    # ------------------------------------------------

                    total_experiment_input_tokens += input_tokens

                    total_experiment_output_tokens += output_tokens

                    total_experiment_tokens += total_tokens

                    total_experiment_cost += total_cost


                    # ------------------------------------------------
                    # STORE RESULT
                    # ------------------------------------------------

                    experiment_results.append(
                        {
                            "temperature": temperature,
                            "run": run_number,
                            "answer": answer,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "cost": total_cost
                        }
                    )


                except Exception as e:

                    experiment_results.append(
                        {
                            "temperature": temperature,
                            "run": run_number,
                            "answer": f"API ERROR: {str(e)}",
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "cost": 0
                        }
                    )


        # --------------------------------------------------------
        # Remove progress bar
        # --------------------------------------------------------

        progress.empty()


        # ========================================================
        # SAVE RESULTS IN SESSION
        # ========================================================

        st.session_state.experiment_results = experiment_results

        st.session_state.total_input_tokens = (
            total_experiment_input_tokens
        )

        st.session_state.total_output_tokens = (
            total_experiment_output_tokens
        )

        st.session_state.total_tokens = (
            total_experiment_tokens
        )

        st.session_state.total_cost = (
            total_experiment_cost
        )


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

if "experiment_results" in st.session_state:

    st.divider()

    st.header("📊 Experiment Results")


    # ========================================================
    # SHOW PROMPT
    # ========================================================

    st.subheader("📝 Prompt Used")

    st.code(
        st.session_state.experiment_prompt,
        language="text"
    )


    # ========================================================
    # OVERALL TOKEN SUMMARY
    # ========================================================

    st.subheader("📈 Overall Experiment Usage")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Input Tokens",
        f"{st.session_state.total_input_tokens:,}"
    )

    col2.metric(
        "Output Tokens",
        f"{st.session_state.total_output_tokens:,}"
    )

    col3.metric(
        "Total Tokens",
        f"{st.session_state.total_tokens:,}"
    )

    col4.metric(
        "Estimated Cost",
        f"${st.session_state.total_cost:.6f}"
    )


    # ========================================================
    # SEPARATE RESULTS BY TEMPERATURE
    # ========================================================

    results = st.session_state.experiment_results

    temperatures = sorted(
        list(
            set(
                result["temperature"]
                for result in results
            )
        )
    )


    # ========================================================
    # DISPLAY EACH TEMPERATURE
    # ========================================================

    columns = st.columns(
        len(temperatures)
    )


    for index, temperature in enumerate(temperatures):

        with columns[index]:

            st.subheader(
                f"🌡️ Temperature = {temperature}"
            )

            temperature_results = [
                result
                for result in results
                if result["temperature"] == temperature
            ]


            for result in temperature_results:

                st.markdown(
                    f"### Run {result['run']}"
                )


                # ------------------------------------------------
                # RESPONSE
                # ------------------------------------------------

                st.write(
                    result["answer"]
                )


                # ------------------------------------------------
                # TOKEN INFORMATION
                # ------------------------------------------------

                st.caption(
                    f"Input: {result['input_tokens']:,} | "
                    f"Output: {result['output_tokens']:,} | "
                    f"Total: {result['total_tokens']:,} | "
                    f"Cost: ${result['cost']:.6f}"
                )


                st.divider()


# ============================================================
# 11. EXPLANATION
# ============================================================

st.divider()

st.header("🎓 What Are We Demonstrating?")

st.markdown(
    """
### Same Prompt

The exact same prompt is sent to the model every time.

### Different Temperature

Only the temperature value changes.

### Low Temperature

The model tends to favor higher-probability choices,
so repeated responses are generally more consistent.

### Higher Temperature

The probability distribution is adjusted so that
lower-probability choices become relatively more likely,
which can produce more variation.

### Important

Temperature does **not** make the model more intelligent.

It controls the variability of token selection.

The experiment runs the same prompt multiple times
so we can observe this variation.
"""
)