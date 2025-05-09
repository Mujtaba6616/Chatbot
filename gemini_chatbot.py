import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

# Configure the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# App title and configuration
st.set_page_config(
    page_title="Gemini AI Chatbot",
    page_icon="🧠",
    layout="wide"
)

# Debug: List available models
try:
    available_models = {model.name: model.name for model in genai.list_models()}
    st.sidebar.success(f"Successfully connected to Google Generative AI API")
except Exception as e:
    st.sidebar.error(f"Error connecting to API: {str(e)}")
    st.sidebar.info("Please check your API key and internet connection")
    available_models = {}

# Use models that are actually available in your API
MODELS = {
    "Gemini 1.5 Pro": "models/gemini-1.5-pro",
    "Gemini 1.5 Flash": "models/gemini-1.5-flash",
    "Gemini 2.0 Flash": "models/gemini-2.0-flash",
    "Gemini Pro Vision": "models/gemini-pro-vision",
}

# Personality presets
PERSONALITIES = {
    "Helpful Assistant": "You are a helpful, friendly assistant.",
    "Academic Expert": "You are an academic expert who provides detailed, scholarly responses with citations.",
    "Creative Writer": "You are a creative writer with a flair for engaging storytelling.",
    "Technical Programmer": "You are a technical programming expert who provides concise, efficient code solutions.",
    "Custom": ""  # For custom system prompts
}

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_personality" not in st.session_state:
    st.session_state.current_personality = "Helpful Assistant"
if "gemini_chat" not in st.session_state:
    # Will initialize later after model selection
    pass

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Chatbot Settings")
    
    # Show available models from API
    st.subheader("Available Models")
    if available_models:
        st.write("Models available from API:")
        for model_name in available_models.keys():
            st.write(f"- {model_name}")
    
    # Model selection
    selected_model_name = st.selectbox(
        "Choose a model",
        list(MODELS.keys()),
        index=0
    )
    
    selected_model = MODELS[selected_model_name]
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values make it more deterministic"
    )
    
    # Output length control
    output_tokens = st.slider(
        "Max output tokens",
        min_value=100,
        max_value=2048,
        value=1000,
        step=100,
        help="Maximum length of the response"
    )
    
    # Personality selection
    personality = st.selectbox(
        "Assistant personality",
        list(PERSONALITIES.keys()),
        index=0
    )
    
    # Custom personality input
    custom_personality = ""
    if personality == "Custom":
        custom_personality = st.text_area(
            "Define custom system prompt:",
            value="You are a helpful assistant.",
            height=100
        )
    
    # Apply settings button
    if st.button("Apply Settings"):
        try:
            # Create a new chat session with updated model
            st.session_state.gemini_chat = genai.GenerativeModel(
                model_name=selected_model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": output_tokens,
                }
            ).start_chat(history=[])
            
            # Set personality by sending an initial system message
            system_prompt = PERSONALITIES[personality]
            if personality == "Custom":
                system_prompt = custom_personality
                
            if system_prompt:
                try:
                    # Send system message to initialize the personality
                    st.session_state.gemini_chat.send_message(
                        f"Please act according to this instruction in all future responses: {system_prompt}"
                    )
                except Exception as e:
                    st.error(f"Error setting personality: {str(e)}")
                    
            st.session_state.current_personality = personality
            st.success(f"Settings applied! Using {selected_model_name} with {personality} personality.")
        except Exception as e:
            st.error(f"Error initializing chat: {str(e)}")
            st.info("Try selecting a different model or check your API key.")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        
        # Don't restart chat session yet - wait for user to apply settings
        if "gemini_chat" in st.session_state:
            st.session_state.pop("gemini_chat")
        
        st.success("Conversation cleared!")
    
    # Export conversation
    if st.button("Export Conversation"):
        conversation_json = json.dumps(st.session_state.conversation_history, indent=2)
        st.download_button(
            label="Download Conversation",
            data=conversation_json,
            file_name="conversation_export.json",
            mime="application/json"
        )
    
    st.divider()
    st.caption("Built with Google Gemini API")

# Main app interface
st.title("🧠 Gemini AI Chatbot")

# Display message if no chat session is initialized
if "gemini_chat" not in st.session_state:
    st.info("Please configure your settings and click 'Apply Settings' to start chatting.")

# Function to generate response with Gemini
def generate_response(prompt):
    if "gemini_chat" not in st.session_state:
        return "Please apply settings first to initialize the chat session."
    
    try:
        # Send message and get response using the chat session
        response = st.session_state.gemini_chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Display chat messages
st.subheader("Conversation")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_prompt = st.chat_input("Type your message here...")

# Process user input
if user_prompt:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state.conversation_history.append({"role": "user", "content": user_prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Add a small delay to simulate thinking (optional)
            time.sleep(0.5)
            
            # Get response from Gemini
            response = generate_response(user_prompt)
            
            # Display the response
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.conversation_history.append({"role": "assistant", "content": response})

# Display current settings
if "gemini_chat" in st.session_state:
    st.sidebar.divider()
    st.sidebar.subheader("Current Settings")
    st.sidebar.write(f"**Model:** {selected_model_name}")
    st.sidebar.write(f"**Temperature:** {temperature}")
    st.sidebar.write(f"**Max tokens:** {output_tokens}")
    st.sidebar.write(f"**Personality:** {st.session_state.current_personality}")

# Add usage instructions at the bottom
with st.expander("How to use this chatbot"):
    st.markdown("""
    ### Instructions
    1. **Configure your settings** in the sidebar (model, temperature, tokens, personality)
    2. **Apply settings** using the button in the sidebar
    3. **Type your message** in the input box and press Enter
    4. **Clear the conversation** anytime using the sidebar button
    5. **Export your conversation** to save it for later
    
    ### About Temperature and Tokens
    - **Temperature (0-1)**: Controls randomness. Lower values make responses more deterministic and focused, while higher values make responses more creative and diverse.
    - **Max tokens (100-2048)**: Controls maximum length of responses. Higher values allow for longer responses.
    
    ### Personalities
    Each personality gives the AI different characteristics:
    - **Helpful Assistant**: General-purpose helpful responses
    - **Academic Expert**: Detailed, scholarly responses
    - **Creative Writer**: More creative, narrative-focused responses
    - **Technical Programmer**: Code-focused technical responses
    - **Custom**: Define your own system prompt
    
    ### Troubleshooting
    - If you see errors related to model availability, try selecting a different model
    - Ensure your API key is correct and has access to the Gemini models
    - Check that you have an active internet connection
    """)