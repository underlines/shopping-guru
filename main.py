from pathlib import Path
from typing import Dict, Optional

import streamlit as st
from dotenv import load_dotenv
import os
import asyncio
import requests
import openai
from browser_use import Agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr
from google import genai

# Load environment variables
load_dotenv()

# Define LLM providers and models
LLM_PROVIDERS: Dict[str, list[str]] = {
    "OpenAI": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    "Google Gemini": ["gemini-pro", "gemini-ultra"],
    "Ollama": ["mistral", "llama2", "gemma"],
    "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
    "Anthropic": ["claude-3-haiku", "claude-3-opus"]
}

def get_api_key(provider: str) -> Optional[str]:
    """Retrieve API key from environment variables."""
    env_var = provider.upper().replace(" ", "_") + "_API_KEY"
    return os.getenv(env_var)

def fetch_latest_models(provider: str) -> list[str]:
    """Fetch latest available models for a given provider."""
    try:
        if provider == "OpenAI":
            client = openai.Client()
            models = client.models.list()
            filtered_models = sorted(
                [model.id for model in models if "gpt" in model.id or "o1" in model.id or "o3" in model.id]
            )
            return filtered_models
        elif provider == "Google Gemini":
            client = genai.Client()
            model_info = client.models.get(model="gemini-2.0-flash")
            return [model_info.name]
        elif provider == "Ollama":
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
        elif provider == "DeepSeek":
            return ["deepseek-chat", "deepseek-reasoner"]
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
    return []

def get_model_instance(provider: str, model: str, api_key: str, num_ctx: int = 16000):
    """Return the appropriate model instance for the selected provider."""
    if provider == "OpenAI":
        return ChatOpenAI(model=model, api_key=api_key)
    elif provider == "Ollama":
        return ChatOllama(model=model, temperature=0.0, num_ctx=num_ctx)
    elif provider == "DeepSeek":
        return ChatOpenAI(base_url='https://api.deepseek.com/v1', model=model, api_key=SecretStr(api_key))
    elif provider == "Google Gemini":
        return ChatGoogleGenerativeAI(model=model, api_key=SecretStr(api_key))
    elif provider == "Anthropic":
        return ChatAnthropic(model_name=model, temperature=0.0, timeout=100)
    else:
        raise ValueError("Unsupported provider")

async def run_agent(task: str, llm, use_vision: Optional[bool], save_conversation_path: Optional[str]):
    """Run the Browser-Use agent asynchronously."""
    agent = Agent(
        task=task,
        llm=llm,
        use_vision=use_vision if use_vision is not None else False,
        save_conversation_path=save_conversation_path if save_conversation_path else None
    )
    await agent.run()

def main():
    st.title("LLM Selection Interface")
    
    # Initialize session state for models if not already present
    if 'fetched_models' not in st.session_state:
        st.session_state.fetched_models = {}
    
    # LLM Provider Selection
    provider = st.selectbox("Select LLM Provider", list(LLM_PROVIDERS.keys()))
    
    # Reset models if provider changes
    if 'current_provider' not in st.session_state or st.session_state.current_provider != provider:
        st.session_state.current_provider = provider
        st.session_state.fetched_models[provider] = LLM_PROVIDERS.get(provider, [])
    
    # Model Selection
    models = st.session_state.fetched_models.get(provider, LLM_PROVIDERS.get(provider, []))
    if st.button("Fetch Latest Models"):
        fetched_models = fetch_latest_models(provider)
        st.session_state.fetched_models[provider] = fetched_models
        models = fetched_models
    
    model = st.selectbox("Select a Model", models, index=0)
    custom_model = st.text_input("Or enter a custom model", "")
    selected_model = custom_model if custom_model else model
    
    # API Key Handling
    stored_api_key = get_api_key(provider)
    temp_api_key = st.text_input(
        f"API Key for {provider}",
        value=stored_api_key if stored_api_key else "",
        type="password"
    )
    show_key = st.checkbox("Show API Key")
    if show_key:
        st.write(temp_api_key)
    
    # Number of context tokens (only for Ollama)
    num_ctx = st.number_input("Number of Context Tokens (for Ollama only)", min_value=512, max_value=32768, value=16000)

    # Agent Settings
    use_vision = st.checkbox("Enable Vision Capabilities", help="Enable/disable vision capabilities. Recommended for better web interaction understanding but can increase costs.")
    save_conversation_path = st.text_input("Save Conversation Path", "", help="Path to save the complete conversation history. Useful for debugging.")

    # Instruction Input
    instructions = st.text_area("Enter instructions:")
    
    # Run Button
    if st.button("Run"):
        if temp_api_key:
            llm_instance = get_model_instance(provider, selected_model, temp_api_key, num_ctx)
            asyncio.run(run_agent(instructions, llm_instance, use_vision, save_conversation_path))
        else:
            st.error("API key is required to run the agent.")
    
if __name__ == "__main__":
    main()
