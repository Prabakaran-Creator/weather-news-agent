import asyncio
from dotenv import load_dotenv
from ddgs import DDGS
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


def search_weather_news(city: str) -> str:
    """Search the web for today's weather news for a city.
    Use this tool when the user wants current weather updates such as
    rain, temperature, storms or forecasts. Not for general trivia."""
    return str(DDGS().news(query=f"{city} weather today", max_results=5))


city = input("Which city? ")
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

agent = AssistantAgent(
    name="weather_news",
    model_client=model_client,
    tools=[search_weather_news],
    system_message="You are a weather news assistant. Always use the search tool first. Then reply with exactly 3 short lines. Each line is one short sentence on its own line. No extra text.",
    reflect_on_tool_use=True,
)


async def main():
    await Console(agent.run_stream(task=f"Give me today's weather news for {city}. Reply in exactly 3 short lines, one sentence per line."))
    await model_client.close()


asyncio.run(main())