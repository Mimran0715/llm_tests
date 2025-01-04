
from dotenv import load_dotenv
import os

from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

from swarm import Swarm, Agent

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
login(HF_TOKEN)

def test_smolagents():
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())
    agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

def test_swarm():
    client = Swarm()

    def transfer_to_agent_b():
        return agent_b


    agent_a = Agent(
        name="Agent A",
        instructions="You are a helpful agent.",
        functions=[transfer_to_agent_b],
    )

    agent_b = Agent(
        name="Agent B",
        instructions="Only speak in Haikus.",
    )

    response = client.run(
        agent=agent_a,
        messages=[{"role": "user", "content": "I want to talk to agent B."}],
    )

    print(response.messages[-1]["content"])

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-3.5-turbo")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break

if __name__ == "__main__":
    stream_graph_updates("What do you know about LangGraph?")