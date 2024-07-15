from typing import List, Optional, TypedDict

from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

from langchain_core.runnables import RunnablePassthrough, RunnableConfig

from gen_ui_backend.types import ChatInputType
from gen_ui_backend.tools.github import github_repo
from gen_ui_backend.tools.invoice import invoice_parser
from gen_ui_backend.tools.weather import weather_data
import os
from dotenv import load_dotenv

class GenerativeUIState(TypedDict, total=False):
    input: HumanMessage
    result: Optional[str]
    """Plain text response if no tool was used."""
    tool_calls: Optional[List[dict]]
    """A list of parsed tool calls."""
    tool_result: Optional[dict]
    """The result of a tool call."""

def generate_chain():
    tools_parser = JsonOutputToolsParser()
    initial_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. You're provided a list of tools, and an input from the user.\n"
                + "Your job is to determine whether or not you have a tool which can handle the users input, or respond with plain text.",
            ),
            MessagesPlaceholder("input"),
        ]
    )
    # Get the API key from environment variables
    load_dotenv()
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")    
    azure_open_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not azure_api_key or not azure_endpoint or not azure_open_api_version:
        raise ValueError("Azure OpenAI environment variables not set properly")

    
    # model = AzureChatOpenAI(model="gpt-4o", temperature=0, streaming=True,  openai_api_key=api_key)
    model = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        openai_api_key=azure_api_key,
        openai_api_version=azure_open_api_version, 
        deployment_name="gpt-4o",
        temperature=0,
        streaming=True
    )
    tools = [github_repo, invoice_parser, weather_data]
    model_with_tools = model.bind_tools(tools)
    chain = initial_prompt | model_with_tools
    return chain, tools_parser

chain, tools_parser = generate_chain()

def invoke_model(state: GenerativeUIState, config: RunnableConfig) -> GenerativeUIState:
    
    result = chain.invoke({"input": state["input"]}, config)

    if not isinstance(result, AIMessage):
        raise ValueError("Invalid result from model. Expected AIMessage.")

    if isinstance(result.tool_calls, list) and len(result.tool_calls) > 0:
        parsed_tools = tools_parser.invoke(result, config)
        return {"tool_calls": parsed_tools}
    else:
        return {"result": str(result.content)}


def invoke_tools_or_return(state: GenerativeUIState) -> str:
    if "result" in state and isinstance(state["result"], str):
        return END
    elif "tool_calls" in state and isinstance(state["tool_calls"], list):
        return "invoke_tools"
    else:
        raise ValueError("Invalid state. No result or tool calls found.")


def invoke_tools(state: GenerativeUIState) -> GenerativeUIState:
    tools_map = {
        "github-repo": github_repo,
        "invoice-parser": invoice_parser,
        "weather-data": weather_data,
        # "rank-candiate": rank_candidate,
        # "stock-price-realtime": stock_price_realtime,
    }

    if state["tool_calls"] is not None:
        tool = state["tool_calls"][0]
        selected_tool = tools_map[tool["type"]]
        return {"tool_result": selected_tool.invoke(tool["args"])}
    else:
        raise ValueError("No tool calls found in state.")


def create_graph() -> CompiledGraph:
    
    workflow = StateGraph(GenerativeUIState)

    workflow.add_node("invoke_model", invoke_model)  # type: ignore
    workflow.add_node("invoke_tools", invoke_tools)
    workflow.add_conditional_edges("invoke_model", invoke_tools_or_return)
    workflow.set_entry_point("invoke_model")
    workflow.set_finish_point("invoke_tools")

    graph = workflow.compile()
    return graph

def create_agent():
    def parse_input(input_data: ChatInputType):
        return {"input": input_data["input"]}

    def wrapped_invoke_model(state: GenerativeUIState, config: RunnableConfig):
        return invoke_model(state, config)

    agent_chain = (
        RunnablePassthrough.assign(parsed_input=parse_input)
        | RunnablePassthrough.assign(
            model_result=lambda x: wrapped_invoke_model(
                GenerativeUIState(input=x["parsed_input"]["input"]),
                RunnableConfig()
            )
        )
        | (lambda x: x["model_result"])
    )

    return agent_chain.with_types(input_type=ChatInputType, output_type=dict)

def create_agent2():
    def parse_input(input_data: dict):
        return {"input": input_data["input"]}

    def run_model(parsed_input: dict):
        state = GenerativeUIState(input=parsed_input["input"])
        config = RunnableConfig()
        return invoke_model(state, config)

    def process_result(model_result: dict):
        return model_result

    def agent_function(input_data: dict):
        parsed_data = parse_input(input_data)
        model_result = run_model(parsed_data)
        final_result = process_result(model_result)
        return final_result

    return agent_function