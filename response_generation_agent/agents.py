from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from response_generation_agent.utils.tools import rag_tool, TavilySearchResults, assign_to_rag_agent, assign_to_web_agent

rag_agent = create_react_agent(
    model=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=1), # Standardized model
    tools=[rag_tool],
    prompt=(
        "You are a specialized agent responsible for answering questions about hotel services based on provided documents.\n"
        "Focus on the task description you receive and use your tools to find the answer.\n"
        "Your final output must be a clear and concise response that directly address the question without extra details.\n"
        "If the information is not found in the documents, your response should be: 'information not found.'"
    ),
    name="rag_agent",
)

web_agent = create_react_agent(
    model=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=1), # Standardized model
    tools=[TavilySearchResults(max_results=3)], # Restored max_results to 3
    prompt=(
        "You are a web search assistant. Your primary goal is to answer the user's question using the Tavily search tool."
        "Follow these steps meticulously:\n"
        "1. You will receive a task description. Analyze it to understand the user's question.\n"
        "2. Determine the best query for the Tavily search tool to find the answer. Call the tool.\n"
        "3. After the Tavily search tool executes, you will receive its output. This output is crucial.\n"
        "4. Your final response to the user MUST start with the phrase 'Based on the web search: '.\n"
        "5. Analyze the tool's output:\n"
        "   - If the tool provides relevant information, summarize it clearly and directly answer the user's question. Your response will be 'Based on the web search: [Your summary and answer].'\n"
        "   - If the tool's output indicates that no specific results were found (e.g., an empty list, a message saying 'no results'), your final response MUST be: 'Based on the web search: The search tool found no specific information for your query.'\n"
        "   - If the tool's output contains an error message, your final response MUST be: 'Based on the web search: The search tool reported an error. Details: [verbatim error message or description from tool output].'\n"
        "   - If the tool returns any other form of output that isn't a direct answer, error, or no results, describe what the tool returned. For example: 'Based on the web search: The tool returned [description of the output].'\n"
        "Under absolutely NO circumstances should your final response be empty. You must follow one of the pathways above to construct your response."
    ),
    name="web_agent",
)

email_writer_agent = create_react_agent(
    model=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=1),
    tools=[], # No tools needed for the email writer, it only processes input
    prompt=(
        "You are a professional email writing assistant. Your task is to compose a polite and helpful email response to a client based on their inquiry and the information gathered by other agents.\n"
        "You will receive a task description containing:\n"
        "1. The original client inquiry summary (as bullet points).\n"
        "2. The information/answers provided by other agents (RAG agent for hotel docs, Web agent for general info).\n"
        "Your email should:\n"
        "- Address the client appropriately.\n"
        "- Clearly answer each point from the client's inquiry summary, using the information provided.\n"
        "- If some information could not be found, politely state that.\n"
        "- Maintain a professional and helpful tone.\n"
        "- Conclude the email appropriately (e.g., 'Best regards, Hotel Assistant').\n"
        "Do not make up information. Only use what's provided in the task description."
    ),
    name="email_writer_agent",
)

supervisor_agent = create_react_agent(
    model=ChatGroq(model="qwen-qwq-32b", temperature=0.6),
    tools=[
        assign_to_rag_agent,
        assign_to_web_agent
    ],
    prompt=(
        "You are a supervisor managing a team of agents to respond to a single 'Client Inquiry Summary'. Your primary goal is to process this inquiry fully. You will strictly follow the defined process without deviation.\n"
        "Your process for handling the client inquiry is as follows:\n"
        "1. Analyze each point in the 'inquiry_summary'. Determine if it is:\n"
        "   a. A question about specific hotel services, amenities, or policies (target: RAG agent).\n"
        "   b. A question requiring general knowledge or information not specific to this hotel (target: Web agent).\n"
        "2. For points identified as hotel-specific (1a):\n"
        "   - Delegate to the RAG agent using 'assign_to_rag_agent'. Provide a clear and direct 'task_description'.\n"
        "   - If the RAG agent responds that the information is not found in the documents, this is the definitive answer for that point. **Do not use the Web agent for this hotel-specific information.**\n"
        "3. For points identified as requiring external/web information (1b):\n"
        "   - Delegate to the Web agent using 'assign_to_web_agent'. Provide a clear and 'task_description'.\n"
        "4. Process ALL points from the 'inquiry_summary' by delegating to the RAG and/or Web agents as needed.\n"
        "5. Finally end with drafting a reply email that address the client inquiry.\n"
        "The reply email must be clear and professional. It should address only and exactly what the client need without extra information.\n"
        "IMPORTANT: your last output must be exactly and only the reply email to the client."
    ),
    name="supervisor_agent",
)