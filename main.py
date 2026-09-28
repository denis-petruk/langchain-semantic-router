from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_ollama import OllamaLLM

# 1. Initialize Local Model
MODEL_NAME = "mistral-nemo:12b"
llm = OllamaLLM(model=MODEL_NAME, temperature=0.01)

# 2. Classifier Setup
router_prompt = PromptTemplate(
    template=('''
    Given the user query below, classify it as exactly one of the following words: `algorithm`, `design`, or `general`.
    Do not output any other text, greetings, or punctuation. Just the single word.
    Query: {query}
    Classification:'''),
    input_variables=['query'],
)

router_chain = router_prompt | llm | StrOutputParser()

# 3. Specialized Destination Prompts
algo_prompt = PromptTemplate(
    template=('''
        You are a strict computer science professor.
        Explain the time and space complexity of this concept, focusing on graph algorithms or sorting methods if applicable.
        Query: {query}'''),
    input_variables=['query'],
)

design_prompt = PromptTemplate(
    template=('''
        You are a precise UI/UX designer.
        Output clean SVG code and focus on typography (Futura/Nunito) and precise hex codes.
        Query: {query}'''),
    input_variables=['query'],
)

general_prompt = PromptTemplate(
    template=('''
    You are a helpful assistant.
    Answer concisely.
    Query: {query}'''),
    input_variables=['query'],
)

# 4. Specialized Destination Chains
algo_chain = algo_prompt | llm | StrOutputParser()
design_chain = design_prompt | llm | StrOutputParser()
general_chain = general_prompt | llm | StrOutputParser()

#5 Writing conditions for destinations
branch = RunnableBranch(
            (lambda x: "algorithm" in x["topic"].lower(), algo_chain),
                        (lambda x: "design" in x["topic"].lower(), design_chain),
                        (general_chain)
                        )

# 6. Master Assembly Chain
master_chain = {
    "topic": router_chain,
    "query": lambda x: x["query"]
    } | branch

if __name__ == "__main__":
    print("--- Test 1 ALGO ---")
    print(master_chain.invoke({"query": "Explain Dijkstra's algorithm and its Big-O time complexity."}))

    print("\n--- TEST 2 DESIGN ---")
    print(master_chain.invoke({"query": "Design a minimal SVG icon for a modern fitness app using #FF4B4B and #1E1E1E."}))

    print("\n--- Test 3: General ---")
    print(master_chain.invoke({"query": "What is the capital of France?"}))
