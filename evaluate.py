import os
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# Import your compiled graph to test it
from graph import app as agent_graph

# Load environment variables (LANGCHAIN_API_KEY, HUGGINGFACEHUB_API_TOKEN, etc.)
load_dotenv()

# 1. Initialize the LangSmith Client
client = Client()

# 2. Define the Target Predictor Function
def predict_propvoice(inputs: dict):
    """
    This function takes a single query from the LangSmith dataset 
    and passes it through your LangGraph application.
    """
    user_input = inputs["input"]
    
    # Generate a unique thread ID for each test case to ensure state isolation
    config = {"configurable": {"thread_id": f"eval_thread_{hash(user_input)}"}}
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    # Run the graph
    for _ in agent_graph.stream(initial_state, config):
        pass
        
    final_state = agent_graph.get_state(config)
    
    # Extract the AI's final response and routing decision
    ai_msg = final_state.values["messages"][-1]
    response_text = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
    
    # If the router didn't set a next_node (e.g., hit the guardrail), default to 'end'
    next_node = final_state.values.get("next_node", "end")
    
    return {
        "actual_output": response_text, 
        "actual_routing": next_node
    }

# 3. Define the LLM-as-a-Judge Evaluator
# We instantiate a new LLM client specifically for grading
judge_llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash", 
    temperature=0.0, # Zero temperature for strict, deterministic grading
    max_new_tokens=200,
    task='conversational'
)
judge_model = ChatHuggingFace(llm=judge_llm)

# Define the exact JSON schema we want the Judge to output
class JudgeScore(BaseModel):
    score: int = Field(description="Score 1 if the actual routing and output reasonably match the expected behavior. Score 0 if it fails.")
    reasoning: str = Field(description="A brief, 1-sentence explanation for why this score was given.")

judge_parser = JsonOutputParser(pydantic_object=JudgeScore)

judge_prompt = PromptTemplate(
    template="""You are an impartial quality assurance judge evaluating an AI real estate assistant.
    Compare the Actual Output and Routing to the Expected Output and Expected Routing.
    
    User Input: {input}
    
    Expected Routing: {expected_routing}
    Expected Output: {expected_output}
    
    Actual Routing: {actual_routing}
    Actual Output: {actual_output}
    
    Did the AI successfully follow the expected routing and provide an appropriate response?
    {format_instructions}""",
    input_variables=["input", "expected_routing", "expected_output", "actual_routing", "actual_output"],
    partial_variables={"format_instructions": judge_parser.get_format_instructions()},
)

judge_chain = judge_prompt | judge_model | judge_parser

def correctness_evaluator(run, example):
    """
    This custom evaluator is triggered by LangSmith for every test case.
    It passes the expected and actual results to the Judge LLM.
    """
    try:
        result = judge_chain.invoke({
            "input": example.inputs["input"],
            "expected_routing": example.outputs["expected_routing"],
            "expected_output": example.outputs["expected_output"],
            "actual_routing": run.outputs["actual_routing"],
            "actual_output": run.outputs["actual_output"]
        })
        score = result.get("score", 0)
        reasoning = result.get("reasoning", "No reasoning provided.")
    except Exception as e:
        score = 0
        reasoning = f"Evaluation failed to parse JSON: {str(e)}"
        
    # Return the standardized LangSmith feedback dictionary
    return {"key": "accuracy_score", "score": score, "comment": reasoning}

# 4. Execute the Evaluation Suite
if __name__ == "__main__":
    print("🚀 Starting LangSmith Evaluation Pipeline...")
    
    # Ensure this matches the exact name of the dataset you created in LangSmith
    DATASET_NAME = "PropVoice_Golden_v1" 
    
    try:
        experiment_results = evaluate(
            predict_propvoice,
            data=DATASET_NAME,
            evaluators=[correctness_evaluator],
            experiment_prefix="PropVoice-Eval-",
            description="Testing LangGraph routing accuracy and property RAG extraction."
        )
        print("\n✅ Evaluation complete! Open your LangSmith dashboard to view the performance metrics.")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")