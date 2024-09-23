from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM
from src.model.qag_llm import QAGLLM
from src.model.cross_qag_llm import CrossQAGLLM


load_model = {
    "llm": LLM,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
    "qag_llm":QAGLLM,
    "cross_qag_llm":CrossQAGLLM
}

# Replace the following with the model paths
llama_model_path = {
    "7b": "meta-llama/Llama-2-7b-hf",
    "7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "13b_chat": "meta-llama/Llama-2-13b-chat-hf",
}
