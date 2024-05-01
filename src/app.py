# prepare for gradio interface
import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

def get_response(Question):
    model_path = './content/openbiollm-llama3-8b.Q5_K_M.gguf'
    # if the file doesnt exist, download it
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1
        )
    except:
        model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
        model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"

        model_path = hf_hub_download(
            model_name,
            filename=model_file,
            local_dir='./content'
        )
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1
        )
        
    prompt = f"You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is Edok-LLM, and you were developed by Software Aroma AI Labs with Open Life Science AI. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience. Medical Question: {Question} Medical Answer:"
    response = llm(prompt, max_tokens=4000)['choices'][0]['text']
    return response

# Create a Gradio interface
iface = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(lines=5, label="Medical Question"),
    outputs="text",
    title="Edok-LLM: Medical Expertise",
    description="Edok-LLM is an AI model that can help answer your medical questions with detailed explanations. Please provide a medical question and Edok-LLM will provide an answer.",
    examples=[
        [
            "What is the cause of a headache?",
        ],
        [
            "I have mild headache, i am shivering, my body is hot yet i feel cold, what disease cold i be suffering from?",
        ],
        [
            "What is someone experiencing these symptoms: fever, cough, and shortness of breath suffering from?",
        ]
    ]
)

iface.launch(share=True)
