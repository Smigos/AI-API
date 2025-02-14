from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Define paths
BASE_PATH = "D:\AI Stuff\Important"
MODEL_PATH = os.path.join(BASE_PATH, "fine_tuned_gpt2")

# Initialize FastAPI
app = FastAPI()

# Load fine-tuned model
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
print("Model loaded!")

# Request schema
class PromptRequest(BaseModel):
    prompt: str

# Generate response function
def generate_response(prompt: str) -> str:
    try:
        if not prompt.strip():
            return "Input cannot be empty."
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API route to generate a response
@app.post("/generate/")
async def generate(prompt_request: PromptRequest):
    prompt = prompt_request.prompt
    response = generate_response(prompt)
    return {"response": response}

# Root route for testing
@app.get("/")
async def root():
    return {"message": "Welcome to the Conversational AI API!"}
