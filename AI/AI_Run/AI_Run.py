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
# Generate response function
def generate_response(prompt: str) -> str:
    try:
        if not prompt.strip():
            return "Input cannot be empty."
        
        prompt = "User: " + prompt + " AI:"
        
        # Tokenize input WITH attention mask
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            add_special_tokens=True
        ).to(device)

        # Extract attention mask
        attention_mask = inputs["attention_mask"]

        # Generate response with attention mask included
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=True,  # Allows randomness
            temperature=0.9,  # Controls randomness level
            top_p=0.9,  # Enables nucleus sampling
            no_repeat_ngram_size=3,  # Prevents repetitive phrases
            repetition_penalty=1.15,  # Penalizes excessive repetition
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode response properly
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
