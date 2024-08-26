import torch
from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time

# Initialize the FastAPI app
app = FastAPI()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    
# Load the T5 tokenizer and model from the Hugging Face model hub
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", legacy=False, clean_up_tokenization_spaces=True)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)

class ExtractRequest(BaseModel):
    text: str


@app.get("/command")
def extract_info(text: str = Query(..., description="Text to process for keyword extraction")):
    start_time = time.time()
    
    
    # Step 0: Take A Seat
    input_text = f"\"{text}\"\nis the above sentence a request to sit?\nAnswer with either 'yes' or 'no' "
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    trigger_command = tokenizer.decode(outputs[0], skip_special_tokens=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
        
    if "yes" in trigger_command:
        return {
            "input_text": text,
            "is_command": "TakeASeat",
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }


    # Step 1: ReadQuestJournal
    input_text = f"\"{text}\"\nis the above sentence a command to read the quest journal?\nAnswer with either 'yes' or 'no'"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    trigger_command = tokenizer.decode(outputs[0], skip_special_tokens=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
        
    if "yes" in trigger_command:
        return {
            "input_text": text,
            "is_command": "ReadQuestJournal",
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }
    
    
    # Step 2: OpenInventory
    input_text = f"\"{text} (talking to a friend)\"\nis the above sentence an intent to give items?\nAnswer with either 'yes' or 'no'"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    trigger_command = tokenizer.decode(outputs[0], skip_special_tokens=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
        
    if "yes" in trigger_command:
        return {
            "input_text": text,
            "is_command": "ExchangeItems",
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }
    
   
    # Step 2: OpenInventory
    input_text = f"\"{text}\"\nis the above sentence an explicit order to attack?\nAnswer with either 'yes' or 'no'"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    trigger_command = tokenizer.decode(outputs[0], skip_special_tokens=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
        
    if "yes" in trigger_command:
        return {
            "input_text": text,
            "is_command": "Attack",
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }
    
    return {
        "input_text": text,
        "is_command": "Talk",
        "elapsed_time": f"{elapsed_time:.2f} seconds"
    }

@app.get("/extract")
def extract_info(text: str = Query(..., description="Text to process for keyword extraction")):
    start_time = time.time()
    print(text)
    
    # Step 0: Trigger memory
    input_text = f"\"{text}\"\nBased on the paragraph above can we conclude that i need to remember something?\nAnswer with either 'yes' or 'no'"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    trigger_memory = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Memory req: " + trigger_memory.lower())
    answered=False;
    if trigger_memory.lower() == "yes":
        answered=True
    
    if answered==False :
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "input_text": text,
            "prompt": input_text,
            "is_memory_recall": "No",
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }
       
        
    # Step 1: Extract Keywords
    input_text = f"Identify the most salient words in this sentence:\n{text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    generated_tags = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Step 2: Identify Topic
    input_text = f"Identify all names in this sentence:\n{text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    generated_tags += "|" + tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Step 3: Identify Subjects
    input_text = f"Identify the topic of this sentence:\n{text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    generated_tags += "|" + tokenizer.decode(outputs[0], skip_special_tokens=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return {
        "input_text": text,
        "generated_tags": generated_tags,
        "is_memory_recall": "Yes",
        "elapsed_time": f"{elapsed_time:.2f} seconds",
        "trigger_memory":trigger_memory
    }

@app.get("/topic")
def extract_info(text: str = Query(..., description="Text to process for keyword extraction")):
    start_time = time.time()
    
    # Step 1: Extract Keywords
    input_text = f"What are the keywords in the following sentence:\n\n{text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    generated_tags = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return {
        "input_text": text,
        "generated_tags": generated_tags,
        "elapsed_time": f"{elapsed_time:.2f} seconds"
    }


@app.get("/task")
def extract_info(text: str = Query(..., description="Text to process for keyword extraction")):
    start_time = time.time()
    
     # Step 3: SetCurrentTask
    input_text = f"bool_q:\"{text}\"\n\nCan we conclude that we have a new goal?"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    trigger_command = tokenizer.decode(outputs[0], skip_special_tokens=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
        
    if "yes" in trigger_command:
        input_text = f"Read the text and answer the question:\"{text}\"\n\nWhat is our goal?"
        input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
        outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
        trigger_command = tokenizer.decode(outputs[0], skip_special_tokens=True)

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            "input_text": text,
            "is_command": "SetCurrentTask@"+trigger_command,
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }
    
    return {
        "input_text": text,
        "generated_tags": "",
        "elapsed_time": f"{elapsed_time:.2f} seconds"
    }

# To run the FastAPI app, save the script as `main.py` and run the following command:
# uvicorn main:app --reload

