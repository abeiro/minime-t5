import torch
import re
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os


import time
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="Text Processing Service",
    description="A service for text processing including command extraction, topic analysis, and text embeddings",
    version="1.0.0"
)

# Check if GPU is available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if device.type == 'cuda':
    logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
   

# Load the T5 tokenizer and model from the Hugging Face model hub
logger.info("Loading T5 model...")
start_time = time.time()
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", legacy=False, clean_up_tokenization_spaces=True)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
t5_load_time = time.time() - start_time
logger.info(f"T5 model loaded in {t5_load_time:.2f} seconds")

# Load the sentence transformer model
embedding_model = None  # Global variable to hold the model instance

# Global classifier model
word_classifier_model = None
word_classifier_tokenizer = None

CUSTOM_MODEL_PATH = "~custom-MiniLM"  # path to your fine-tuned model


class ExtractRequest(BaseModel):
    text: str

class TextInput(BaseModel):
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
            # "is_command": "ExchangeItems or GiveItemToPlayer or TakeItemsFromPlayer",
            "is_command": "ExchangeItems",
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }
    
   
    # input_text = f"\"{text} (talking to a friend)\"\nis the above sentence an intent to give gold?\nAnswer with either 'yes' or 'no'"
    # input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    # outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    # trigger_command = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
        
    # if "yes" in trigger_command:
    #     return {
    #         "input_text": text,
    #         "is_command": "TakeGoldFromPlayer (amount on target property)",
    #         "elapsed_time": f"{elapsed_time:.2f} seconds"
    #     }
    
    
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
    
   

    # Step 2: OpenInventory
    # input_text = f"\"{text}\"\nis the main verb move or come?\nAnswer with either 'yes' or 'no'"
    # input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    # outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    # trigger_command = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
        
    # if "yes" in trigger_command:
    #     return {
    #         "input_text": text,
    #         "response":trigger_command,
    #         "is_command": "FollowPlayer or LeadTheWayTo",
    #         "elapsed_time": f"{elapsed_time:.2f} seconds"
    #     }
    # 
    # return {
    #     "input_text": text,
    #     "is_command": "Talk",
    #     "elapsed_time": f"{elapsed_time:.2f} seconds"
    # }


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
        print(f"{elapsed_time:.2f} seconds")
        return {
            "input_text": text,
            "prompt": input_text,
            "is_memory_recall": "No",
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Memory req detection: {elapsed_time:.2f} seconds")
 
    input_text = f"\"{text}\"\nExtract sailent words from above paragraph"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    trigger_memory = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Word extraction: {elapsed_time:.2f} seconds {trigger_memory}")
    
    return {
        "input_text": text,
        "generated_tags": "",
        "is_memory_recall": "Yes",
        "elapsed_time": f"{elapsed_time:.2f} seconds",
        "trigger_memory":trigger_memory
    }



@app.get("/detectMemory")
def detectMemory(text: str = Query(..., description="Text to process for keyword extraction")):
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{elapsed_time:.2f} seconds")
        
    if answered==False :
        return {
            "input_text": text,
            "prompt": input_text,
            "is_memory_recall": "No",
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }

    
    return {
        "input_text": text,
        "generated_tags": "",
        "is_memory_recall": "Yes",
        "elapsed_time": f"{elapsed_time:.2f} seconds",
        "trigger_memory":""
    }

@app.get("/topic")
def extract_info(text: str = Query(..., description="Text to process for keyword extraction")):
    start_time = time.time()
    
    # Step 1: Extract Keywords
    #input_text = f"samsum: Dialogue:\n{text}\nWhat were the main points in that conversation? "
    #input_text = f"Here is a dialogue:\n{text}\n\nWrite a short summary!"

    #input_text = f"samsum: Here is a dialogue:\n{text}\n\nWhat were they talking about?"

    #input_text = f"ag_news_subset: Write a short title:\n{text} "
    #input_text = f"newsroom:Here is an article:\n\n{text}\n\nWrite a title for it."
    #input_text = f"Identify the most salient words in this sentence:\n\n{text}"
    input_text = f"What is the subject of the following sentence:\n\n{text}"

    print(input_text)

    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=25, num_beams=5, early_stopping=True)
    generated_tags = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(generated_tags)
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

@app.get("/posttopic")
def extract_info(text: str = Query(..., description="Text to process for keyword extraction")):
    start_time = time.time()
    
    # Step 1: Extract Keywords
    input_text = f"What is the subject of the following sentence:\n\n{text}"
 
    print(input_text)

    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=25, num_beams=5, early_stopping=True)
    generated_tags = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(generated_tags)
    return {
        "input_text": text,
        "generated_tags": generated_tags,
        "elapsed_time": f"{elapsed_time:.2f} seconds"
    }

@app.get("/translate")
def extract_info(text: str = Query(..., description="Text to process for keyword extraction")):
    start_time = time.time()
    
    # Step 1: Extract Keywords
    input_text = f"Translate to english:\n{text} "

    print(input_text)

    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=25, num_beams=5, early_stopping=True)
    generated_tags = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(generated_tags)
    return {
        "input_text": text,
        "generated_tags": generated_tags,
        "elapsed_time": f"{elapsed_time:.2f} seconds"
    }



def hashtags_to_text(text: str) -> str:
    """
    Convert hashtags in a text to readable words, preserving the rest of the text.
    Converts the result to lowercase.
    Example:
        #TundraHideout → tundra hideout
        #DragonSoulAbsorption → dragon soul absorption
        #Titan'sFist → titan's fist
    """
    def replace_tag(match):
        tag = match.group(1)
        # Split CamelCase and words with apostrophes, numbers, or lowercase sequences
        words = re.findall(r"[A-Z][a-z']*|[a-z]+|\d+", tag)
        return " ".join(words)
    
    return re.sub(r"#(\w+)", replace_tag, text).lower()


@app.post("/embed")
async def create_embedding(text_input: TextInput):
    """
    Generate embedding for the input text, converting hashtags to readable words.
    """
    global embedding_model
    try:
        if embedding_model is None:
            logger.info("Loading sentence transformer model for the first time...")
            start_time_model = time.time()

            # Check if custom fine-tuned model exists
            if os.path.exists(CUSTOM_MODEL_PATH):
                logger.info(f"Custom fine-tuned model found at '{CUSTOM_MODEL_PATH}', loading...")
                embedding_model = SentenceTransformer(CUSTOM_MODEL_PATH, device=device)
            else:
                logger.info("Custom model not found, loading default 'all-MiniLM-L6-v2'...")
                embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

            model_load_time = time.time() - start_time_model
            logger.info(f"Sentence transformer model loaded in {model_load_time:.2f} seconds")

        # Transform hashtags into readable text
        cleaned_text = hashtags_to_text(text_input.text)

        # Generate embedding with timing
        logger.info(f"Processing text for embedding: {cleaned_text[:100]}...")  # preview first 100 chars
        start_time = time.time()
        embedding = embedding_model.encode(cleaned_text)
        generation_time = time.time() - start_time

        embedding_list = embedding.tolist()

        logger.info(f"Embedding generated in {generation_time:.4f} seconds")
        
        return {
            "embedding": embedding_list,
            "dimensions": len(embedding_list),
            "timing": {
                "generation_time_seconds": generation_time,
                "text_length": len(cleaned_text)
            },
            "processed_text": cleaned_text
        }
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/tembed")
async def create_embedding(text_input: TextInput):
    """
    Generate embedding for the input text.
    
    Args:
        text_input: TextInput object containing the text to embed
                   (only the part before '|' will be used)
        
    Returns:
        Dictionary containing the embedding vector and timing information
    """
    global embedding_model
    try:
        # Split input on "|" and keep only first part
        text_to_process = text_input.text.split("/", 1)[0].strip()

        # Log request
        logger.info(f"Processing text for embedding: {text_to_process[:250]}...")

        start_time = time.time()

        # Load embedding model if not already loaded
        if embedding_model is None:
            logger.info("Loading sentence transformer model for the first time...")
            start_time_model = time.time()
            embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=device
            )
            model_load_time = time.time() - start_time_model
            logger.info(f"Sentence transformer model loaded in {model_load_time:.2f} seconds")

        # Translate text first
        text_preinput = f"Translate to English literally, word for word:\n\"{text_to_process}\""
        logger.info(f"Text (first 150 chars): {text_preinput[:150]}...")

        input_ids = tokenizer.encode(
            text_preinput,
            return_tensors='pt',
            truncation=True
        ).to(device)

        outputs = model.generate(
            input_ids,
            max_length=min(len(text_to_process) + 50, 50),  # safeguard length
            num_beams=5,
            early_stopping=True
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"Translated text (first 150 chars): {generated_text}...")

        # Generate embedding
        embedding_start = time.time()
        embedding = embedding_model.encode(generated_text)
        generation_time = time.time() - embedding_start

        # Convert to list for JSON serialization
        embedding_list = embedding.tolist()

        # Log completion
        total_time = time.time() - start_time
        logger.info(f"Embedding generated in {generation_time:.4f} seconds (total {total_time:.4f}s)")

        return {
            "embedding": embedding_list,
            "dimensions": len(embedding_list),
            "timing": {
                "generation_time_seconds": generation_time,
                "total_time_seconds": total_time,
                "text_length": len(text_to_process)
            }
        }

    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn.functional as F

# Global vars
word_classifier_model = None
word_classifier_tokenizer = None

@app.get("/wordClasiffier")
def word_classifier(text: str = Query(..., description="Word to classify as noun/verb/other")):
    global word_classifier_model, word_classifier_tokenizer
    start_time = time.time()

    try:
        # Load pretrained POS model on first request
        if word_classifier_model is None or word_classifier_tokenizer is None:
            logger.info("Loading POS tagging BERT model (first time)...")
            word_classifier_tokenizer = AutoTokenizer.from_pretrained(
                "vblagoje/bert-english-uncased-finetuned-pos"
            )
            word_classifier_model = AutoModelForTokenClassification.from_pretrained(
                "vblagoje/bert-english-uncased-finetuned-pos"
            ).to(device)
            logger.info("POS model loaded successfully.")

        # Tokenize word
        inputs = word_classifier_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            is_split_into_words=False
        ).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = word_classifier_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)

        # Decode predicted POS tag for the first token
        pred_id = torch.argmax(probs[0][1]).item()  # skip [CLS] token at index 0
        predicted_pos = word_classifier_model.config.id2label[pred_id]

        # Map to noun/verb/other
        if predicted_pos.startswith("NOUN") or predicted_pos.startswith("PROPN"):
            final_label = "noun"
        elif predicted_pos.startswith("VERB"):
            final_label = "verb"
        else:
            final_label = "other"

        end_time = time.time()
        elapsed_time = end_time - start_time

        return {
            "input_text": text,
            "classification": final_label,
            "pos_tag": predicted_pos,
            "confidence": probs[0][1][pred_id].item(),
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }

    except Exception as e:
        logger.error(f"Error in word classifier: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
