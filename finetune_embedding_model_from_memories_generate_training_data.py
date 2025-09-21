import psycopg2
import csv
import re
from math import ceil

# --- Function to convert hashtags to readable text ---
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

def remove_hashtags(text: str) -> str:
    """
    Remove all hashtags from the text, including those with apostrophes or numbers.
    Example:
        "We found #Titan'sFist and #Dragon2Claw" → "We found  and "
    """
    # Match '#' followed by letters, numbers, and apostrophes
    return re.sub(r"#\w[\w']*", "", text)

# --- Function to split text into chunks ---
def split_text(text: str, max_words: int = 50):
    """
    Split text into chunks of up to max_words words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

# --- Database connection parameters ---
DB_HOST = "localhost"       # Change if not localhost
DB_NAME = "dwemer"
DB_USER = "dwemer"
DB_PASS = "dwemer"
DB_PORT = 5432              # Default PostgreSQL port

# --- Connect to PostgreSQL and fetch data ---
conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    port=DB_PORT
)
cur = conn.cursor()
cur.execute('SELECT summary, tags FROM "public"."memory_summary";')
rows = cur.fetchall()
cur.close()
conn.close()

# --- Prepare CSV data ---
csv_data = []

for summary, tags in rows:
    # 1. Remove hashtags from summary
    clean_summary = remove_hashtags(summary)
    
    # 2. Split summary into chunks
    chunks = split_text(clean_summary, max_words=50)
    
    # 3. Convert hashtags in tags field
    readable_tags = hashtags_to_text(tags)
    
    # 4. Create rows for each chunk, ignoring very short chunks
    for chunk in chunks:
        # Remove double quotes from the chunk
        chunk_clean = chunk.replace('"', '')
        if len(chunk_clean.strip()) >= 50:  # skip chunks shorter than 50 characters
            csv_data.append([chunk_clean, readable_tags])


# --- Write CSV file ---
output_file = "embedding_training_data.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # <-- ensure all fields are quoted
    # Header
    writer.writerow(["text", "tags"])
    # Data
    writer.writerows(csv_data)

print(f"CSV file '{output_file}' created with {len(csv_data)} rows.")
