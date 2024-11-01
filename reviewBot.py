from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
print("initialising the llm")
friendly_llm = ChatOllama(
    model="llama3.1",
    temperature=0.7,
)
casual_llm = ChatOllama(
    model='llama3.1',
    temperature=0.5
)
professional_llm = ChatOllama(
    model='llama3.1',
    temperature=0.2
)
print("ollama loaded")
embedding_model = OllamaEmbeddings(model="llama3.1")
print("embeddings model created")
parser = StrOutputParser()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can use "*" for testing, but restrict to specific URLs in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=200, length_function=len
)


chat_prompt = PromptTemplate.from_template(
    """
        you are a professional review system , your task is to reply to the human reviews after analysing the human sentiement from the input given.
        reply in a warm and professional manner just like how a professional human hotel manager would reply .
        Input :{input}
    """
)
friendly_review_response_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant for a restaurant. Your task is to reply to customer reviews "
            "as if you are a human, in a warm, professional, and personal tone. "
            "Analyze the review and determine if it is positive, neutral, or negative."
            "Make sure each response sounds unique and human-like. Don't sound robotic.",
        ),
        (
            "user",
            "Customer Name: {name}\n"
            "Review: {review_content}\n"
            "manager name : restaurant manager\n"
            "customer rating:{rating}\n"
            "restaurant name :{restaurant_name}",
        ),
        (
            "assistant",
            "Based on the review sentiment, craft a reply as a human restaurant manager would. "
            "Be sure to address concerns, praise compliments, and be understanding. "
            "Use variations in tone and phrasing, so that responses sound human each time."
            "\n\nReply:",
        ),
    ]
)
casual_review_response_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant for a restaurant. Your task is to reply to customer reviews "
            "as if you are a human, in a warm, professional, and personal tone. "
            "Analyze the review and determine if it is positive, neutral, or negative."
            "Make sure each response sounds unique and human-like. Don't sound robotic."
            "the response should not exceed a maximum of 5 SENTENCES"
        ),
        (
            "user",
            "Customer Name: {name}\n"
            "Review: {review_content}\n"
            "manager name : restaurant manager\n"
            "customer rating:{rating}\n"
            "restaurant name :{restaurant_name}",
        ),
        (
            "assistant",
            "Based on the review sentiment, craft a reply as a human restaurant manager would. "
            "Be sure to address concerns, praise compliments, and be understanding. "
            "Use variations in tone and phrasing, so that responses sound human each time."
            "\n\nReply:",
        ),
    ]
)
professional_review_response_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant for a restaurant. Your task is to reply to customer reviews "
            "as if you are a human, in a warm, professional, and personal tone. "
            "Analyze the review and determine if it is positive, neutral, or negative."
            "Make sure each response sounds unique and human-like. Don't sound robotic."
            "the response should not exceed a maximum of 3 SENTENCES"
        ),
        (
            "user",
            "Customer Name: {name}\n"
            "Review: {review_content}\n"
            "manager name : restaurant manager\n"
            "customer rating:{rating}\n"
            "restaurant name :{restaurant_name}",
        ),
        (
            "assistant",
            "Based on the review sentiment, craft a reply as a human restaurant manager would. "
            "Be sure to address concerns, praise compliments, and be understanding. "
            "Use variations in tone and phrasing, so that responses sound human each time."
            "\n\nReply:",
        ),
    ]
)

print("model ready")

restaurant_reviews = {
    "restaurant_name": "The Gourmet Spot",
    "reviews": [
        {
            "review_id": 1,
            "customer_name": "John Doe",
            "rating": 5,
            "review_text": "The food was exceptional! Great ambiance and the staff were very attentive. Will definitely come back!",
            "date": "2023-09-12",
        },
        {
            "review_id": 2,
            "customer_name": "Jane Smith",
            "rating": 4,
            "review_text": "Loved the appetizers, but the main course was a bit too salty for my taste. Overall a good experience.",
            "date": "2023-09-14",
        },
        {
            "review_id": 3,
            "customer_name": "Sam Wilson",
            "rating": 3,
            "review_text": "The service was slow, but the food was decent. Nothing too special.",
            "date": "2023-09-18",
        },
        {
            "review_id": 4,
            "customer_name": "Emily Davis",
            "rating": 5,
            "review_text": "Best dining experience I've had in a while! The chef's special was mind-blowing. Highly recommended!",
            "date": "2023-09-21",
        },
        {
            "review_id": 5,
            "customer_name": "Mark Lee",
            "rating": 2,
            "review_text": "Disappointed. The food was cold when served, and the portions were smaller than expected.",
            "date": "2023-09-22",
        },
    ],
}


firendnly_llm_chain = friendly_review_response_template | friendly_llm | parser
casual_llm_chain = casual_review_response_template | casual_llm | parser
professional_llm_chain = professional_review_response_template | professional_llm | parser

response_json=[]


@app.get('/')
def home():
    return {
        "hello"
    }

@app.post('/friendly_ai')
async def ai(request: Request):
    # Initialize the response list for each request to prevent shared state issues
    response_json = []

    body = await request.json()
    print("Request received on endpoint friendly_ai")
    reviews = body['reviews']
    restaurant_name = body['restaurant_name']

    for no, review in enumerate(reviews):
        name = review['customer_name']
        rating = review['rating']
        text = review['review_text']
        
        print('received data :\n')
        print("{\n", "name:", name, "\n", "rating:", rating, "\n", "review:", text, "\n", "}")

        try:
            # Generate response using the LLM chain
            response = firendnly_llm_chain.invoke({"review_content": text, "name": name, "rating": rating, "restaurant_name": restaurant_name})
        except Exception as e:
            # Print any errors that occur during processing
            print(f"Error generating response for review {no+1}: {e}")
            response = "An error occurred while generating the response."

        print("response", no+1, "generated out of", len(reviews))

        response_json.append({
            "restaurant": restaurant_name,
            "customer_id": review["review_id"],
            "customer": name,
            "date": review["date"],
            "customer_rating": rating,
            "customer_review": text,
            "ai_response": response
        })

        print("{\n", "restaurant_name:", restaurant_name, "\nid:", review["review_id"], "\ncustomer:", name, "\ndate:", review["date"], "\ncustomer_rating:", rating, "\ncustomer_review:", text, "\nai_response:", response, "\n}")

    print("\nfinished processing...\nsuccessfully returned the response")
    return response_json


@app.post('/casual_ai')
async def ai(request: Request):
    # Initialize the response list for each request to prevent shared state issues
    response_json = []

    body = await request.json()
    print("Request received on endpoint casual_ai")
    reviews = body['reviews']
    restaurant_name = body['restaurant_name']

    for no, review in enumerate(reviews):
        name = review['customer_name']
        rating = review['rating']
        text = review['review_text']
        
        print('received data :\n')
        print("{\n", "name:", name, "\n", "rating:", rating, "\n", "review:", text, "\n", "}")

        try:
            # Generate response using the LLM chain
            response = casual_llm_chain.invoke({"review_content": text, "name": name, "rating": rating, "restaurant_name": restaurant_name})
        except Exception as e:
            # Print any errors that occur during processing
            print(f"Error generating response for review {no+1}: {e}")
            response = "An error occurred while generating the response."

        print("response", no+1, "generated out of", len(reviews))

        response_json.append({
            "restaurant": restaurant_name,
            "customer_id": review["review_id"],
            "customer": name,
            "date": review["date"],
            "customer_rating": rating,
            "customer_review": text,
            "ai_response": response
        })

        print("{\n", "restaurant_name:", restaurant_name, "\nid:", review["review_id"], "\ncustomer:", name, "\ndate:", review["date"], "\ncustomer_rating:", rating, "\ncustomer_review:", text, "\nai_response:", response, "\n}")

    print("\nfinished processing...\nsuccessfully returned the response")
    return response_json

@app.post('/professional_ai')
async def ai(request: Request):
    # Initialize the response list for each request to prevent shared state issues
    response_json = []

    body = await request.json()
    print("Request received on endpoint professional ai")
    reviews = body['reviews']
    restaurant_name = body['restaurant_name']

    for no, review in enumerate(reviews):
        name = review['customer_name']
        rating = review['rating']
        text = review['review_text']
        
        print('received data :\n')
        print("{\n", "name:", name, "\n", "rating:", rating, "\n", "review:", text, "\n", "}")

        try:
            # Generate response using the LLM chain
            response = professional_llm_chain.invoke({"review_content": text, "name": name, "rating": rating, "restaurant_name": restaurant_name})
        except Exception as e:
            # Print any errors that occur during processing
            print(f"Error generating response for review {no+1}: {e}")
            response = "An error occurred while generating the response."

        print("response", no+1, "generated out of", len(reviews))

        response_json.append({
            "restaurant": restaurant_name,
            "customer_id": review["review_id"],
            "customer": name,
            "date": review["date"],
            "customer_rating": rating,
            "customer_review": text,
            "ai_response": response
        })

        print("{\n", "restaurant_name:", restaurant_name, "\nid:", review["review_id"], "\ncustomer:", name, "\ndate:", review["date"], "\ncustomer_rating:", rating, "\ncustomer_review:", text, "\nai_response:", response, "\n}")

    print("\nfinished processing...\nsuccessfully returned the response")
    return response_json

