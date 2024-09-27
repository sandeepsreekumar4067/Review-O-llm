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

print("initialising the llm")
llm = ChatOllama(
    model="llama3.1",
    temperature=0.7,
)
print("ollama loaded")
embedding_model = OllamaEmbeddings(model="llama3.1")
print("embeddings model created")
parser = StrOutputParser()


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
review_response_template = ChatPromptTemplate.from_messages(
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
            "Customer Name: {name}\n" "Review: {review_content}\n" "manager name : restaurant manager",
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
examples = {
    "contextual": [
        "good","bad","wait","long","wouldnt reccomend"
    ],
    "casual": [
        "hi","hello","hey","oh"
    ]
}
threshold = 1.4
legal_embeddings = embedding_model.embed_documents(examples["contextual"])
casual_embeddings = embedding_model.embed_documents(examples["casual"])
def classify_query(query):
    # Embed the query
    query_embedding = embedding_model.embed_documents([query])

    # Calculate similarities

    legal_similarity = cosine_similarity(query_embedding, legal_embeddings)
    casual_similarity = cosine_similarity(query_embedding, casual_embeddings)

    # Get the maximum similarity score for legal and casual categories
    max_legal_similarity = max(legal_similarity[0])
    max_casual_similarity = max(casual_similarity[0])

    # Set a threshold for classifying the query
    threshold = 0.4

    # Compare similarities to classify as legal or casual
    if max_legal_similarity > threshold and max_legal_similarity > max_casual_similarity:
        return {"status":"legal","score":max_legal_similarity}
    else:
        return {"status":"casual","score":max_casual_similarity}

print("model ready")
chain = review_response_template | llm | parser
while 1:
    name = input("Enter the name :")
    if name.lower() == 'bye':
        break
    review = input("Enter the review :")
    response = chain.invoke({"review_content": review,"name":name})
    print(response)
