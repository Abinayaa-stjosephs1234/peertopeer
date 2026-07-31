# p2p_learning_platform_revised.py
import time
import re
import nltk
import os

from p2pnetwork.node import Node

# --- NLP and QA Dependencies ---
from youtube_transcript_api import YouTubeTranscriptApi
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# CORRECTED LINE: This is how multiple imports should be structured
from langchain_community.vectorstores import Chroma
# The code will download the model on first run, so expect a delay!

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 

# --- NLP & QA FUNCTIONS ---

def get_transcript_and_summarize(youtube_url, summary_sentences=3):
    """Fetches transcript, cleans it, and generates a summary."""
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
    video_id = video_id.group(1) if video_id else None

    if not video_id:
        return None, None, "Invalid YouTube URL."

    try:
        # 1. Transcription 
        # Using a reliable method for quick hackathon success
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        full_text = ' '.join([item['text'] for item in transcript_list])
        
        # 2. Summarization 
        parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = ' '.join([str(sentence) for sentence in summarizer(parser.document, summary_sentences)])
        
        return video_id, full_text, summary
        
    except Exception as e:
        return video_id, None, f"Error processing video: {e}"

def setup_qa_system(text_data, node_id):
    """
    Creates an IN-MEMORY vector store for a node's knowledge base.
    This avoids file system conflicts common in hackathon environments.
    """
    # 1. Split the text into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents([text_data], [{"source": f"{node_id}'s Knowledge"}])
    
    # 2. Create Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Create the vector store IN-MEMORY (no persist directory needed)
    vector_store = Chroma.from_documents(
        docs, 
        embeddings, 
        collection_name=f"{node_id}_kb"
    )
    
    return vector_store

def ask_question(vector_store, question):
    """Retrieves the most relevant document chunk to answer the question."""
    # Retrieve the top 1 most relevant chunk
    results = vector_store.similarity_search(question, k=1)
    
    if results:
        # The answer is the content of the most relevant retrieved chunk
        return results[0].page_content
    else:
        return "I could not find any relevant information in the knowledge base."


# --- P2P NODE CLASS ---

class LearningNode(Node):
    def __init__(self, host, port, node_id, youtube_url=None):
        super(LearningNode, self).__init__(host, port, node_id)
        print(f"Node '{node_id}' started on {host}:{port}")
        self.qa_store = None
        self.video_summary = "No video loaded."
        self.is_knowledge_source = bool(youtube_url)
        
        # Initialize the knowledge base if a URL is provided (NodeA)
        if self.is_knowledge_source:
            self._load_knowledge(youtube_url)

    def _load_knowledge(self, url):
        """Internal function to process video and set up QA system."""
        print(f"Processing video from URL: {url}")
        video_id, raw_text, summary_or_error = get_transcript_and_summarize(url, summary_sentences=5)
        
        if raw_text:
            self.video_summary = summary_or_error
            self.qa_store = setup_qa_system(raw_text, self.id)
            print(f"✅ Knowledge Base Ready with {len(raw_text)} characters. Summary: {self.video_summary[:100]}...")
        else:
            self.video_summary = f"Error: {summary_or_error}"
            print(f"❌ Failed to load knowledge. Error: {summary_or_error}")

    # --- P2P Network Event Handlers ---
    
    def outbound_node_connected(self, node):
        print(f"-> Outbound connected to {node.host}:{node.port}")

    def node_message(self, node, data):
        """Handle incoming messages from peers (e.g., QA requests)."""
        msg_type = data.get("type", "UNKNOWN")
        msg_content = data.get("content", "N/A")
        
        print(f"\n--- RECEIVED MESSAGE from {node.id} ({msg_type}) at {time.time():.2f} ---")

        if msg_type == "QA_REQUEST":
            print(f"Question: {msg_content}")
            
            if self.qa_store:
                # 1. Search the local QA system
                retrieved_answer = ask_question(self.qa_store, msg_content)
                
                # 2. Respond to the peer
                response_content = {
                    "question": msg_content,
                    "answer_chunk": retrieved_answer,
                    "summary": self.video_summary
                }
            else:
                response_content = {"error": "Knowledge base not loaded on this node."}
                
            self.send_to_node(node, {
                "type": "QA_RESPONSE",
                "content": response_content,
                "timestamp": time.time()
            })
            
        elif msg_type == "QA_RESPONSE":
            # Display the answer received from NodeA
            content = msg_content
            print(f"Answer received for: '{content.get('question', 'N/A')}'")
            print("\n-----------------------------------------------------")
            print("Video Summary:")
            print(content.get('summary', 'N/A'))
            print("\nRelevant Answer Chunk:")
            print(content.get('answer_chunk', 'N/A'))
            print("-----------------------------------------------------")


# --- EXECUTION: RUN IN TWO TERMINALS ---

if __name__ == '__main__':
    # A video for the knowledge base
    VIDEO_URL = "https://www.youtube.com/watch?v=F3Q3Y709c0g" 

    print("--- P2P LEARNING PLATFORM STARTUP ---")
    print("Run this script in two separate terminals.")
    print("---------------------------------------")

    # Ensure NLTK is downloaded once
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK punkt data...")
        nltk.download('punkt', quiet=True)


    mode = input("Enter 'A' to run NodeA (Knowledge Source) or 'B' to run NodeB (Questioner): ").upper()

    if mode == 'A':
        # Node A: Hosts the video knowledge
        node_a = LearningNode("127.0.0.1", 8001, "NodeA", youtube_url=VIDEO_URL)
        node_a.start()
        
        # Keep NodeA running indefinitely to receive connections/requests
        try:
            while True:
                time.sleep(1) 
        except KeyboardInterrupt:
            print("\nNodeA stopped.")
            node_a.stop()


    elif mode == 'B':
        # Node B: Connects to NodeA and asks questions
        node_b = LearningNode("127.0.0.1", 8002, "NodeB")
        node_b.start()
        
        time.sleep(1) 
        
        # 1. Connect to Node A
        print("\n--- Connecting to NodeA ---")
        node_b.connect_with_node('127.0.0.1', 8001)
        time.sleep(3)
        
        try:
            # 2. Interactive QA Loop (New Feature)
            while True:
                question = input("\nAsk a question (or type 'quit' to exit): ")
                if question.lower() == 'quit':
                    break

                # Send request to the first connected outbound node (NodeA)
                if node_b.nodes_outbound:
                    print(f"--- Sending '{question}' to NodeA...")
                    node_b.send_to_node(node_b.nodes_outbound[0], {
                        "type": "QA_REQUEST",
                        "content": question,
                        "timestamp": time.time()
                    })
                    time.sleep(5) # Give time for processing and response
                else:
                    print("❌ Connection to NodeA lost. Please ensure NodeA is running.")
                    break
        
        except KeyboardInterrupt:
            pass # Exit gracefully on Ctrl+C

        # 3. Clean up
        print("\nNodeB stopped.")
        node_b.stop()
        
    else:
        print("Invalid mode selected. Exiting.")