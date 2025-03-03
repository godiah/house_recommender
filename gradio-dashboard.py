import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import gradio as gr
load_dotenv()


houses = pd.read_csv("house_data_cleaned_v2.csv")

# Persistence directory for the vector database
persist_directory = "db_house_persistence"

# Load and split documents for vector search
raw_documents = TextLoader("house_tag.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the vector database if it exists; otherwise, create and persist it.
if os.path.exists(persist_directory):
    db_house = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    db_house = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_directory)
    db_house.persist()

# Function to retrieve recommendations with filters
def retrieve_house_recommendations(
        query: str,
        category: str = None,
        neighbourhood_quality: str = "All",
        house_condition: str = "All",
        crime_rate: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16,
)->pd.DataFrame:
    recs = db_house.similarity_search_with_score(query, k=initial_top_k)
    houses_list = [int(doc.page_content.strip('"').split()[0]) for doc, _ in recs]
    house_recs = houses[houses["tag"].isin(houses_list)]

    # Apply filters
    if category != "All":
        house_recs = house_recs[house_recs["house_type"] == category]
    if neighbourhood_quality != "All":
        house_recs = house_recs[house_recs["neighbourhood_quality"] == neighbourhood_quality]
    if house_condition != "All":
        house_recs = house_recs[house_recs["house_condition"] == house_condition]
    if crime_rate != "All":
        house_recs = house_recs[house_recs["crime_rate"] == crime_rate]

    return house_recs.head(final_top_k)

# Function to specify what we want to display on dashboard
def recommend_houses(
        query: str,
        category: str = None,
        neighbourhood_quality: str = "All",
        house_condition: str = "All",
        crime_rate: str = "All"
):
    recommendations = retrieve_house_recommendations(query, category, neighbourhood_quality, house_condition, crime_rate)
    results = []

    # Default thumbnail placeholder image
    placeholder = "https://placehold.co/600x400/gray/white?text=House+Image"

    for _, row in recommendations.iterrows():
        keywords = row['house_keywords'].split()[:10]
        truncated_keywords = " ".join(keywords)

        caption = (
            f"Price: ${row['price']:,.2f}\n"
            f"Bedrooms: {row['bedrooms']}  Bathrooms: {row['bathrooms']}\n"
            f"Garden Size: {row['garden_size']}\n"
            f"Location: {row.get('co-ordinates', 'N/A')}\n"
            f"Keywords: {truncated_keywords}..."
        )
        results.append((placeholder, caption))
    if not results:
        return [(placeholder, "No matching houses found. Try adjusting your filters.")]

    return results

categories = ["All"] + sorted(houses["house_type"].unique())
neighbourhood_options = ["All"] + sorted(houses["neighbourhood_quality"].unique())
condition_options = ["All"] + sorted(houses["house_condition"].unique())
crime_options = ["All"] + sorted(houses["crime_rate"].unique())

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# House Recommender Dashboard")

    # Search query
    with gr.Row():
        with gr.Column(scale=8):
            user_query = gr.Textbox(
                label="Enter house features:",
                placeholder="e.g., open floor plan, high ceilings, built-in wardrobes, air conditioning",
                lines=2
            )
        with gr.Column(scale=1):
            submit_button = gr.Button("Find Recommendations", variant="primary", size="lg")

    # Filters
    with gr.Row():
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="House Type:",
            value="All",
            container=True
        )
        neighbourhood_dropdown = gr.Dropdown(
            choices=neighbourhood_options,
            label="Neighbourhood Quality:",
            value="All",
            container=True
        )
        condition_dropdown = gr.Dropdown(
            choices=condition_options,
            label="House Condition:",
            value="All",
            container=True
        )
        crime_dropdown = gr.Dropdown(
            choices=crime_options,
            label="Crime Rate:",
            value="All",
            container=True
        )

    gr.Markdown("### Recommended Houses")
    output_gallery = gr.Gallery(
        label="House Recommendations",
        columns=4,
        rows=4,
        object_fit="contain",
        height="600px",
        show_label=False,
        elem_id="house_gallery"
    )

    submit_button.click(
        fn=recommend_houses,
        inputs=[
            user_query,
            category_dropdown,
            neighbourhood_dropdown,
            condition_dropdown,
            crime_dropdown
        ],
        outputs=output_gallery
    )

if __name__ == "__main__":
    dashboard.launch(share=True)
