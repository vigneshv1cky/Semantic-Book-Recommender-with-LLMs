# --- Code from `llm-semantic-book-recommender.ipynb` ---
from google.colab import drive

drive.mount("/content/drive")

import kagglehub
from matplotlib.pyplot import plot_date

# Download latest version
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")

print("Path to dataset files:", path)

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

####################################################################################################
# Sentiment Analysis
####################################################################################################

books = pd.read_csv(f"{path}/books.csv")
books

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6))
ax = plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)

plt.xlabel("Columns")
plt.ylabel("Missing values")

plt.show()

import numpy as np

books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2024 - books["published_year"]

columns_of_interest = [
    "num_pages",
    "age_of_book",
    "missing_description",
    "average_rating",
]

correlation_matrix = books[columns_of_interest].corr(method="spearman")

sns.set_theme(style="white")
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"label": "Spearman correlation"},
)
heatmap.set_title("Correlation heatmap")
plt.show()

# books[
#     ~(books["description"].isna())
#     & ~(books["num_pages"].isna())
#     & ~(books["average_rating"].isna())
#     & ~(books["published_year"].isna())
# ].head()

book_missing = books[
    books["description"].notna()
    & books["num_pages"].notna()
    & books["average_rating"].notna()
    & books["published_year"].notna()
].head

book_missing = books[
    books["description"].notna()
    & books["num_pages"].notna()
    & books["average_rating"].notna()
    & books["published_year"].notna()
].shape

book_missing = book_missing = books[
    books["description"].notna()
    & books["num_pages"].notna()
    & books["average_rating"].notna()
    & books["published_year"].notna()
]

book_missing["categories"].value_counts().reset_index().sort_values(
    "count", ascending=False
)

book_missing["description"].str.split().str.len()

book_missing["words_in_description"] = book_missing["description"].str.split().str.len()

book_missing.loc[book_missing["words_in_description"].between(5, 14), "description"]

book_missing.loc[book_missing["words_in_description"].between(15, 20), "description"]

book_missing[book_missing["words_in_description"] >= 25]

book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25]

np.where(
    book_missing_25_words["subtitle"].isna(),
    book_missing_25_words["title"],
    book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1),
)

book_missing_25_words["title_and_subtitle"] = np.where(
    book_missing_25_words["subtitle"].isna(),
    book_missing_25_words["title"],
    book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1),
)

 book_missing_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)

book_missing_25_words["tagged_description"] = (
    book_missing_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
)

books_cleaned = book_missing_25_words.copy(deep=True)
books_cleaned

####################################################################################################
# Vector Search
####################################################################################################

# !pip install langchain_community langchain_openai langchain_chroma

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import pandas as pd

books = books_cleaned.copy(deep=True)

books["tagged_description"].to_csv(
    "/content/drive/Othercomputers/Vignesh MacBook Air/Semantic-Book-Recommender-with-LLMs/tagged_description.txt",
    sep="\n",
    index=False,
    header=False,
)

raw_documents = TextLoader(
    "/content/drive/Othercomputers/Vignesh MacBook Air/Semantic-Book-Recommender-with-LLMs/tagged_description.txt"
).load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

documents[0]

from google.colab import userdata
import os

# Set the OpenAI API key in the environment variable
os.environ["OPENAI_API_KEY"] = userdata.get("open-ai-api-key")

db_books = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
db_books

query = "A book to teach children about nature"
docs = db_books.similarity_search(query, k=10)
docs

books[books["isbn13"] == int(docs[0].page_content.split()[0].strip())]

def retrieve_semantic_recommendations(
    query: str,
    top_k: int = 10,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=50)

    books_list = []

    for i in range(0, len(recs)):
        books_list += [int(recs[i].page_content.strip('"').split()[0])]

    return books[books["isbn13"].isin(books_list)]


retrieve_semantic_recommendations("A book to teach children about nature")



books = books_cleaned.copy(deep=True)
books

books["categories"].value_counts().reset_index()

books["categories"].value_counts().reset_index().query("count > 50")

category_mapping = {
    "Fiction": "Fiction",
    "Juvenile Fiction": "Children's Fiction",
    "Biography & Autobiography": "Nonfiction",
    "History": "Nonfiction",
    "Literary Criticism": "Nonfiction",
    "Philosophy": "Nonfiction",
    "Religion": "Nonfiction",
    "Comics & Graphic Novels": "Fiction",
    "Drama": "Fiction",
    "Juvenile Nonfiction": "Children's Nonfiction",
    "Science": "Nonfiction",
    "Poetry": "Fiction",
}

books["simple_categories"] = books["categories"].map(category_mapping)

books.head()

books.dropna(subset=["simple_categories"]).head()

from transformers import pipeline

fiction_categories = ["Fiction", "Nonfiction"]

pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

sequence = books.loc[
    books["simple_categories"] == "Fiction", "description"
].reset_index(drop=True)[0]

pipe(sequence, fiction_categories)

import numpy as np

max_index = np.argmax(pipe(sequence, fiction_categories)["scores"])
max_label = pipe(sequence, fiction_categories)["labels"][max_index]
max_label

def generate_predictions(sequence, categories):
    predictions = pipe(sequence, categories)
    max_index = np.argmax(predictions["scores"])
    max_label = predictions["labels"][max_index]
    return max_label

from tqdm import tqdm

actual_cats = []
predicted_cats = []

for i in tqdm(range(0, 300)):
    sequence = books.loc[
        books["simple_categories"] == "Fiction", "description"
    ].reset_index(drop=True)[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ["Fiction"]

for i in tqdm(range(0, 300)):
    sequence = books.loc[
        books["simple_categories"] == "Nonfiction", "description"
    ].reset_index(drop=True)[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ["Nonfiction"]

predictions_df = pd.DataFrame(
    {"actual_categories": actual_cats, "predicted_categories": predicted_cats}
)

predictions_df["correct_prediction"] = np.where(
    predictions_df["actual_categories"] == predictions_df["predicted_categories"], 1, 0
)

predictions_df["correct_prediction"].sum() / len(predictions_df)

isbns = []
predicted_cats = []

missing_cats = books.loc[
    books["simple_categories"].isna(), ["isbn13", "description"]
].reset_index(drop=True)

isbns = []
predicted_cats = []

missing_cats = books.loc[
    books["simple_categories"].isna(), ["isbn13", "description"]
].reset_index(drop=True)

for i in tqdm(range(0, len(missing_cats))):
    sequence = missing_cats["description"][i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    isbns += [missing_cats["isbn13"][i]]

missing_predicted_df = pd.DataFrame(
    {"isbn13": isbns, "predicted_categories": predicted_cats}
)

books = pd.merge(books, missing_predicted_df, on="isbn13", how="left")
books["simple_categories"] = np.where(
    books["simple_categories"].isna(),
    books["predicted_categories"],
    books["simple_categories"],
)
books = books.drop(columns=["predicted_categories"])

books[
    books["categories"]
    .str.lower()
    .isin(
        [
            "romance",
            "science fiction",
            "scifi",
            "fantasy",
            "horror",
            "mystery",
            "thriller",
            "comedy",
            "crime",
            "historical",
        ]
    )
]

####################################################################################################
# Sentiment Analysis
####################################################################################################

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0,
)
classifier("I love this!")

books["description"][0]

classifier(books["description"][0])

classifier(books["description"][0].split("."))

sentences = books["description"][0].split(".")
sentences

predictions = classifier(sentences)
predictions

sorted(predictions[0], key=lambda x: x["label"])

import numpy as np

emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}


def calculate_max_emotion_scores(predictions):
    per_emotion_scores = {label: [] for label in emotion_labels}
    for prediction in predictions:
        sorted_predictions = sorted(prediction, key=lambda x: x["label"])
        for index, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}

for i in range(10):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])

emotion_scores

from tqdm import tqdm

emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

for i in tqdm(range(len(books))):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])

emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn
emotions_df

books = pd.merge(books, emotions_df, on="isbn13")
books

####################################################################################################
# Gradio
####################################################################################################

%cd "/content/drive/Othercomputers/Vignesh MacBook Air/Semantic-Book-Recommender-with-LLMs"

%pip install gradio

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()


books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(
            final_top_k
        )
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
        )
        category_dropdown = gr.Dropdown(
            choices=categories, label="Select a category:", value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, label="Select an emotional tone:", value="All"
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

dashboard.launch(inline=True)


