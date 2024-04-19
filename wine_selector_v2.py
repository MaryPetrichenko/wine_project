import pandas as pd
import re
import sys
import multiprocessing
import time
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel


def measure_execution_time(func):
    """
    The decorator function to measure the execution time of the function
    """
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {
              end_time - start_time} seconds")
        return result
    return wrapper


def country_fill(df):
    """
    The function takes first words in 'title' column before digits
    representing the year for the rows with null values in 'country' column
    and searches through the dataframe in other rows to get the country and
    fills null values if any
    """

    pattern = r"^(.*?)(?=\b\d{4}\b)"  # find the first words before digits
    # list of titles with null values as countries
    titles = list(df[df.country.isnull()].title)
    title_country_dict = (
        {}
    )  # a dictionary to keep track of matching titles and countries
    for title in titles:
        match = re.search(pattern, title)
        countries = []  # a list of found countries
        try:
            matched_string = match.group()
            mask = df["title"].str.contains(
                matched_string, case=False, na=False)
            # a list of unique country values
            countries = list(
                set(df[mask][df[mask]["country"].notnull()].country))
            if len(countries) == 1:
                title_country_dict[matched_string] = countries[0]
        except AttributeError:
            continue

    for key, value in title_country_dict.items():
        mask = df["title"].str.contains(key, case=False, na=False)
        df.loc[mask, "country"] = df.loc[mask, "country"].fillna(
            value
        )  # replaces all nulls with the found country names


def split_df(data: pd.DataFrame, chunk_size: int = 10000):
    """
    The function takes a dataset and a chunk size as input and returns a list of dataframes
    """

    chunks = []
    pointer = 0
    while pointer < len(data):
        if pointer + chunk_size > len(data):
            chunk_size = len(data) - pointer

        chunk = data[pointer: pointer + chunk_size]
        chunks.append(chunk)
        pointer += chunk_size

    return chunks


def choose_wine_part(chunk: pd.DataFrame, tfidf: TfidfVectorizer,
                     wine_request: str, n_similar: int = 5,
                     kernel=linear_kernel):

    print(f"Started processing chunk at {chunk.index[0]}...")
    p = list(chunk['description'])
    # Add the request to the portion - now its index is the last one!
    p.append(wine_request)

    tfidf_matrix = tfidf.fit_transform(p)
    cosine_sim = kernel(tfidf_matrix)

    # Choose the match for the last index of sim matrix
    sim_scores = list(cosine_sim[-1])
    sim_scores = sorted(enumerate(sim_scores),
                        key=lambda x: x[1], reverse=True)

    # Get the top 5 most similar wines
    sim_scores = sim_scores[1: n_similar + 1]

    print(f"Ended processing chunk at {chunk.index[0]}...")

    # Set the scores in the selection dictionary
    return {chunk.index[idx]: score for idx, score in sim_scores}


@measure_execution_time
def choose_wine(data: pd.DataFrame, tfidf: TfidfVectorizer, wine_request: str,
                chunk_size: int = 10000, n_similar: int = 5,
                kernel=linear_kernel):
    """
    The function takes the dataset, the vectorizer, and a wine request as input
    and returns the most similar wines to the request.

    Result is a dictionary with the index of the wine in the dataset as a key
    and the similarity score as a value.
    """

    selection = {}

    # Divide dataset by chunks of 10000 rows
    portions = split_df(data, chunk_size)

    for cnt, portion in enumerate(portions):
        selection.update(choose_wine_part(portion, tfidf,
                                          wine_request, n_similar, kernel))

    return selection


@measure_execution_time
def choose_wine_par(data: pd.DataFrame, tfidf: TfidfVectorizer, wine_request: str,
                    chunk_size: int = 10000, n_similar: int = 5,
                    kernel=linear_kernel):
    """
    The function takes the dataset, the vectorizer, and a wine request as input
    and returns the most similar wines to the request.

    Result is a dictionary with the index of the wine in the dataset as a key
    and the similarity score as a value.
    """

    selection = {}

    # Divide dataset by chunks of 10000 rows
    portions = split_df(data, chunk_size)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    res = pool.map(choose_wine_part, portions, tfidf,
                   wine_request, n_similar, kernel)

    pool.join()
    pool.close()

    return selection


def main():
    # Take the description from the argument
    request = sys.argv[1] if len(sys.argv) > 1 else None

    # Load the data
    print("Loading data...")
    data = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)

    print("Apply corrections to the dataset...")
    # Apply changes to country field
    country_fill(data)

    # Drop rows with null values in 'country' and 'variety' columns
    data.dropna(subset=["country", 'variety'], inplace=True)

    # Create vectorizer with additional stop words
    stop_words = ENGLISH_STOP_WORDS.union(['wine', 'flavors', 'aromas',
                                           'offers', 'notes', 'nose', 'drink',
                                           'shows', 'well', 'very'])
    tfidf = TfidfVectorizer(stop_words=list(stop_words))

    wine_request = "A very spicy wine, layering honey and lychees. Opulent, ripe, wearing its richness on its sleeve. Probably for medium-term aging, but it could well surprise."
    wine_request = "Lychees honey and mild wine. Opulent, full-bodied."
    wine_choice = choose_wine(
        data, tfidf, request if request else wine_request
    )

    # Create a dataframe with the selected wines and their scores
    selected_wines = pd.DataFrame(
        list(wine_choice.items()), columns=['index', 'score'])
    selected_wines['index'] = selected_wines['index'].astype(int)
    selected_wines = selected_wines.merge(
        data[['title', 'description']], left_on='index', right_index=True)
    selected_wines.sort_values('score', ascending=False, inplace=True)

    print(selected_wines[['title', 'description', 'score']].head())
    selected_wines.to_csv('results\\selected_wines.csv', index=False)


if __name__ == "__main__":
    main()
