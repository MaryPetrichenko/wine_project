'''
WineSelector is the class that provides functionality to select the wine by its description
'''
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from wine_selector_utils import WineSelectorUtils


class WineSelector:
    '''
    WineSelector is the class that provides functionality to select the wine by its description
    '''

    def __init__(self, chunk_size: int = 10000, n_similar: int = 5):
        self.__chunk_size = chunk_size
        self.__n_similar = n_similar
        self.__data = None
        self.__utils = WineSelectorUtils()

        # Create vectorizer with additional stop words
        self.__stop_words = self.__utils.get_stop_words()
        self.__tfidf = TfidfVectorizer(stop_words=list(self.__stop_words))

    def __country_fill(self):
        """
        The function takes first words in 'title' column before digits
        representing the year for the rows with null values in 'country' column
        and searches through the dataframe in other rows to get the country and
        fills null values if any
        """

        pattern = r"^(.*?)(?=\b\d{4}\b)"  # find the first words before digits
        # list of titles with null values as countries
        titles = list(self.__data[self.__data.country.isnull()].title)
        title_country_dict = (
            {}
        )  # a dictionary to keep track of matching titles and countries
        for title in titles:
            match = re.search(pattern, title)
            countries = []  # a list of found countries
            try:
                matched_string = match.group()
                mask = self.__data["title"].str.contains(
                    matched_string, case=False, na=False)
                # a list of unique country values
                countries = list(
                    set(self.__data[mask][self.__data[mask]["country"].notnull()].country))
                if len(countries) == 1:
                    title_country_dict[matched_string] = countries[0]
            except AttributeError:
                continue

        for key, value in title_country_dict.items():
            mask = self.__data["title"].str.contains(key, case=False, na=False)
            self.__data.loc[mask, "country"] = self.__data.loc[mask, "country"].fillna(
                value
            )  # replaces all nulls with the found country names

    def __split_data_to_chunks(self, data: pd.DataFrame) -> list[pd.DataFrame]:
        """
        The method splits the data into chunks of the size defined by the chunk_size attribute
        """

        chunks = []

        pointer = 0
        chunk_size = self.__chunk_size

        while pointer < len(data):
            if pointer + chunk_size > len(data):
                chunk_size = len(data) - pointer

            chunk = data[pointer: pointer + chunk_size]
            chunks.append(chunk)
            pointer += chunk_size

        return chunks

    def get_data(self):
        '''
        Get the data
        '''
        return self.__data

    def preprocess_data(self, fileName: str):
        '''
        Initialize the data
        '''
        if self.__data is None:
            print("Loading data...")
            self.__data = pd.read_csv(fileName, index_col=0)

            print("Apply correction to the DB...")
            self.__country_fill()

            print("Dropping rows with null values in 'country' and 'variety' columns...")
            self.__data.dropna(subset=["country", 'variety'], inplace=True)

            print("Creating a compound description...")
            cols = ['description', 'variety', 'province', 'title']
            self.__data['compound_description'] = self.__data.apply(
                lambda row: ' '.join(str(row[col]) for col in cols), axis=1)

            print("Assigning wine types...")
            self.__utils.assign_wine_types(self.__data)

        print("Data is loaded and preprocessed!")

    def save_preprocessed_data(self, fileName: str):
        '''
        Save the preprocessed data to the file
        '''
        if self.__data is not None:
            self.__data.to_csv(fileName, index=True,
                               index_label='index', header=True)

    def load_preprocessed_data(self, fileName: str):
        '''
        Load the preprocessed data from the file
        '''
        self.__data = pd.read_csv(fileName, index_col=0)

    def __choose_wine(self, data: pd.DataFrame, wine_request: str):
        """
        The function takes the dataset, the vectorizer, and a wine request as input
        and returns the most similar wines to the request.

        Result is a dictionary with the index of the wine in the dataset as a key
        and the similarity score as a value.
        """

        selection = {}

        portions = self.__split_data_to_chunks(data)
        for cnt, portion in enumerate(portions):
            p = list(portion['compound_description'])
            # Add the request to the portion - now its index is the last one!
            p.append(wine_request)

            tfidf_matrix = self.__tfidf.fit_transform(p)
            cosine_sim = linear_kernel(tfidf_matrix)

            # Choose the match for the last index of sim matrix
            sim_scores = list(cosine_sim[-1])
            sim_scores = sorted(enumerate(sim_scores),
                                key=lambda x: x[1], reverse=True)

            # Get the top 5 most similar wines
            sim_scores = sim_scores[1: self.__n_similar + 1]

            for idx, score in sim_scores:
                # Get index of the wine in initial dataset
                wine_idx = portion.index[idx]
                selection[wine_idx] = score

            print(f'\rPortion {cnt+1} of {len(portions)} done!', end='')

        print('\nDone')
        return selection

    def select_wine(self, request: str,
                    type_filter: list[str] | None = None,
                    price_filter: list[float] | None = None):
        '''
        Select the wine by the request
        '''

        # Filter the data by type and price
        data = self.__data
        if type_filter:
            data = data[data['type'].isin(type_filter)]
        if price_filter:
            data = data[(data['price'] >= price_filter[0]) &
                        (data['price'] <= price_filter[1])]

        wine_choice = self.__choose_wine(data, request)

        # Create a dataframe with the selected wines and their scores
        selected_wines = pd.DataFrame(
            list(wine_choice.items()), columns=['index', 'score'])
        selected_wines['index'] = selected_wines['index'].astype(int)
        selected_wines.set_index('index', inplace=True)
        selected_wines = selected_wines.merge(
            data[['title', 'description', 'type', 'price', 'points', 'variety']], left_index=True, right_index=True)
        selected_wines.sort_values('score', ascending=False, inplace=True)

        return selected_wines
