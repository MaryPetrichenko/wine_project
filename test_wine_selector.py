import wine_selector as ws
import os

SRC_FILENAME = 'winemag-data-130k-v2.csv'
DB_FILENAME = 'winemag-data-130k-v2-preprocessed.csv'


def main():
    '''
    Main function to test the WineSelector class
    '''
    # Create an instance of the WineSelector class
    selector = ws.WineSelector(chunk_size=10000, n_similar=5)

    if os.path.exists(DB_FILENAME):
        print("Loading preprocessed data...")
        selector.load_preprocessed_data(DB_FILENAME)
    else:
        print("Preprocessing data...")
        selector.preprocess_data(SRC_FILENAME)
        print("Saving preprocessed data...")
        selector.save_preprocessed_data(DB_FILENAME)

    df = selector.get_data()
    print(f"Data shape base:\t{df.shape}")
    df.drop_duplicates(
        subset=['title', 'description', 'price', 'points'], inplace=True)
    print(f"Data shape dropped:\t{df.shape}")
    print(df[df.index == 11]['description'].values[0])

    return

    # Select the wine
    request = 'Nice, mild pear taste, hint of vanialla and apple'
    selected_wines = selector.select_wine(
        request, type_filter=['red', 'rose'], price_filter=[50, 100])

    print(selected_wines[['title', 'description', 'score']])

    # Save the results
    # selected_wines.to_csv('results\\selected_wines.csv', index=False)


if __name__ == '__main__':
    main()
