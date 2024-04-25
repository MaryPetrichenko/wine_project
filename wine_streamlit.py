import streamlit as st
import wine_selector as ws
import os

SRC_FILENAME = 'winemag-data-130k-v2.csv'
DB_FILENAME = 'winemag-data-130k-v2-preprocessed.csv'


@st.cache_data()
def load_data():
    '''
    Load the data from the cache. If the data is not in the cache, load it 
    rom the file and preprocess it
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

    selector.get_data().drop_duplicates(
        subset=['title', 'description', 'price', 'points'], inplace=True)

    return selector


def get_type_filter():
    '''
    Get the type filter
    '''
    type_filter = []
    if st.session_state.white_state:
        type_filter.append('white')
    if st.session_state.red_state:
        type_filter.append('red')
    if st.session_state.rose_state:
        type_filter.append('rose')
    if st.session_state.sparkling_state:
        type_filter.append('sparkling')
    return type_filter


def btn_click():
    '''
    Button click event
    '''
    with st.spinner("Searching for the best wines..."):
        type_filter = get_type_filter()
        price_filter = list(st.session_state.price)
        result = st.session_state.data.select_wine(
            st.session_state.prompt, type_filter=type_filter,
            price_filter=price_filter)
        st.session_state.result = result if not result.empty else None


# Page configuration
st.set_page_config(page_title="Wine Advisor",
                   page_icon="🍷", layout="centered")

# Connect data to the app
st.session_state.data = load_data()

st.title("Welcome to the Wine Advisor", anchor="center")
st.image("wine_image.jpg")
st.subheader(
    "Uncork the Perfect Wine Experience with Wine Advisor: Your Palate's Personal Sommelier!")

# Set up wine type filters
st.checkbox("White wines", key="white_state")
st.checkbox("Red wines", key="red_state")
st.checkbox("Rosé wines", key="rose_state")
st.checkbox(
    "Let's go bubbles (Sparkling, Frizzante, Prosecco, Champagne...)",
    key="sparkling_state")

# Set up price filter
left, right = st.columns(2)
left.number_input("Minimal price of the wine", value=0,
                  min_value=0, key="min_price")
right.number_input("Maximal price of the wine", value=100,
                   min_value=0, key="max_price")
st.slider("Price of the wine",
          min_value=st.session_state.min_price,
          max_value=st.session_state.max_price,
          value=(st.session_state.min_price, st.session_state.max_price),
          key="price")

# Set the input prompt and search button
st.text_input(
    "Detail the Flavor Profile and Bouquet of the Wine You're Eager to Sample",
    max_chars=200, key="prompt", placeholder="E.g. 'Mild pear taste, hint of vanilla and apple'")
st.button("Search", on_click=btn_click)

# Display the results if any
if 'result' in st.session_state:
    st.title("Top 5 Recommendations")
    st.divider()
    if st.session_state.result is not None:
        for idx, row in st.session_state.result.head(5).iterrows():
            st.header(f"{row['title']}")
            st.subheader(f"{row['variety']} ({row['type'].capitalize()})")
            st.subheader(f"Price: {row['price']}$ ({row['points']} points)")
            st.caption(f"Match: {row['score']*100:.1f}%")
            st.write(f"{row['description']}")
            st.divider()
    else:
        st.header("No wines found")

st.markdown(
    "This app utilizes data from [Wine Enthusiasts](https://www.wineenthusiast.com/). I acknowledge and appreciate their contribution to enriching my app's content.")
