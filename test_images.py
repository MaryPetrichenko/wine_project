from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from wordcloud import WordCloud, STOPWORDS


def main():
    # get data directory (using getcwd() is needed to support running example in generated IPython notebook)
    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

    # Read the whole text.
    text = '''By default, an empty Opc Ua server configuration is created in the ./build/conf folder, no runner configurations are added. 
    Possible simulation configurations are defined in ./scripts/daqConfig/simvariants. To generate system configurations 
    from their compact json representations, run the following Python setup script from the root project folder:'''

    # read the mask image
    # taken from
    # http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
    alice_mask = np.array(Image.open(path.join(d, "wine.webp")))
    alice_mask = np.apply_along_axis(
        lambda x: 255 if x[0] == 0 else x[0], axis=2, arr=alice_mask)
    print(alice_mask)

    stopwords = set(STOPWORDS)
    stopwords.add("said")

    wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask,
                   stopwords=stopwords, contour_width=3, contour_color='steelblue')

    # generate word cloud
    wc.generate(text)

    # store to file
    wc.to_file(path.join(d, "alice_result.png"))

    # show
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.figure()
    plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
