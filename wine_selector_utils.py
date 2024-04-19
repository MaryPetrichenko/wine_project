'''
Utility class for the wine selector
'''
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class WineSelectorUtils:
    '''
    Utility class for the wine selector
    '''

    def __init__(self):
        # Colours of wine
        red = ['pinot noir', 'st. laurent,', 'zweigelt', 'gamay', 'sangiovese', 'garnacha', 'syrah', 'malbec', 'mourvèdre', 'cabernet sauvignon', 'merlot', 'nebbiolo',
               'tempranillo-merlot', 'red blend', 'frappato', 'nerello mascalese', "nero d'avola", 'bordeaux-style red blend', 'tempranillo blend', 'portuguese red', 'rhône-style red blend',
               'tempranillo', 'cabernet merlot', 'shiraz', 'petite sirah', 'sangiovese grosso', 'corvina, rondinella, molinara', 'carmenère', 'grenache', 'barbera', 'dolcetto',
               'montepulciano', 'pinot nero', 'blaufränkisch', 'aglianico', 'mencía', 'touriga nacional', 'tinta de toro', 'tannat', 'petit verdot', 'monastrell', 'cabernet sauvignon-merlot',
               'pinotage', 'bonarda', 'sagrantino', 'cabernet sauvignon-syrah', 'tinto fino', 'malbec-merlot', 'st. laurent', 'negroamaro', 'cabernet blend', 'lagrein', 'carignan', 'spätburgunder',
               'provence red blend', 'austrian red blend', 'malbec-cabernet sauvignon', 'syrah-grenache', 'tempranillo-cabernet sauvignon', 'alicante bouschet', 'grenache-syrah', 'prugnolo gentile',
               'cabernet sauvignon-carmenère', 'graciano', 'pedro ximénez', 'shiraz-cabernet sauvignon', 'shiraz-viognier', 'syrah-cabernet sauvignon', 'saperavi', 'merlot-cabernet sauvignon',
               'lemberger', 'castelão', 'xinomavro', 'claret', 'teroldego', 'tempranillo-garnacha', 'tannat-cabernet', 'petite verdot', 'carignane', 'touriga nacional-cabernet sauvignon',
               'dornfelder', 'charbono', 'cabernet sauvignon-shiraz', 'trincadeira', 'baga', 'abouriou', 'merlot-cabernet franc', 'cannonau', 'bobal', 'schiava', 'gamay noir', 'corvina', 'cabernet sauvignon-cabernet franc',
               'port', 'primitivo', 'cabernet franc-merlot', 'syrah-mourvèdre', 'monastrell-syrah', 'malbec-syrah', 'touriga nacional blend', 'nero di troia', 'tinto del pais', 'pinot noir-gamay',
               'piedirosso', 'chambourcin', 'cabernet sauvignon-malbec', 'syrah-viognier', 'plavac mali', 'grenache-carignan', 'negrette', 'alfrocheiro', 'baco noir', 'syrah-petite sirah',
               'tannat-cabernet franc', 'malbec-tannat', 'pinot meunier', 'raboso', 'cabernet franc-cabernet sauvignon', 'syrah-cabernet', 'kalecik karasi', 'norton', 'roter veltliner', 'roter']
        white = ['pinot grigio', 'albarino', 'vinho verde', 'muscadet', 'sauvignon blanc', 'unoaked chardonnay', 'chenin blanc', 'chardonnay', 'viognier', 'marssanne', 'portuguese white',
                 'riesling', 'pinot gris', 'gewürztraminer', 'white blend', 'grüner veltliner', 'cabernet franc', 'bordeaux-style white blend', 'rhône-style white blend', 'pinot blanc', 'sauvignon', 'albariño',
                 'torrontés', 'verdejo', 'roussanne', 'turbiana', 'melon', 'vermentino', 'garganega', 'g-s-m', 'verdicchio', 'pinot bianco', 'fiano', 'vernaccia', 'grillo', 'alvarinho', 'sémillon',
                 'grenache blanc', 'friulano', 'greco', 'viura', 'sherry', 'falanghina', 'ribolla gialla', 'arneis', 'austrian white blend', 'arinto', 'fumé blanc', 'sauvignon blanc-semillon', 'marsanne',
                 'verdelho', 'petit manseng', 'moscatel', 'vidal blanc', 'pecorino', 'semillon-sauvignon blanc', 'trebbiano', 'sylvaner', 'alsace white blend', 'gros and petit manseng', 'weissburgunder',
                 'müller-thurgau', 'malvasia', 'assyrtico', 'encruzado', 'traminer', 'silvaner', 'garnacha blanca', 'carricante', 'provence white blend', 'moschofilero', 'chardonnay-viognier', 'zibibbo', 'picolit',
                 'aligoté', 'kerner', 'fernão pires', 'chenin blanc-chardonnay', 'rkatsiteli', 'insolia', 'inzolia', 'auxerrois', 'loureiro', 'scheurebe', 'gros manseng', 'catarratto', 'viognier-chardonnay',
                 'rotgipfler', 'colombard', 'welschriesling', 'macabeo', 'jacquère', 'passerina', 'verdejo-viura', 'muskat ottonel', 'marsanne-roussanne', 'savagnin', 'tocai friulano', 'orange muscat', 'xarel-lo',
                 'colombard-sauvignon blanc', 'palomino', 'picpoul', 'coda di volpe', 'antão vaz', 'assyrtiko', 'hondarrabi zuri', 'seyval blanc', 'moscato giallo', 'malvasia bianca', 'zierfandler',
                 'sauvignon gris', 'pansa blanca', 'white riesling', 'chenin blanc-viognier', 'albana', 'moscato', 'muscat', 'furmint', 'tokaji']
        rose = ['rose', 'rosé', 'rosato', 'rosado', 'portuguese rosé']
        sparkling = ['lambrusco', 'glera', 'champagne blend', 'sparkling blend', 'prosecco', 'portuguese sparkling',
                     'sparkling', 'frizzante', 'lambrusco di sorbara', 'lambrusco grasparossa', 'sparkling']
        self.__wine_types = {
            'red': red,
            'white': white,
            'rose': rose,
            'sparkling': sparkling
        }

        type_custom_stopwords = ['wine', 'red', 'flavors', 'blend', 'rosé', 'acidity', 'de',
                                 'sparkling', 'champagne', 'white', 'blanc', 'aromas', 'valley',
                                 'tannins', 'palate', 'nv', 'finish', 'drink', 'california',
                                 'provence', 'château', 'côtes', 'cabernet', 'pinot', 'sauvingon',
                                 'vineyard', 'notes', 'brut', 'cava', 'cuvée', 'nose',
                                 'noir', 'fruit', 'fruity', 'fruits', 'dry', 'texture',
                                 'color', 'touch', 'well', 'balanced', 'character',
                                 'sauvignon', 'offers', 'fine', 'full', 'fine',
                                 'made', 'good', 'years', 'la', 'shows', 'sample', 'rich']
        self.__stop_words = ENGLISH_STOP_WORDS.union(type_custom_stopwords)

    def __check_if_any_of_list(self, row, columns, lst):
        for col in columns:
            try:
                if any(item.lower() in row[col].lower() for item in lst):
                    return True
            except Exception:
                continue
        return False

    def __get_wine_type(self, row, column):
        for key, value in self.__wine_types.items():
            if self.__check_if_any_of_list(row, [column], value):
                return key
        return None

    def assign_wine_types(self, df):
        '''
        Assign wine types to the dataframe
        '''
        for index, row in df.iterrows():
            # Check variety field
            type_by_variety = self.__get_wine_type(row, 'variety')

            # Correct the type if found in title or designation
            type_by_title = self.__get_wine_type(row, 'title')
            type_by_design = self.__get_wine_type(row, 'designation')

            # Assign the final type
            if type_by_design:
                df.at[index, 'type'] = type_by_design
            elif type_by_title:
                df.at[index, 'type'] = type_by_title
            elif type_by_variety:
                df.at[index, 'type'] = type_by_variety
            else:
                df.at[index, 'type'] = 'unknown'

        df.loc[(df['variety'].str.contains('zinfandel', case=False, na=False)) &
               (df['designation'].str.contains('rosé', case=False, na=False, regex=True)), 'type'] = 'rose'
        df.loc[(df['variety'].str.contains('zinfandel', case=False, na=False)) &
               (df['designation'].str.contains('white ', case=False, na=False, regex=True)), 'type'] = 'white'
        df.loc[(df['variety'].str.contains('zinfandel', case=False, na=False)) &
               (df['type'].str.contains('unknown', case=False, na=False, regex=True)), 'type'] = 'red'

        return df

    def assign_wine_types_fast(self, df):
        '''
        Assign wine types to the dataframe
        '''
        # Check wine types based on variety, title, and designation
        type_by_variety = df.apply(
            lambda row: self.__get_wine_type(row, 'variety'), axis=1)
        type_by_title = df.apply(
            lambda row: self.__get_wine_type(row, 'title'), axis=1)
        type_by_design = df.apply(
            lambda row: self.__get_wine_type(row, 'designation'), axis=1)

        # Assign the final type based on priority
        df['type'] = type_by_design.combine_first(
            type_by_title).combine_first(type_by_variety).fillna('unknown')

        # Further refinement based on specific conditions
        zinfandel_mask = df['variety'].str.contains(
            'zinfandel', case=False, na=False)
        rose_mask = df['designation'].str.contains(
            'rosé', case=False, na=False, regex=True)
        white_mask = df['designation'].str.contains(
            'white', case=False, na=False, regex=True)
        unknown_mask = df['type'].str.contains(
            'unknown', case=False, na=False, regex=True)

        df.loc[zinfandel_mask & rose_mask, 'type'] = 'rose'
        df.loc[zinfandel_mask & white_mask, 'type'] = 'white'
        df.loc[zinfandel_mask & unknown_mask, 'type'] = 'red'

        return df

    def get_wine_types(self):
        '''
        Get the wine types
        '''
        return self.__wine_types

    def get_stop_words(self):
        '''
        Get the stop words
        '''
        return self.__stop_words
