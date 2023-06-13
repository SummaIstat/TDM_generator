import logging
import os
import sys
import json
import string
from urllib.request import urlopen
from datetime import datetime
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# italian dictionary
# https://github.com/sigmasaur/AnagramSolver/blob/main/dictionary.txt

# logging configuration
logger = logging.getLogger("TDM_generator")
logger.setLevel(logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.DEBUG)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
now = datetime.now()
dateTime = now.strftime("%Y-%m-%d_%H_%M_%S")
LOG_FILE_NAME = "TDM_generator_" + dateTime + ".log"
fileHandler = logging.FileHandler(LOG_FILE_NAME)
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

# Solr configuration
SOLR_IP_ADDRESS = ""
SOLR_PORT_NUMBER = ""
SOLR_CORE_NAME = ""
SOLR_MAX_DOCS = ""

# parameters
MIN_NUM_OF_OCCURRENCES = ""
MAX_NUM_DOCS_PER_FIRM = ""
MIN_WORD_LENGTH = ""
MAX_WORD_LENGTH = ""
ITA_DICTIONARY_FILE = ""
ENG_DICTIONARY_FILE = ""
LOG_LEVEL = ""

italian_words = []
english_words = []


def main(argv):
    logger.info("********************************************")
    logger.info("**********   TDM_generator   ***************")
    logger.info("********************************************\n\n")

    now = datetime.now()
    dateTime = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Starting datetime: " + dateTime)

    global italian_words
    global english_words
    load_external_configuration()
    italian_words = load_dictionary(ITA_DICTIONARY_FILE)
    english_words = load_dictionary(ENG_DICTIONARY_FILE)

    logger.info("Acquiring the complete list of words indexed in Solr")
    set_1 = get_all_solr_words_by_field("titolo")
    # set_2 = get_all_solr_words_by_field("metatagDescription")
    # set_3 = get_all_solr_words_by_field("metatagKeywords")
    # set_4 = get_all_solr_words_by_field("links")
    set_5 = get_all_solr_words_by_field("corpoPagina")
    solr_sorted_words_list = list(set().union(set_1).union(set_5))
    solr_sorted_words_list.sort()
    solr_sorted_words_list = get_lista_ripulita(solr_sorted_words_list)

    logger.info("Producing the first line of the TDM with the stemmed version of all the words contained in Solr")
    solr_stemmed_words = get_stemmed_words(solr_sorted_words_list)
    solr_stemmed_words = list(solr_stemmed_words)
    solr_stemmed_words.sort()  # contenuto della prima riga contenente la versione stemmata di tutte le parole

    logger.info("Getting the complete list of company ids in solr")
    set_ids = get_firm_ids()
    solr_firm_id_sorted_list = list(set_ids)
    solr_firm_id_sorted_list.sort()

    logger.info("Getting for each firm the complete list of words indexed in solr")

    now = datetime.now()
    dateTime = now.strftime("%Y-%m-%d_%H_%M_%S")
    outputFileName = "TDM_" + dateTime + ".csv"

    with open(outputFileName, 'a+', encoding='utf-8') as f:
        first_row = "firmId\t" + "\t".join(solr_stemmed_words)
        f.writelines(first_row + "\n")
        list_len = len(solr_firm_id_sorted_list)
        for i, firm_id in enumerate(solr_firm_id_sorted_list):
            logger.info("Processing firm " + str(i+1) + " / " + str(list_len) + " having firmId = " + firm_id)
            text = get_text_from_docs(firm_id)
            tokenized_text = word_tokenize(text)
            tokenized_text = get_lista_ripulita(tokenized_text)
            stemmed_words_counts = get_stemmed_words_count_dict(tokenized_text)
            f.writelines(firm_id)
            for word in solr_stemmed_words:
                if stemmed_words_counts.get(word, 0) == 0:
                    f.writelines("\t" + "0")
                else:
                    f.writelines("\t" + str(stemmed_words_counts[word]))
            f.writelines("\n")
            f.flush()

    now = datetime.now()
    dateTime = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Ending datetime: " + dateTime)


def load_external_configuration():
    global SOLR_IP_ADDRESS
    global SOLR_PORT_NUMBER
    global SOLR_CORE_NAME
    global SOLR_MAX_DOCS
    global MIN_NUM_OF_OCCURRENCES
    global MAX_NUM_DOCS_PER_FIRM
    global MIN_WORD_LENGTH
    global MAX_WORD_LENGTH
    global ITA_DICTIONARY_FILE
    global ENG_DICTIONARY_FILE
    global LOG_LEVEL

    config_file = "config.cfg"
    if not os.path.isfile(config_file):
        logger.error("No \"config.cfg\" configuration file found in the program directory !")
        raise FileNotFoundError("No \"config.cfg\" configuration file found in the program directory !")

    external_settings = dict()
    with open(config_file, "rt") as f:
        for line in f.readlines():
            if not line.startswith("#"):
                tokens = line.split("=")
                if len(tokens) == 2:
                    external_settings[tokens[0]] = tokens[1]

    ITA_DICTIONARY_FILE = str(external_settings.get("ITA_DICTIONARY_FILE", "")).rstrip()
    if not os.path.isfile(ITA_DICTIONARY_FILE):
        logger.error("Invalid ITA_DICTIONARY_FILE parameter !")
        raise FileNotFoundError("Invalid ITA_DICTIONARY_FILE parameter !")

    ENG_DICTIONARY_FILE = str(external_settings.get("ENG_DICTIONARY_FILE", "")).rstrip()
    if not os.path.isfile(ENG_DICTIONARY_FILE):
        logger.error("Invalid ENG_DICTIONARY_FILE parameter !")
        raise FileNotFoundError("Invalid ENG_DICTIONARY_FILE parameter !")

    SOLR_IP_ADDRESS = str(external_settings.get("SOLR_IP_ADDRESS", "")).rstrip()

    SOLR_PORT_NUMBER = str(external_settings.get("SOLR_PORT_NUMBER", "")).rstrip()
    try:
        int(SOLR_PORT_NUMBER)
    except:
        logger.error("Invalid SOLR_PORT_NUMBER parameter !")
        sys.exit("Invalid SOLR_PORT_NUMBER parameter ! \nSOLR_PORT_NUMBER must me an integer")

    SOLR_CORE_NAME = str(external_settings.get("SOLR_CORE_NAME", "")).rstrip()

    SOLR_MAX_DOCS = str(external_settings.get("SOLR_MAX_DOCS", "")).rstrip()
    try:
        int(SOLR_MAX_DOCS)
    except:
        logger.error("Invalid SOLR_MAX_DOCS parameter !")
        sys.exit("Invalid SOLR_MAX_DOCS parameter ! \nSOLR_MAX_DOCS must me an integer")

    MIN_NUM_OF_OCCURRENCES = str(external_settings.get("MIN_NUM_OF_OCCURRENCES", "")).rstrip()
    try:
        MIN_NUM_OF_OCCURRENCES = int(MIN_NUM_OF_OCCURRENCES)
    except:
        logger.error("Invalid MIN_NUM_OF_OCCURRENCES parameter !")
        sys.exit("Invalid MIN_NUM_OF_OCCURRENCES parameter ! \nMIN_NUM_OF_OCCURRENCES must me an integer")

    MAX_NUM_DOCS_PER_FIRM = str(external_settings.get("MAX_NUM_DOCS_PER_FIRM", "")).rstrip()
    try:
        MAX_NUM_DOCS_PER_FIRM = int(MAX_NUM_DOCS_PER_FIRM)
    except:
        logger.error("Invalid MAX_NUM_DOCS_PER_FIRM parameter !")
        sys.exit("Invalid MAX_NUM_DOCS_PER_FIRM parameter ! \nMAX_NUM_DOCS_PER_FIRM must me an integer")

    MIN_WORD_LENGTH = str(external_settings.get("MIN_WORD_LENGTH", "")).rstrip()
    try:
        MIN_WORD_LENGTH = int(MIN_WORD_LENGTH)
    except:
        logger.error("Invalid MIN_WORD_LENGTH parameter !")
        sys.exit("Invalid MIN_WORD_LENGTH parameter ! \nMIN_WORD_LENGTH must me an integer")

    MAX_WORD_LENGTH = str(external_settings.get("MAX_WORD_LENGTH", "")).rstrip()
    try:
        MAX_WORD_LENGTH = int(MAX_WORD_LENGTH)
    except:
        logger.error("Invalid MAX_WORD_LENGTH parameter !")
        sys.exit("Invalid MAX_WORD_LENGTH parameter ! \nMAX_WORD_LENGTH must me an integer")

    LOG_LEVEL = str(external_settings.get("LOG_LEVEL", "INFO")).rstrip()
    consoleHandler.setLevel(LOG_LEVEL)
    fileHandler.setLevel(LOG_LEVEL)


def get_stemmed_words_count_dict(tokenized_text):
    global italian_words
    it_stemmer = SnowballStemmer('italian')
    en_stemmer = SnowballStemmer('english')
    stemmed_words = dict()
    for word in tokenized_text:
        if italian_words.get(word, 0) != 0:
            stemmed_word = it_stemmer.stem(word)  # italian word
            stemmed_words[stemmed_word] = stemmed_words.get(stemmed_word, 0) + 1
        elif english_words.get(word, 0) != 0:
            stemmed_word = en_stemmer.stem(word)  # english word
            stemmed_words[stemmed_word] = stemmed_words.get(stemmed_word, 0) + 1
    return stemmed_words


def get_text_from_docs(firm_id):
    fieldList = ["titolo", "corpoPagina"]
    partialSolrQueryUrl = getPartialSolrQueryUrl(SOLR_IP_ADDRESS, SOLR_PORT_NUMBER, SOLR_CORE_NAME, fieldList)
    query = partialSolrQueryUrl + "firmId" + "%3A" + str(firm_id) + "&rows=" + str(MAX_NUM_DOCS_PER_FIRM) + "&wt=json"
    connection = urlopen(query)
    response = json.load(connection)
    text = []
    logger.info("docs found = " + str(len(response['response']['docs'])))
    for node in response['response']['docs']:
        text_titolo = node["titolo"]
        text_corpo = node["corpoPagina"]
        text.append(text_titolo)
        text.append(text_corpo)
    return " ".join(text)


def get_lista_ripulita(word_list):
    cleaned_word_list = []
    for word in word_list:
        if isAcceptable(word):
            cleaned_word_list.append(word)
    return cleaned_word_list


def isAcceptable(word):

    if "'" in word:
        word = word.split("'")[1]  # l'uomo ==> uomo

    if not word.isalpha():
        return False  # se la lunghezza è almeno 1 e tutti i caratteri sono lettere

    if len(word) < MIN_WORD_LENGTH or len(word) > MAX_WORD_LENGTH:
        return False

    if word[0] not in list(string.ascii_letters):
        return False

    return True


def getPartialSolrQueryUrl(ipAddress, portNumber, coreName, fieldList, filterQuery=""):
    queryUrl = "http://" + \
               ipAddress + \
               ":" + \
               portNumber + \
               "/solr/" + \
               coreName + \
               "/select?fl=" + \
               '%2C'.join(fieldList) +\
               filterQuery + \
               "&q="
    return queryUrl


def get_all_solr_words_by_field(field):
    # http://localhost:8983/solr/firms/terms?terms.fl=corpoPagina&terms.lower=a&terms.sort=index&terms.prefix=b&terms.limit=-1&wt=xml
    connection = urlopen("http://" + SOLR_IP_ADDRESS + ":" + SOLR_PORT_NUMBER + "/solr/" + SOLR_CORE_NAME + "/terms?" +
                        "terms.fl=" + field +
                        "&"
                        "terms.lower=" + "a"
                        "&"
                        "terms.sort=" + "index"
                        "&"
                        "terms.limit=" + "-1"
                        "&"
                        "wt=json")
    response = json.load(connection)
    word_set = set()
    word_list = response['terms'][field][0::2]
    num_list = response['terms'][field][1::2]
    zip_iterator = zip(word_list, num_list)
    word_dict = dict(zip_iterator)
    for word in word_dict:
        num_occurencies = word_dict[word]
        if num_occurencies > MIN_NUM_OF_OCCURRENCES:
            word_set.add(word)
    return word_set


def get_firm_ids():
    fieldList = ["firmId"]
    partialSolrQueryUrl = getPartialSolrQueryUrl(SOLR_IP_ADDRESS, SOLR_PORT_NUMBER, SOLR_CORE_NAME, fieldList)
    query = partialSolrQueryUrl + "*%3A*" + "&rows=" + SOLR_MAX_DOCS + "&wt=json"
    connection = urlopen(query)
    response = json.load(connection)
    set_ids = set()
    for node in response['response']['docs']:
        set_ids.add(node["firmId"])
    return set_ids


def load_dictionary(file):
    words = dict()
    with open(file, encoding="UTF-8") as f:
        for line in f.readlines():
            words[line.strip()] = 1
    return words


def get_stemmed_words(solr_sorted_words_list):
    global italian_words
    it_stemmer = SnowballStemmer('italian')
    en_stemmer = SnowballStemmer('english')
    stemmed_words = set()
    for word in solr_sorted_words_list:
        if italian_words.get(word, 0) != 0:
            stemmed_words.add(it_stemmer.stem(word))  # italian word
        else:
            stemmed_words.add(en_stemmer.stem(word))  # english word
    return list(stemmed_words)


# Entry point
if __name__ == "__main__":
    main(sys.argv[1:])
