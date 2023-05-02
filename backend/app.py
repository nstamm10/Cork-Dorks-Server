import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from decouple import config
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import ssl
import requests
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = config('MY_SQL_PASS')
MYSQL_PORT = 3306
MYSQL_DATABASE = "corkdorks"
KEY = 'acfa3a9e02eb4fe98c201ddb70f3333b'

mysql_engine = MySQLDatabaseHandler(MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
def sql_search(country):
    query_sql = f"""SELECT title FROM wine_data WHERE LOWER( country ) LIKE '%%{country.lower()}%%' limit 10"""
    keys = ["title"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys,i)) for i in data])

def price_and_points(max_price, min_points):
    query_sql = f"""SELECT title, price, points FROM wine_data WHERE price <= {max_price} AND points >= {min_points} ORDER BY points DESC limit 3"""
    keys = ["title", "price", "points"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys,i)) for i in data])

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    price= request.args.get("max_price")
    points= request.args.get("min_points")
    pnp = price_and_points(price,points)
    return pnp

'''
Returns inverted index representation of wine descriptions
Returns dictionary of the form: {term : [(wine_title, count), ...]}
'''
def description_inverted_index(price=10000, country = None, variety = None):
    # Fetching Data
    query_sql = f"""SELECT title, description, country, designation, province,region_1,region_2,variety,winery 
    FROM wine_data"""
    where_statement = ""

    if price:
        where_statement += f""" WHERE price <= {price}"""

    if country:
            if len(where_statement) == 0:
                where_statement += f""" WHERE country = '{country}'"""
            else: 
                where_statement += f""" AND country = '{country}'"""

    if variety:
         if len(where_statement) == 0:
            where_statement += f""" WHERE variety = '{variety}'"""
         else: 
            where_statement += f""" AND variety = '{variety}'"""

    query_sql += where_statement

    data = mysql_engine.query_selector(query_sql)
    lst =[]
    for i in data:
        dict = {}
        words = ""
        for j in i:
            if j:
                words+=(" "+j)
        dict["title"] = i[0]
        dict["description_words"] = words
        lst.append(dict)
   
    input_dict = json.loads(json.dumps(lst))
    tokenizer = TreebankWordTokenizer()
    for i in range(len(input_dict)):
        input_dict[i]["description_words"] = tokenizer.tokenize(input_dict[i]["description_words"])
        input_dict[i]["description_words"] = [x.lower() for x in input_dict[i]["description_words"]]
    
    titles = [x['title'] for x in input_dict]
    title_dict = {k : v for k, v in enumerate(titles)}
    # Building Inverted Index
    dic = {}
    for i in range(len(input_dict)):
        for tok in input_dict[i]['description_words']:
            if tok not in dic.keys():
                dic[tok] = {i : 1}
            else:
                if i not in dic[tok].keys():
                    dic[tok][i] = 1
                else:
                    dic[tok][i] += 1
        
    inv_index = {}
    for tok in dic.keys():
        inv_index[tok] = []
        for k, v in dic[tok].items():
            inv_index[tok].append((k, v))
        inv_index[tok].sort(key = lambda tup : tup[0])
    return inv_index, title_dict

'''
Returns sorted wine titles by number of or_words contained in titles description. If
wine title contains 0 'or_words', title is not in list.
'''
def boolean_search(or_words, description):
    inv_index = description[0]
    title_dict = description[1]
  
    postings = inv_index[or_words[0]]
    for i in range(1, len(or_words)):
        try:
            current_lst = inv_index[or_words[i]]
            postings = or_merge_postings(postings, current_lst)
        except KeyError:
            pass
    title_postings = []
    for posting in postings:
        title_postings.append((title_dict[posting[0]], posting[1]))
    titles = sorted(title_postings, key= lambda x: x[1], reverse=True)
    titles = [val[0] for val in title_postings if val[1] > 0]
    return titles
    
def or_merge_postings(lst1, lst2):
    p1, p2 = 0, 0
    output = []
    while p1 < len(lst1) and p2 < len(lst2):
        if lst1[p1][0] > lst2[p2][0]:
            output.append(lst2[p2])
            p2 += 1
        elif lst1[p1][0] < lst2[p2][0]:
            output.append(lst1[p1])
            p1 += 1
        else:
            count = lst1[p1][1] + lst2[p2][1]
            output.append((lst1[p1][0], count))
            p1 += 1
            p2 += 1
    while p2 < len(lst2):
        output.append(lst2[p2])
        p2 += 1
    while p1 < len(lst1):
        output.append(lst1[p1])
        p1 += 1
    return output

def download_packages():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
download_packages()

def tokenizer(sentence):
    return word_tokenize(sentence)

def pos_tagger(tokens):
    return nltk.pos_tag(tokens)

def stopword_treatment(tokens):
    stopword = stopwords.words('english')
    res = []
    for tok in tokens:
        if tok[0].lower() not in stopword:
            res.append(tuple([tok[0].lower(), tok[1]]))
    return res

pos_tag_map = {
    'NN': [ wn.NOUN ],
    'JJ': [ wn.ADJ, wn.ADJ_SAT ],
    'RB': [ wn.ADV ],
    'VB': [ wn.VERB ]
}

def pos_tag_convert(pos_tag):
    root = pos_tag[0:2]
    try:
        pos_tag_map[root]
        return pos_tag_map[root]
    except KeyError:
        return ''
    
def get_synsets(tokens):
    synsets = []
    for tok in tokens:
        pos_tag = pos_tag_convert(tok[1])
        if pos_tag == '':
            continue
        else:
            synsets.append(wn.synsets(tok[0], pos_tag))
    return synsets

def get_synset_toks(synsets):
    tokens = {}
    for synset in synsets:
        for s in synset:
            if s.name() in tokens:
                tokens[s.name().split('.')[0]] += 1
            else:
                tokens[s.name().split('.')[0]] = 1
    return tokens

def get_hypernyms(synsets):
    hypernyms = []
    for synset in synsets:
        for s in synset:
            hypernyms.append(s.hypernyms())
    return hypernyms

def get_hypernym_toks(hypernyms):
    tokens = {}
    for hypernym in hypernyms:
        for h in hypernyms:
            for hh in h:
                if hh.name().split('.')[0] in tokens:
                    tokens[hh.name().split('.')[0]] += 1
                else:
                    tokens[hh.name().split('.')[0]] = 1
    return tokens

def underscore(tokens):
    new_toks = {}
    for k in tokens.keys():
        x = re.sub(r'_', ' ', k)
        new_toks[x] = tokens[k]
    return new_toks

def query_expansion(query):
    tokenizer = TreebankWordTokenizer()
    description = tokenizer.tokenize(query)
    tokens = pos_tagger(description)
    tokens = stopword_treatment(tokens)

    synsets = get_synsets(tokens)
    synoynms = get_synset_toks(synsets)
    synoynms = underscore(synoynms)

    hypernyms = get_hypernyms(synsets)
    hypernyms = get_hypernym_toks(hypernyms)
    hypernyms = underscore(hypernyms)

    expanded = {**synoynms, **hypernyms}
    expanded = list(expanded.keys())

    tagged_tokens = nltk.pos_tag(description)
    adjectives = [word for word, tag in tagged_tokens if tag == 'JJ' or tag =='NN']

    final_list = list(set(adjectives).union(set(expanded)))
    return final_list
  
#query_expansion(query) expanded input query
#doc_dict.values() doc query
#compute cosine_sim(expanded_query, doc_dict.values)
def cosine_sim(query, titles):
    query_sql = f"""SELECT * FROM wine_data"""
    keys = ["country", "description", "designation", "points", "price", "province", "region_1", "region_2", "title", "variety", "winery"]
    data = mysql_engine.query_selector(query_sql)

    doc_dict = {}
    for doc in data:
        if doc['title'] in titles:
            doc_text = f"{doc['description']} {doc['designation']} {doc['points']} {doc['price']} {doc['province']} {doc['region_1']} {doc['region_2']} {doc['variety']} {doc['winery']}"
            doc_tokens = [word for word in TreebankWordTokenizer().tokenize(doc_text) if word not in string.punctuation and not word.isdigit()]
            doc_dict[doc['title']] = doc_tokens

    docs = [" ".join(doc_dict[title]) for title in doc_dict.keys()]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

    #Compute cosine sim
    query_vec = tfidf_vectorizer.transform([' '.join(query)])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix)

    #Get titles and cosine similarities
    titles = list(doc_dict.keys())
    cosine_similarities = cosine_similarities[0]
    
    # Sort results by cosine similarity in descending order
    results = sorted(zip(titles, cosine_similarities), key=lambda x: round(x[1]*100,3), reverse=True)[:3]
    output_titles = [val[0] for val in results]
    return output_titles

#return string which represents rationale for selected wine wines
def rationale(query_words,wine_info,price=None,country = None, variety = None):
    ans = "This wine is recommended because it is "
    tokenizer = TreebankWordTokenizer()
    description = tokenizer.tokenize(wine_info['description'])
    query_words = set(query_words)

    pos_tags = nltk.pos_tag(query_words)
    adjectives = [word for (word, tag) in pos_tags if tag.startswith('JJ')]

    matched_words=[]
    for word in adjectives:
        if word in description and word != "wine":
            matched_words.append(word)

    for ind, word in enumerate(matched_words):
        ans+= str(word)
        if ind!= len(matched_words)-1:
            ans +=", "
        else:
            ans += ""
    if price:
        if len(matched_words) == 0:
            ans+= " $"+str(wine_info['price'])
        else:
            ans+= " and is $"+str(wine_info['price'])
    if country:
        ans += ", is from "+str(country)
    if variety:
        ans += ", and is of the "+str(variety)+" variety."
    
    return ans


@app.route("/description")
def description_search():
    price= request.args.get("max_price")
    query = request.args.get("description")
    country = request.args.get("country")
    variety = request.args.get("variety")
    expanded_query = query_expansion(query)
    inv_index = description_inverted_index(price=price, country=country, variety=variety)
    titles = boolean_search(expanded_query, inv_index)
    cos_sim = cosine_sim(expanded_query, titles) 
    query_sql = f"""SELECT * FROM wine_data"""
    keys = ["country", "description", "designation", "points", "price", "province",
            "region_1", "region_2", "title", "variety", "winery"]
    data = mysql_engine.query_selector(query_sql)
    dic = [dict(zip(keys,i)) for i in data if i["title"] in cos_sim]
    for wine in dic:
        variety = wine['variety']
        url = f"https://api.spoonacular.com/food/wine/dishes?apiKey={KEY}&wine={variety}"
        response = requests.get(url)
        try:
            wine['pairing'] = json.loads(response.text)['pairings'][0]
        except KeyError:
            wine['pairing'] = 'No Pairing Found'     

        wine['rationale'] = rationale(expanded_query,wine,price=price, variety=variety, country=country)
    return json.dumps(dic)


# app.run(debug=True)