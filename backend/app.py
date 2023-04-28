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
import ssl
import requests

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
def description_inverted_index(price=None, minpoint = None, country = None, region = None, winery = None, variety = None):
    # Fetching Data
    query_sql = f"""SELECT title, description, country, designation, province,region_1,region_2,variety,winery 
    FROM wine_data"""
    where_statement = ""

    if price:
        where_statement += f""" WHERE price <= {price}"""
    
    if minpoint:
        if where_statement.isEmpty():
            where_statement += f""" WHERE price >= {minpoint}"""
        else: 
            where_statement += f""" AND price >= {minpoint}"""

    if country:
        if where_statement.isEmpty():
            where_statement += f""" WHERE country = {country}"""
        else: 
            where_statement += f""" AND country = {country}"""

    if region:
        if where_statement.isEmpty():
            where_statement += f""" WHERE region = {region}"""
        else: 
            where_statement += f""" AND region = {region}"""
    
    if winery:
        if where_statement.isEmpty():
            where_statement += f""" WHERE winery = {winery}"""
        else: 
            where_statement += f""" AND winery = {winery}"""

    if variety:
        if where_statement.isEmpty():
            where_statement += f""" WHERE variety = {variety}"""
        else: 
            where_statement += f""" AND variety = {variety}"""

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
        current_lst = inv_index[or_words[i]]
        postings = or_merge_postings(postings, current_lst)
    title_postings = []
    for posting in postings:
        title_postings.append((title_dict[posting[0]], posting[1]))
    title_postings.sort(key = lambda tup : tup[1], reverse=True)
    return list(zip(*title_postings[:3]))[0]
    
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
    tokens = tokenizer(query)
    tokens = pos_tagger(tokens)
    tokens = stopword_treatment(tokens)
    
    synsets = get_synsets(tokens)
    synoynms = get_synset_toks(synsets)
    synoynms = underscore(synoynms)

    hypernyms = get_hypernyms(synsets)
    hypernyms = get_hypernym_toks(hypernyms)
    hypernyms = underscore(hypernyms)

    expanded = {**synoynms, **hypernyms}
    expanded = list(expanded.keys())
    tokenized_original = TreebankWordTokenizer().tokenize(query)
    for tok in tokenized_original:
        if tok not in expanded:
            expanded.append(tok)

    return expanded

#return string which represents rationale for selected wine
#Assumes that the given wine WAS SUCCESFULLY matched
#Please do not call rationale for a wine that we do not have a match for.
#This is because our inverted_index uses an "AND" statement for searching these params
# against the SQL database!
def rationale(query_words,wine_info,price=None, minpoint = None, country = None, region = None, winery = None, variety = None):
    ans = "This wine is recommended because it is "
    tokenizer = TreebankWordTokenizer()
    description = tokenizer.tokenize(wine_info['description'])

    matched_words=[]
    for word in query_words:
        if word in description:
            matched_words.append(word)

    for ind, word in enumerate(matched_words):
        ans+= str(word)
        if ind!= len(matched_words)-1:
            ans +=", "
        else:
            ans += " "
    if price:
        if len(matched_words) == 0:
            ans+= "less than or equal to "+str(price)
        else:
            ans+= "and is less than or equal to $"+str(price)
    if minpoint:
        ans += ", has a rating of at least "+str(minpoint)
    if country:
        ans += ", is from "+str(country)
    if region:
        ans += ", is from the "+str(region) + " region"
    if winery:
        ans += ", is from "+str(winery)
    if variety:
        ans += ", and is of the "+str(variety)+" variety."
        
    return ans



@app.route("/description")
def description_search():
    price= request.args.get("max_price")
    query= request.args.get("description")
    expanded_query = query_expansion(query)
    inv_index = description_inverted_index(price)
    titles = boolean_search(expanded_query, inv_index)
    query_sql = f"""SELECT * FROM wine_data WHERE title in {titles}"""
    keys = ["country", "description", "designation", "points", "price", "province",
            "region_1", "region_2", "title", "variety", "winery"]
    data = mysql_engine.query_selector(query_sql)
    dic = [dict(zip(keys,i)) for i in data]
    for wine in dic:
        variety = wine['variety']
        url = f"https://api.spoonacular.com/food/wine/dishes?apiKey={KEY}&wine={variety}"
        response = requests.get(url)
        try:
            wine['pairing'] = json.loads(response.text)['pairings'][0]
        except KeyError:
            wine['pairing'] = 'No Pairing Found'     

        reasoning = rationale(expanded_query,wine,price)
        wine['rationale'] = reasoning
    return json.dumps(dic)


app.run(debug=True)