import pandas as pd
import nltk
import re
import stanza
import warnings
import pickle
import datetime
import traceback

from google_play_scraper import Sort, reviews_all
from time import sleep
from urllib.parse import quote
from collections import Counter
from nltk.corpus import stopwords
from gensim.models import ldamodel
from sqlalchemy import create_engine
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()

try:
    print('---------- 1. Updating comments from Google Play')
    app_id = 'br.gov.meugovbr'
    result = reviews_all(
        app_id,
        sleep_milliseconds=0, # defaults to 0
        lang='pt', # defaults to 'en'
        country='br', # defaults to 'us'
        sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT
    )
    df = pd.DataFrame.from_dict(result)
    df.to_csv('data/data_android.csv')

    #######################################################################################

    print('---------- 2. Processing comments from Google Play')
    df = pd.read_csv('data/data_android.csv')

    df.drop('Unnamed: 0',axis=1,inplace=True)
    df_ngram = df.copy()
    df_ngram['score'] = df_ngram['score'].map({1: 'Ruim', 2: 'Ruim', 3: 'Neutro', 4: 'Bom', 5: 'Bom'})
    df_ngram[df_ngram['score'] == 'Ruim'].head()

    df_bad = df_ngram[df_ngram['score']=='Ruim']
    data=df_bad['content'].to_frame()
    data=data.rename(columns={'content':'text'})
    data['text']=data['text'].str.lower()
    data['text'].head()
    data[data.isnull().any(axis=1)]
    for letter in '1234567890.(/':
        data['text']= data['text'].str.replace(letter,'')
    data['text']=data['text'].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')

    #######################################################################################

    print('---------- 3. Starting Model')
    with open('model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    data['text'] = data['text'].apply(tokenizer.tokenize)
            
    stopwords_list = stopwords.words('portuguese')

    stanza.download('pt')
    nlp = stanza.Pipeline('pt', tokenize_no_ssplit=True, use_gpu= True)

    def basic_clean(text):
        import nltk
        import re
        import unicodedata
        """
        A simple function to clean up the data. All the words that
        are not designated as a stop word is then lemmatized after
        encoding and basic regex parsing are performed.
        """
        text_lemma=''
        for sent in nlp(text).sentences:
            for word in sent.words:
                text_lemma += word.lemma + " "

        stopwords = nltk.corpus.stopwords.words('portuguese')

        text_lemma = (unicodedata.normalize('NFKD', text_lemma)
                      .encode('ascii', 'ignore')
                      .decode('utf-8', 'ignore')
                      .lower())

        words = re.sub(r'[^\w\s]', '', text_lemma).split()

        return [word for word in words if word not in stopwords]

    data['text']=data['text'].progress_apply(basic_clean)
    data['text'].to_csv('data/data_bad_lemma.csv', index=False)

    data = pd.read_csv("data/data_bad_lemma.csv",converters={"text": lambda x: x.strip("[]").replace("'","").split(", ")})

    all_words = [word for tokens in data['text'] for word in tokens]
    sentence_lengths = [len(tokens) for tokens in data['text']]

    VOCAB = sorted(list(set(all_words)))

    print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    print("Max sentence length is %s" % max(sentence_lengths))

    counted_words = Counter(all_words)

    words = []
    counts = []
    for letter, count in counted_words.most_common(25):
        words.append(letter)
        counts.append(count)

    def LDA_topics(model, num_topics):
        word_dict = {};
        for i in range(num_topics):
            words = model.show_topic(i, topn = 50);
            word_dict['Words of Topic ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
        return pd.DataFrame(word_dict)

    data_text=pd.DataFrame(data['text'])
    data_text=data_text.rename(columns={'text':'text'})
    num_topics=8

    lda2vec = ldamodel.LdaModel.load('model/lda2model')

    # loading
    with open('model/lda2model.id2word', 'rb') as handle:
        id2word1 = pickle.load(handle)

    #######################################################################################

    print('---------- 4. Classifying')

    LDA_topics(lda2vec, num_topics)

    df_bad['processed_content'] = data['text']

    import unicodedata
    def basic_clean(text):
        """
        A simple function to clean up the data. All the words that
        are not designated as a stop word is then lemmatized after
        encoding and basic regex parsing are performed.
        """
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('portuguese')
        text = (unicodedata.normalize('NFKD', text)
                .encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower())
        words = re.sub(r'[^\w\s]', '', text).split()
        return [wnl.lemmatize(word) for word in words if word not in stopwords]

    from tqdm import tqdm

    column_names = ["topic0","topic1","topic2","topic3","topic4","topic5","topic6","topic7"]

    df_topics = pd.DataFrame(columns = column_names)
    df_bad['processed_content'] = df_bad['processed_content'].dropna()

    for row in tqdm(df_bad['processed_content'].dropna()):
        words = row
        new_text_corpus =  id2word1.doc2bow(words)
        topic = 0
        prob = 0
        result = lda2vec.get_document_topics(new_text_corpus)
        topic0 = 0
        topic1 = 0
        topic2 = 0
        topic3 = 0
        topic4 = 0
        topic5 = 0
        topic6 = 0
        topic7 = 0
        # 0
        try:
            if result[0][0] == 0:
                topic0 = result[0][1]
            elif result[0][0] == 1:
                topic1 = result[0][1]
            elif result[0][0] == 2:
                topic2 = result[0][1]
            elif result[0][0] == 3:
                topic3 = result[0][1]
            elif result[0][0] == 4:
                topic4 = result[0][1]
            elif result[0][0] == 5:
                topic5 = result[0][1]
            elif result[0][0] == 6:
                topic6 = result[0][1]
            elif result[0][0] == 7:
                topic7 = result[0][1]
        except:
            pass

        # 1
        try:
            if result[1][0] == 0:
                topic0 = result[1][1]
            elif result[1][0] == 1:
                topic1 = result[1][1]
            elif result[1][0] == 2:
                topic2 = result[1][1]
            elif result[1][0] == 3:
                topic3 = result[1][1]
            elif result[1][0] == 4:
                topic4 = result[1][1]
            elif result[1][0] == 5:
                topic5 = result[1][1]
            elif result[1][0] == 6:
                topic6 = result[1][1]
            elif result[1][0] == 7:
                topic7 = result[1][1]
        except:
            pass

        # 2
        try:
            if result[2][0] == 0:
                topic0 = result[2][1]
            elif result[2][0] == 1:
                topic1 = result[2][1]
            elif result[2][0] == 2:
                topic2 = result[2][1]
            elif result[2][0] == 3:
                topic3 = result[2][1]
            elif result[2][0] == 4:
                topic4 = result[2][1]
            elif result[2][0] == 5:
                topic5 = result[2][1]
            elif result[2][0] == 6:
                topic6 = result[2][1]
            elif result[2][0] == 7:
                topic7 = result[2][1]
        except:
            pass

        # 3
        try:
            if result[3][0] == 0:
                topic0 = result[3][1]
            elif result[3][0] == 1:
                topic1 = result[3][1]
            elif result[3][0] == 2:
                topic2 = result[3][1]
            elif result[3][0] == 3:
                topic3 = result[3][1]
            elif result[3][0] == 4:
                topic4 = result[3][1]
            elif result[3][0] == 5:
                topic5 = result[3][1]
            elif result[3][0] == 6:
                topic6 = result[3][1]
            elif result[3][0] == 7:
                topic7 = result[3][1]
        except:
            pass

        # 4
        try:
            if result[4][0] == 0:
                topic0 = result[4][1]
            elif result[4][0] == 1:
                topic1 = result[4][1]
            elif result[4][0] == 2:
                topic2 = result[4][1]
            elif result[4][0] == 3:
                topic3 = result[4][1]
            elif result[4][0] == 4:
                topic4 = result[4][1]
            elif result[4][0] == 5:
                topic5 = result[4][1]
            elif result[4][0] == 6:
                topic6 = result[4][1]
            elif result[4][0] == 7:
                topic7 = result[4][1]
        except:
            pass

        # 5
        try:
            if result[5][0] == 0:
                topic0 = result[5][1]
            elif result[5][0] == 1:
                topic1 = result[5][1]
            elif result[5][0] == 2:
                topic2 = result[5][1]
            elif result[5][0] == 3:
                topic3 = result[5][1]
            elif result[5][0] == 4:
                topic4 = result[5][1]
            elif result[5][0] == 5:
                topic5 = result[5][1]
            elif result[5][0] == 6:
                topic6 = result[5][1]
            elif result[5][0] == 7:
                topic7 = result[5][1]
        except:
            pass

        # 6
        try:
            if result[6][0] == 0:
                topic0 = result[6][1]
            elif result[6][0] == 1:
                topic1 = result[6][1]
            elif result[6][0] == 2:
                topic2 = result[6][1]
            elif result[6][0] == 3:
                topic3 = result[6][1]
            elif result[6][0] == 4:
                topic4 = result[6][1]
            elif result[6][0] == 5:
                topic5 = result[6][1]
            elif result[6][0] == 6:
                topic6 = result[6][1]
            elif result[6][0] == 7:
                topic7 = result[6][1]
        except:
            pass

        # 7
        try:
            if result[7][0] == 0:
                topic0 = result[7][1]
            elif result[7][0] == 1:
                topic1 = result[7][1]
            elif result[7][0] == 2:
                topic2 = result[7][1]
            elif result[7][0] == 3:
                topic3 = result[7][1]
            elif result[7][0] == 4:
                topic4 = result[7][1]
            elif result[7][0] == 5:
                topic5 = result[7][1]
            elif result[7][0] == 6:
                topic6 = result[7][1]
            elif result[7][0] == 7:
                topic7 = result[7][1]
        except:
            pass

        df_row = {
            'topic0': topic0,
            'topic1': topic1,
            'topic2': topic2,
            'topic3': topic3,
            'topic4': topic4,
            'topic5': topic5,
            'topic6': topic6,
            'topic7': topic7
        }
        df_topics = df_topics.append(df_row, ignore_index = True)

    df_topics.to_csv('data/lda2vec_result_prob_topic.csv', index=False)
    df_topics = pd.read_csv('data/lda2vec_result_prob_topic.csv')
    df_topics[['topic0', 'topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6',
               'topic7']] = df_topics[['topic0', 'topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6',
                                       'topic7']].astype('float')

    df_topics['result_topic'] = df_topics[['topic0', 'topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6',
                                           'topic7']].idxmax(axis=1)

    df_bad.join(df_topics).to_csv('data/lda2vec_result_all_data.csv', index=False)
    df_bad_final = pd.read_csv('data/lda2vec_result_all_data.csv')
    df_bad = pd.read_csv('data/lda2vec_result_all_data.csv')

    words_bad = basic_clean(''.join(str(df_bad['processed_content'].tolist())))
    words_bad = list(filter(('nao').__ne__, words_bad))

    LDA_topics(lda2vec, num_topics).to_csv('data/lda2vec_topic_words.csv', index=False)
    LDA_topics(lda2vec, num_topics)

    df_bad_interval = df_bad_final.copy()
    df_bad_interval['at'] = pd.to_datetime(df_bad_interval['at'])

    df_bad_interval = df_bad_interval.set_index('at')
    df_bad_interval = df_bad_interval.groupby([pd.Grouper(freq='15D'),'result_topic'])
    df_bad_interval = df_bad_interval.count().reset_index()
    df_bad_interval[['at','result_topic','score']].to_csv('data/lda2vec_table_result_original_result.csv', index=False)

    df_bad_interval['result_topic'] = df_bad_interval['result_topic'].map(
        {
            'topic0': 'Reconhecimento Facial',
            'topic1': 'Problema Genérico',
            'topic2': 'Aplicativo Lento e/ou Complicado',
            'topic3': 'Problema Genérico',
            'topic4': 'Acesso ao Aplicativo/Conta Gov.br',
            'topic5': 'Problema Genérico',
            'topic6': 'Abrir o Aplicativo',
            'topic7': 'Abrir o Aplicativo',
        }
    )

    df_bad_interval.to_csv('data/lda2vec_table_result.csv', index=False)

    df_db = pd.read_csv('data/lda2vec_result_all_data.csv')
    df_db['result_topic_named'] = df_db['result_topic'].map(
        {
            'topic0': 'Reconhecimento Facial',
            'topic1': 'Problema Genérico',
            'topic2': 'Aplicativo Lento e/ou Complicado',
            'topic3': 'Problema Genérico',
            'topic4': 'Acesso ao Aplicativo/Conta Gov.br',
            'topic5': 'Problema Genérico',
            'topic6': 'Abrir o Aplicativo',
            'topic7': 'Abrir o Aplicativo',
        }
    )

    #######################################################################################

    print('---------- 5. Saving results to local database')

    print("Open connection")
    engine = create_engine('postgresql://comments:%s@db:5432/comments' % quote('o&@3p66e67t%w1'))

    print("Start Populate DB")
    df_db = pd.read_csv('data/lda2vec_result_all_data.csv')
    df_db['origin'] = "Google Play"
    df_db['result_topic_named'] = df_db['result_topic'].map(
        {
            'topic0': 'Reconhecimento Facial',
            'topic1': 'Problema Genérico',
            'topic2': 'Aplicativo Lento e/ou Complicado',
            'topic3': 'Problema Genérico',
            'topic4': 'Acesso ao Aplicativo/Conta Gov.br',
            'topic5': 'Problema Genérico',
            'topic6': 'Abrir o Aplicativo',
            'topic7': 'Abrir o Aplicativo',
        }
    )

    print("Delete if exists (table comments)")
    with engine.connect() as connection:
        result = connection.execute("DROP TABLE IF EXISTS comments;")

    print("Save to db (lda2vec_result_all_data.csv)")
    df_db.to_sql('comments', engine)

    df_all_data = pd.read_csv('data/data_android.csv')

    print("Delete if exists (table all_data)")
    with engine.connect() as connection:
        result = connection.execute("DROP TABLE IF EXISTS all_data;")    

    print("Save to db (data_android.csv)")
    df_all_data.to_sql('all_data', engine)

    df = pd.read_csv('data/lda2vec_result_all_data.csv')
    df_m = pd.read_csv('data/monitoramento_inst.csv', thousands=',')
    df_d = pd.read_csv('data/monitoramento_dau.csv', thousands=',')

    df['at'] = pd.to_datetime(df['at'],infer_datetime_format=True)
    df['at'] = df['at'].dt.date

    df_m['Date']  = pd.to_datetime(df_m['Date'],format='%b %d, %Y')
    df_d['Date']  = pd.to_datetime(df_d['Date'],format='%b %d, %Y')

    df_m['Date'] = df_m['Date'].dt.date
    df_d['Date'] = df_d['Date'].dt.date

    df = df[df['at'] > df_d['Date'][0]]

    df = df.set_index('at')
    df_m = df_m.set_index('Date')
    df_d = df_d.set_index('Date')

    df_d = df_d.drop(columns=['Notes'])
    df_d = df_d.rename(columns={'Daily Active Users (DAU) (Unique users, 30 days rolling average, Daily): All countries / regions':'Daily Active Users (DAU)'})

    df_m = df_m.drop(columns=['Notes'])
    df_m = df_m.rename(columns={'Install base (All devices, Unique devices, Per interval, Daily): All countries / regions':'Android'})

    df = df.join(df_d)
    df = df.join(df_m)

    df.dropna(subset=['Android'],inplace=True)

    df.index = pd.to_datetime(df.index)

    df_group = df.groupby([pd.Grouper(freq='15D'),'result_topic'])
    df_test = df.groupby([pd.Grouper(freq='1D'),'result_topic']).agg({'score':'count','Android': 'mean','Daily Active Users (DAU)': 'mean'})

    df_final = pd.DataFrame()

    for date, new_df in df_test.groupby(level=0):
        # print(new_df['result_topic'].sum())
        # print(new_df['Instalações Diarias'].mean())
        new_df['PPMI'] = (new_df['score']/new_df['Android'].mean()) *1000
        new_df['Normalized'] = (new_df['score']/new_df['score'].sum())
        new_df['PPMDAU'] = (new_df['score']/new_df['Daily Active Users (DAU)'].mean()) *1000
        df_final = df_final.append(new_df)
    
    df_final = df_final.reset_index()
    df_final['Date'] = df_final['level_0']
    df_final = df_final.drop('level_0',axis=1)
    df_final['reviews_count'] = df_final['score']
    df_final = df_final.drop(['reviews_count','Daily Active Users (DAU)'],axis=1)
    df_final['result_topic'] = df_final['result_topic'].map(
        {
            'topic0': 'Reconhecimento Facial',
            'topic1': 'Problema Genérico',
            'topic2': 'Aplicativo Lento e/ou Complicado',
            'topic3': 'Problema Genérico',
            'topic4': 'Acesso ao Aplicativo/Conta Gov.br',
            'topic5': 'Problema Genérico',
            'topic6': 'Abrir o Aplicativo',
            'topic7': 'Abrir o Aplicativo',
        }
    )

    df_final.to_csv('data/monitoramento_norm.csv',index=False)

    print("Delete if exists (table monitoramento_norm)")
    with engine.connect() as connection:
        result = connection.execute("DROP TABLE IF EXISTS monitoramento_norm;")

    print("Save to db (monitoramento_norm)")
    df_final.to_sql('monitoramento_norm', engine)

    df_final = df_final.drop(['result_topic','score','Android'],axis=1)
    df_final = df_final.set_index('Date')
    df_final = df_final.join(df)
    df_final.to_csv('data/all_data_norm.csv',index=False)
    
    print("Delete if exists (table monitoramento_inst)")
    with engine.connect() as connection:
        result = connection.execute("DROP TABLE IF EXISTS monitoramento_inst;")

    print("Save to db (monitoramento_inst)")
    df_m.to_sql('monitoramento_inst', engine)


    print("Delete if exists (table monitoramento_dau)")
    with engine.connect() as connection:
        result = connection.execute("DROP TABLE IF EXISTS monitoramento_dau;")

    print("Save to db (monitoramento_dau)")
    df_d.to_sql('monitoramento_dau', engine)

    print("Delete if exists (all_data_norm)")
    with engine.connect() as connection:
        result = connection.execute("DROP TABLE IF EXISTS all_data_norm;")

    print("Save to db (all_data_norm)")
    all_data_norm = pd.read_csv('data/all_data_norm.csv')
    all_data_norm.to_sql('all_data_norm', engine)

except Exception as e:
    print(e)
    traceback.print_exc()
    print("Nothing to do")
