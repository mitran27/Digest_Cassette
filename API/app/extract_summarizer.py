
import urllib.request as scrap
import re
import bs4

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scipy import spatial
import networkx as nx


def cosine(l1,l2):      
    
    
    try:
        
        if(len(l1)!=len(l2)):raise ValueError('Dimension of two vectors must be same')
        c=0
        for i in range(len(l1)): c+= l1[i]*l2[i]
        cosine = c / float((sum(l1)*sum(l2))**0.5)
        return cosine
    except Exception as error:
        
        print(error)
    
    
    
def get_wiki(topic):
    para=scrap.urlopen('https://en.wikipedia.org/wiki/'+topic).read()
    soup=bs4.BeautifulSoup(para,'lxml')
    
    
    text=""
    for para in soup.find_all('p'):
        
        s=re.sub("\[\d+\]","",para.text)
        
        s= re.sub(r"\([^()]*\)", "", s)
        s=re.sub("""[^A-Za-z0-9.,: ]+""",' ',s)
        s+=' '
        text+=s 
        
    return text    
        
def mean(x):
    return sum(x)/len(x)
def extractive_summarise(topic,length=10):
    
    def clean(sent):
        
        
        s= re.sub(r"\([^()]*\)", "",sent)
        s=re.sub("""[^A-Za-z0-9.,: ]+""",' ',s)
        
    
        impsent=[word for word in s.lower().split() if word not in stopwords.words('english')]
        return impsent,s.lower()



    source=get_wiki(topic)

    
    sentences=source.split('.')
    nodes=[] # (important word sentence) in the paragraph are the nodes of the graph 
    sentence_list=[] # cleaned sentence
    
    
    max_len=0
    for i in sentences:
        
        imp_word,sent=clean(i)
        
        if(len(imp_word)==0):continue # skip if there are no important words
        
        nodes.append(imp_word)   # store the import nodes
        sentence_list.append(sent)  # store the sentences
        
     
    ls=min(len(sentence_list),300)   
    
    
    
    
    # calculating embeddings 
    """
    create word 2 vec object for 100 dimension
    
    replace the words with their corresponding embedding vectors in all sentences
    
    Add the sentence: no words x 100  take sume on axis 0  (adding the semmatic  of words)
    
    """
    sentence_embeddings=[]
    
    word_embedding=Word2Vec(sentences=nodes,vector_size=100,window=5,min_count=1)
    
    for i in range(ls):
        
        
        sentence=nodes[i]
        sent_emb=[]
        for w in sentence:
            sent_emb.append(word_embedding.wv[w])
        sent_emb=np.array(sent_emb)    
        sentence_embeddings.append(mean(sent_emb))   
    Graph=np.zeros((ls,ls))
    print("graph created")
    for i in range(ls):
        for j in range(ls):
            if(i!=j):
                Graph[i][j] =  1 - spatial.distance.cosine(sentence_embeddings[i], sentence_embeddings[j])
    
    
    
    

    nx_graph = nx.from_numpy_array(Graph)
    scores = nx.pagerank(nx_graph,max_iter=1000)
    print("text rank completed")
        
    top_sentence = sorted(((scores[i],sentence_list[i]) for i in range(ls)), reverse=True)
    return top_sentence[:length]
      


    


            
        
   
    

        
    
    
    
        
        
        
    
    
    
     
