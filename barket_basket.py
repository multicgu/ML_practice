import pandas as pd
import time

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('C:\\Users\\tgu2\\.spyder-py3\\RS6-master\\L3\\MarketBasket\\Market_Basket_Optimisation.csv',header=None)

df = df.fillna(111)
temp_set = set()
for id,item in df.items():
    for i in item:
        if i is not 111:
            temp_set.add(i)

df2 = pd.DataFrame(index=range(len(df)),columns=temp_set)
for tup in df.itertuples():
    for i in tup[1:]:
        if i in df2.columns:
            df2.loc[tup[0]][i] = 1
df2=df2.fillna(0)


frequent_items = apriori(df2,min_support=0.03,use_colnames=True)
rules = association_rules(frequent_items,metric="lift",min_threshold=0.5)

print("Frequent_items:",frequent_items)
print("Association_rules:",rules)



# 词云展示
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
#from PIL import Image
import numpy as np
from lxml import etree
from nltk.tokenize import word_tokenize

# 去掉停用词
def remove_stop_words(f):
	stop_words = ['a']
	for stop_word in stop_words:
		f = f.replace(stop_word, '')
	return f

# 生成词云
def create_word_cloud(f):
	print('根据词频，开始生成词云!')
	f = remove_stop_words(f)
	cut_text = word_tokenize(f)
	#print(cut_text)
	cut_text = " ".join(cut_text)
	wc = WordCloud(
		max_words=100,
		width=2000,
		height=1200,
    )
	wordcloud = wc.generate(cut_text)
	# 写词云图片
	wordcloud.to_file("wordcloud_barket.jpg")
	# 显示词云文件
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()
# 数据加载
df = df.replace(111,'a')
df.columns=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']
all_word=' '
for col in df.columns:
    temp_col = ' '.join(df[col])
    all_word = all_word + temp_col
#print(all_word)

# 生成词云
create_word_cloud(all_word)
