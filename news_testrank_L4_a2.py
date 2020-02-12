from textrank4zh import TextRank4Keyword, TextRank4Sentence
import jieba

f = open("C:\\Users\\tgu2\\.spyder-py3\\RS6-master\\L4\\textrank\\news.txt",encoding='utf-8')
text = f.read()
f.close()

tr4w = TextRank4Keyword()
tr4w.analyze(text=text, lower=True, window=3)
print('关键词：')
for item in tr4w.get_keywords(20, word_min_len=2):
    print(item.word, item.weight)


tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source = "all_filters")
print('摘要：')
for item in tr4s.get_key_sentences(num=3):
    print(item.index, item.weight, item.sentence)