## ShareGPT_from_Tweets

ローカルLLM（LM-Studio）を用いてtweets.jsから質問→回答のShareGPT形式のデータセットを作成するスクリプト。

`extract_plain_text_tweets.py`  
tweets.jsから引用リツイート、メンション、画像などのメディアつきツイート、リンク付きツイートを省いた純粋なテキストツイートのみをjsonlで抜き出すスクリプト。

`generate_sharegpt_from_tweets.py`  
上記で作成したjsonlをもとに、ツイート1つ1つに対して逆質問を生成するスクリプト（デフォルトはLM-Studioのgpt-oss-20b）。
質問→回答のShareGPT形式のjsonlを生成する。

-----

A script that uses a local LLM (LM Studio) to create a ShareGPT-format question–answer dataset from tweets.js.

`extract_plain_text_tweets.py`  
A script that extracts only pure text tweets from tweets.js as JSONL, excluding quote retweets, mentions, tweets with images or other media, and tweets containing links.

`generate_sharegpt_from_tweets.py`  
A script that, based on the JSONL created above, generates reverse questions for each individual tweet (by default using LM Studio’s gpt-oss-20b).  
It outputs ShareGPT-format JSONL consisting of question–answer pairs.

