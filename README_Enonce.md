# Project 2025 : Information Retrieval

The main goal of this project is to create a search engine.
You will found 2~000 documents from french wikipedia, with 50 queries.

## Work to do

Code and create a search engine (index and reverse index) the matching model (between a query and documents).
You can use classical TF-iDF approache, vector approaches, or combine both of them.
Feel free to explore Re-Ranking and Learning-To-Rank approaches and intent detection techniques.

It is forbidden to use frameworks such as ElasticSearch, Vespa, LangChain, etc. You need to code your own search engine.
On the other side, it is ok if you use external code taken on github or from a paper, but you must cite it.

### Data provided

There is a set of queries in the ```requetes.jsonl``` file, which is a json-line file type. There is one query per line like this:
```{"Answer file": "wiki_042186.txt", "Queries": ["course à pied", "trail"]}```
Each query correspond to an answer file, this means when you perform a query, you should have as answer the right file.
All data files are in a set of documents stored in the follolwing zip file : ```wiki_split_extract_2k.zip```.



## Things to send back

At the end of the project, you will send me by email a report that contains :

- A report explaining what you have done (≃ 5-10 pages + references)
- I want you to write a report. This also means you have to put your name on it and to do an effort of presentation! ;-)
- upload the report in teams before the deadline (github / gitlab accepted)
- No Jupiternote !!!
- ready-to-work script (in python or demonstration) (github / gitlab accepted)
- a link to download the model if you created one (github / gitlab accepted)
- 1 report per group (do not upload one report per student)
- If you send me your report lately, you will be penalised
- Plagianism equal to zero
- No report equal to zero
- the report shall have a methodology, evaluation protocol, and an evaluation analysis of the work you've done.

**Deadline: 16th of November 2025, 23:59 CET**