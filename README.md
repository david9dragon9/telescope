# 🔭 Telescope: LLM enhanced, multi-stage web search
Telescope is a library for intelligent web searches that uses LLMs in different stages to perform smart filtering, information extraction, embedding search, and final summarization to give an accurate answer to each search.

# Capabilities
Telescope has several different stages of LLM-enhanced search:
- Web search: Web search using DuckDuckGo and result scraping to obtain full text for each search result. This can be substituted for any other search tool (e.g. Wikipedia), or even your local documents.
- Sentence Transformers embedding retrieval: Sentence Transformers embeddings are evaluated for each retrieval result, as well as for the query, and the query embedding is used to search over retrieval results.
- LLM extraction: This stage uses an LLM to extract only information relevant to a given query from often long search results, in order to create a shorter, more concise summary with all of the key information.
- LLM filtering: This stage uses selection rank with an LLM comparing two search results to determine which is more relevant to a given query. This produces the top k relevant results, which are then used in the next stage. Few-shot prompting is used to ensure that the LLM follows the proper format.
- LLM summary: Finally, the LLM is given all of the search results and uses them to form a final summary that is based off of the information contained in the search results.

Other points of interest:
- Hydra for easy configuration
- LLM agnostic

# Usage
When running Telescope, first consider the following parameters:
- layers: This is by far the most important parameter. It specifies the sequence of layers that you would like to run in your search funnel. For example, if you set this to ["embedding", "extraction"], it will run the embedding and extraction layers and return the final results. If you set this to "all", Telescope will automatically run ["embedding", "extraction", "filtering", "summary"]
- out_results: This is the number of results that you would like to be output from each layer in the layers that you specified. For example, if you set this to [10, 4], this indicates that you would like 10 results to be output after the first layer and 4 results to be output after the second layer. Note that the extraction layer does not change the number of results, so the output results from an extraction layer are the same number as the previous layer.
- num_initial_results: This is the number of initial results to start with from the DuckDuckGo Web Search.

- input_path: If you would like to specify a set of pre-determined queries to run the funnel on, use this parameter and specify a path to a txt file. Place one query on each line. Example txt file:
```
capital of France
history of mooncakes
```
- input_results: If you would like to continue running the funnel with spreviously calculated results from earlier stages, saved into JSON format in a folder, then specify the top-level directory here. Telescope will automatically read the final results from the run in the folder if it exists and use it as input.
- output_path: Directory to save the intermediate and final results from this run into. For each search query (both interactive and pre-determined), a folder will be created with the search query as name. Inside the folder, there will be two files: intermediate.json and final.json, which will contain the intermediate and final results respectively.

- chat_model_name: This is the name of the Chat model to use for the Extraction, Filtering, and Summary stages (e.g. lmsys/vicuna-7b-v1.5).
- embedding_model_name: This is the name of the text embedding model to use for the Embedding stage (e.g. hkunlp/instructor-base).
- tool_name: This is the name of the search tool that you would like to use. At the moment, only duckduckgo is supported, though feel free to add your own custom tool.
- max_new_tokens: This is the maximum number of tokens that will be output by the chat model in all stages. This prevents overly long summaries. 512

Then, run `main.py`:
```
python3 main.py --config-name config \
                layers='["embedding", "extraction", "filtering", "summary"]' \
                out_results="[10, 10, 3, 1]" \
                num_initial_results=10 \
                embedding_model_name=hkunlp/instructor-base \
                chat_model_name=lmsys/vicuna-7b-1.5-16k \
                tool_name=duckduckgo \
                output_path=/path/to/outputs
```

Telescope will first run your pre-determined queries, and then an interactive prompt will show up asking you for a search query.