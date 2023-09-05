from telescope.layers import get_layer
from telescope.llms import get_embedding_llm, LLM
from telescope.tools.duckduckgo import DuckDuckGoSearcher
from colorama import Fore
import hydra
import copy
import json
import os
import tqdm


def run_layers(tool, layer_names, layers, query, num_initial_results, results=None):
    intermediate_results = {}
    if results is None:
        results = tool.search(query, num_results=num_initial_results)
    else:
        results = results["final"]
    intermediate_results["initial"] = copy.deepcopy(results)
    pbar = tqdm.tqdm(zip(layer_names, layers))
    for layer_name, layer in pbar:
        pbar.set_description(f"Currently running layer: {layer_name}")
        results = layer.search(query, results)
        intermediate_results[layer_name] = copy.deepcopy(results)
        print(Fore.GREEN + f"Finished layer: {layer_name}" + Fore.RESET)
    intermediate_results["final"] = results
    return intermediate_results, results


@hydra.main("./telescope/configs/")
def main(cfg):
    if cfg.layers == "all":
        cfg.layers = ["embedding", "filtering", "extraction", "summary"]
    assert len(cfg.layers) == len(
        cfg.out_results
    ), "Number of layers must match number of output results"
    all_layers = []
    llms = {}
    for layer in cfg.layers:
        if layer == "embedding":
            llms["embedding"] = (
                cfg.embedding_model_name,
                get_embedding_llm(cfg.embedding_model_name),
            )
        elif (
            layer in ["extraction", "filtering", "summary"]
            and "instruction" not in llms
        ):
            llms["instruction"] = (
                cfg.chat_model_name,
                LLM(cfg.chat_model_name, max_new_tokens=cfg.max_new_tokens),
            )

    if cfg.tool_name == "duckduckgo":
        tool = DuckDuckGoSearcher()

    for i, layer in enumerate(cfg.layers):
        curr_layer = get_layer(layer, out_results=cfg.out_results[i], llms=llms)
        all_layers.append(curr_layer)

    def process_query(query, results=None):
        print("-----------------------------")
        print("-----------------------------")
        print(f"QUERY: {query}")
        intermediate_results, final_result = run_layers(
            tool,
            cfg.layers,
            all_layers,
            query,
            num_initial_results=cfg.num_initial_results,
            results=results,
        )
        if cfg.output_path is not None:
            output_folder = os.path.join(cfg.output_path, query.replace(" ", "_"))
            os.makedirs(output_folder, exist_ok=True)
            with open(os.path.join(output_folder, "intermediate.json"), "w") as f:
                json.dump(intermediate_results, f)
            with open(os.path.join(output_folder, "final.json"), "w") as f:
                json.dump(final_result, f)
        print(f"FINAL RESULT: {final_result}")

    if cfg.input_path is not None:
        with open(cfg.input_path, "r") as f:
            queries = [x.strip() for x in f.readlines()]
        for query in queries:
            if cfg.input_results is not None:
                path = os.path.join(
                    cfg.input_results, query.replace(" ", "_"), "intermediate.json"
                )
                if os.path.exists(path):
                    with open(path, "r") as f:
                        results = json.load(f)
            else:
                results = None
            process_query(query, results=results)

    while True:
        user_input = input("SEARCH QUERY: ").strip()
        if user_input == "quit":
            break
        process_query(user_input)


if __name__ == "__main__":
    main()
