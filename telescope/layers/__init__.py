from telescope.layers.embeddings import EmbeddingSearchLayer
from telescope.layers.llm_extraction import LLMExtractionLayer
from telescope.layers.llm_filtering import LLMFilteringLayer
from telescope.layers.final_summary import FinalSummaryLayer


def get_layer(layer_name, out_results, llms):
    if layer_name == "embedding":
        return EmbeddingSearchLayer(out_results=out_results, llms=llms)
    elif layer_name == "extraction":
        return LLMExtractionLayer(out_results=out_results, llms=llms)
    elif layer_name == "filtering":
        return LLMFilteringLayer(out_results=out_results, llms=llms)
    elif layer_name == "summary":
        return FinalSummaryLayer(llms=llms)
    else:
        raise ValueError(f"Unsupported layer: {layer_name}")
