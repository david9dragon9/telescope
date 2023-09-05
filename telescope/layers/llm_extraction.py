from fastchat.model.model_adapter import get_conversation_template
import copy


def extract(query, text, llm, conversation):
    conversation.messages = []
    conversation.append_message(
        conversation.roles[0],
        f"Summarize the following text:\n{text}\n subject to the query {query}.",
    )
    conversation.append_message(conversation.roles[1], None)
    result = llm.respond(conversation.get_prompt())
    return result


class LLMExtractionLayer:
    def __init__(self, out_results, llms):
        self.out_results = out_results
        self.instruction_llm = llms["instruction"][1]
        self.conversation = get_conversation_template(llms["instruction"][0])
        self.conversation.system = "A user asks the assistant to summarize a given chunk of raw text scraped from a website, subject to a query. The assistant extracts the information from the text that is relevant to the query only and does not include any commentary in the summary or any outside information from the assistant's own knowledge."

    def search(self, query, input_results):
        assert self.out_results == len(
            input_results
        ), "Extraction does not change the number of results."
        for result in input_results.values():
            extracted_text = extract(
                query, result["text"], self.instruction_llm, self.conversation
            )
            result["text"] = extracted_text
        return input_results
