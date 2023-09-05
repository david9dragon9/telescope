from fastchat.model.model_adapter import get_conversation_template
import copy


class FinalSummaryLayer:
    def __init__(self, llms):
        self.instruction_llm = llms["instruction"][1]
        self.conversation = get_conversation_template(llms["instruction"][0])
        self.conversation.system = "A user asks the assistant to summarize information collected from various search results into a final, concise, organized summary. The assistant does not include any commentary in the summary or any outside information from the assistant's own knowledge."

    def search(self, query, input_results):
        final_prompt = "Summarize the following search results:\n"
        for result in input_results.values():
            final_prompt = f"{final_prompt}{result['title']}: {result['text']}\n"
        final_prompt = f"{final_prompt} subject to the query {query}. Only include information relevant to the query in your summary."
        self.conversation.messages = []
        self.conversation.append_message(self.conversation.roles[0], final_prompt)
        self.conversation.append_message(self.conversation.roles[1], None)
        return self.instruction_llm.respond(self.conversation.get_prompt())
