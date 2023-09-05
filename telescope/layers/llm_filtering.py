from fastchat.model.model_adapter import get_conversation_template
import random
import copy

fs_examples = [
    [
        "Here are two search results:\n\n\n\n\n\nResult (A): 'Cats & Dogs' uncovers the truth about the high-tech, secret war being waged in neighborhoods everywhere that humans aren't even aware of: an eternal struggle between the two great armies of Cats and Dogs. The story follows a Cat plan to destroy a new vaccine that, if developed, would destroy all hu.\n\n\n\n\nResult (B): Generally, cat colours, which can be termed differently depending on the breed, are: white, red, blue, black, cream, cinnamon, fawn and brown. The colour of a cat depends completely on genetics. The two primary colours in cats are black and red. \n\n\n\n\n Which search result is more relevant to the search query: fur color of cats? Place (A) anywhere in your answer to indicate that Result (A) is more relevant, and (B) anywhere in your answer to indicate that Result (B) is more relevant. Your last answer will be your final answer.",
        "Result (B) is more relevant to the search query as it discusses the fur color of cats in detail.",
    ]
]


def compare(result1, result2, query, llm, conversation):
    conversation.messages = []
    for fse in fs_examples:
        conversation.append_message(conversation.roles[0], fse[0])
        conversation.append_message(conversation.roles[1], fse[1])
    final_prompt = f"Here are two search results:\n\n\n\n\n\nResult (A): {result1['text']}\n\n\n\n\n\nResult (B): {result2['text']}\n\n\n\n\n Which search result is more relevant to the search query: {query}? Place (A) anywhere in your answer to indicate that Result (A) is more relevant, and (B) anywhere in your answer to indicate that Result (B) is more relevant. Your last answer will be your final answer."
    conversation.append_message(conversation.roles[0], final_prompt)
    conversation.append_message(conversation.roles[1], None)
    # import pdb; pdb.set_trace()
    result = llm.respond(conversation.get_prompt())
    if result.find("(A)") == -1 and result.find("(B)") == -1:
        return random.randint(0, 1)
    elif result.find("(A)") == -1:
        return 1
    elif result.find("(B)") == -1:
        return 0
    else:
        return int(result.find("(B)") > result.find("(A)"))


def swap(results, x, y):
    temp = copy.deepcopy(results[x])
    results[x] = copy.deepcopy(results[y])
    results[y] = temp


def rank(results, k, start, end, query, llm, conversation):
    print(start, end)
    pivot = copy.deepcopy(results[random.randint(start, end)])
    left_edge = partition(results, start, end, pivot, query, llm, conversation)
    left_size = left_edge - start + 1
    if k == left_size - 1:
        return left_edge
    elif k < left_size:
        return rank(results, k, start, left_edge, query, llm, conversation)
    else:
        return rank(
            results, k - left_size, left_edge + 1, end, query, llm, conversation
        )


def partition(results, start, end, pivot, query, llm, conversation):
    while start <= end:
        if compare(pivot, results[start], query, llm, conversation) == 0:
            swap(results, start, end)
            end -= 1
        elif compare(pivot, results[end], query, llm, conversation) == 1:
            swap(results, start, end)
            start += 1
        else:
            end -= 1
            start += 1
    return start - 1


class LLMFilteringLayer:
    def __init__(self, out_results, llms):
        self.out_results = out_results
        self.instruction_llm = llms["instruction"][1]
        self.conversation = get_conversation_template(llms["instruction"][0])
        self.conversation.system = "A user asks the assistant to determine which of two search results are more relevant to a given search query, by saying (A) to indicate result (A) and (B) to indicate result (B). The assistant does not include any commentary in the summary or any outside information from the assistant's own knowledge."

    def search(self, query, input_results):
        results_list = [x for x in input_results.values()]
        rank(
            results_list,
            self.out_results - 1,
            0,
            len(results_list) - 1,
            query,
            self.instruction_llm,
            self.conversation,
        )
        top_k_results = {x["id"]: x for x in results_list[: self.out_results]}
        return top_k_results
