# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Any, List, Optional, Tuple


def partial_format(input_str: str, **kwargs: Any) -> str:
    formatted_str = input_str
    for key, value in kwargs.items():
        formatted_str = formatted_str.replace(f"{{{key}}}", str(value))
    return formatted_str


def remove_spaces_in_name(messages: List[dict]) -> List[dict]:
    for message in messages:
        if "name" in message:
            message["name"] = message["name"].replace(" ", "_")
    return messages


def construct_summary_message(summary: str, prompts: dict) -> str:

    # Use introspection in the case in which there are no other agents.
    if not summary:
        return (
            "Can you verify that your answer is correct. Please reiterate your answer."
        )

    prompt = prompts["summary_prefix_seperator"] + summary + prompts["suffix_seperator"]
    return prompt


def construct_message(
    agents: List[List[str]], question: str, prompts: dict, summary_mode: bool = False
) -> str:

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return (
            "Can you verify that your answer is correct. Please reiterate your answer."
        )

    other_agent_responses = ""
    for agent_response in agents:
        agent_response = remove_question(agent_response[-1], question)  # type: ignore
        other_agent_responses += prompts["agent_response"].format(agent_response)

    prompt = prompts["prefix_seperator"] + other_agent_responses

    if summary_mode:
        prompt += prompts["summary_suffix_seperator"]
    else:
        prompt += prompts["suffix_seperator"]

    return prompt


def construct_eot_message(
    question: str,
    my_solution: str,
    peer_names: List[str],
    peer_responses: List[str],
    confidences: Optional[List[float]] = None,
    format_hint: str = "",
) -> str:
    """Construct a prompt following the official Exchange-of-Thought format."""

    if confidences is None:
        confidences = [-1.0 for _ in peer_names]

    # Support up to two peers as per the reference implementation
    peer_names = peer_names[:2]
    peer_responses = peer_responses[:2]
    confidences = confidences[:2]

    if len(peer_names) == 0:
        return (
            f"Please consider the example provided and think it step by step.\n"
            f"Question: {question}"
        )

    participant1 = peer_names[0]
    response1 = peer_responses[0]
    confidence1 = confidences[0]
    participant2 = peer_names[1] if len(peer_names) > 1 else ""
    response2 = peer_responses[1] if len(peer_names) > 1 else ""
    confidence2 = confidences[1] if len(peer_names) > 1 else -1

    if confidence1 == -1 and participant2 == "":
        query = (
            "Please consider the example provided and think it step by step.\n"
            f"Question: {question}\n"
            f"Your Solution: {my_solution}\n"
            "Here is a solution process from your friend:\n"
            f"{participant1}'s Solution: {response1}\n"
            "Based on your friend's solution, carefully re-examine your previous answer. Utilize your talent and critical thinking to provide a new step-by-step solution process.\n"
            f"Provide the new solution directly, refrain from commenting on your friend's approach, and conclude by stating, \"the answer is{format_hint}.\""
        )

    elif confidence1 == -1 and participant2 != "":
        query = (
            "Please consider the example provided and think it step by step.\n"
            f"Question: {question}\n"
            f"Your Solution: {my_solution}\n"
            "Here is a solution process from your friend:\n"
            f"{participant1}'s Solution: {response1}\n"
            f"{participant2}'s Solution: {response2}\n"
            "Based on your friend's solution, carefully re-examine your previous answer. Utilize your talent and critical thinking to provide a new step-by-step solution process.\n"
            f"Provide the new solution directly, refrain from commenting on your friend's approach, and conclude by stating, \"the answer is{format_hint}.\""
        )

    elif confidence1 != -1 and participant2 == "":
        query = (
            "Please consider the example provided and think it step by step.\n"
            f"Question: {question}\n"
            f"Your Solution: {my_solution}\n"
            "Here is a solution process from your friend:\n"
            f"{participant1}'s Solution: {response1}\n"
            f"{participant1}'s confidence in this solution is: {confidence1}\n"
            "Based on your friend's solution, carefully re-examine your previous answer. If your friend's confidence level is below 0.5, it suggests a high probability that the solution might be incorrect. Remember, solutions with high confidence can also be wrong. Utilize your talent and critical thinking to provide a new step-by-step solution process.\n"
            f"Provide the new solution directly, refrain from commenting on your friend's approach, and conclude by stating, \"the answer is{format_hint}.\""
        )

    else:
        query = (
            "Please consider the example provided and think it step by step.\n"
            f"Question: {question}\n"
            f"Your Solution: {my_solution}\n"
            "Here is a solution process from your friend:\n"
            f"{participant1}'s Solution: {response1}\n"
            f"{participant1}'s confidence in this solution is: {confidence1}\n"
            f"{participant2}'s Solution: {response2}\n"
            f"{participant2}'s confidence in this solution is: {confidence2}\n"
            "Based on your friend's solution, carefully re-examine your previous answer. If your friend's confidence level is below 0.5, it suggests a high probability that the solution might be incorrect. Remember, solutions with high confidence can also be wrong. Utilize your talent and critical thinking to provide a new step-by-step solution process.\n"
            f"Provide the new solution directly, refrain from commenting on your friend's approach, and conclude by stating, \"the answer is{format_hint}.\""
        )

    return query


def construct_message_from_history(
    message_history: List[dict],
    agent_name: Optional[str] = "",
    mode: Optional[str] = "assistant_list",
) -> List[dict]:

    if mode == "assistant_list":
        # default mode: history is given as a list of assistant messages
        messages = []
        for message in message_history:
            messages.append(
                {
                    "role": "assistant",
                    "name": message["agent_name"],
                    "content": message["content"],
                }
            )
        return messages

    elif mode == "one_prompt":
        # history is aggregated into a single user prompt
        user_message = ""
        for message in message_history:
            user_message += (
                "\n\n" + message["agent_name"] + " arguing: " + message["content"]
            )
        user_message = user_message.strip("\n")
        return [
            {
                "role": "user",
                "name": "user",
                "content": user_message,
            }
        ]

    elif mode == "tsinghua_judge":
        # combines one_prompt for each round, separated by previous Judge messages
        current_segment = []
        result = []

        for message in message_history:
            if message["agent_name"] != "Judge":
                current_segment.append(message)
            else:
                result += construct_message_from_history(
                    current_segment, mode="one_prompt"
                )
                result.append(
                    {
                        "role": "assistant",
                        "name": agent_name,
                        "content": message["content"],
                    }
                )
                current_segment = []

        result += construct_message_from_history(current_segment, mode="one_prompt")

        return result

    elif mode == "tsinghua_mad":
        # the agent messages are passed as assistant messages
        # and other agent messages are passed as user messages
        messages = []
        user_message = ""

        two_agent_debate = (
            len(
                {
                    message["agent_name"]
                    for message in message_history
                    if message["agent_name"] != "Judge"
                }
            )
            <= 2
        )

        for message in message_history:

            if message["agent_name"] == "Judge":
                pass

            elif message["agent_name"] == agent_name:

                messages.append(
                    {
                        "role": "user",
                        "name": "user",
                        "content": user_message.strip("\n"),
                    }
                )

                messages.append(
                    {
                        "role": "assistant",
                        "name": message["agent_name"],
                        "content": message["content"],
                    }
                )
                user_message = ""
            else:
                prefix = (
                    ""
                    if two_agent_debate
                    else "\n\n" + message["agent_name"] + " arguing: "
                )
                user_message += prefix + message["content"]

        messages.append(
            {
                "role": "user",
                "name": "user",
                "content": user_message.strip("\n"),
            }
        )
        return messages
    else:
        raise ValueError("Invalid mode")


def remove_question(string: str, question: str) -> str:
    pattern = f"(?=({re.escape(question)}))"
    matches = re.findall(pattern, string)

    for match in matches:
        string = string.replace(match, "", 1)

    return string


def most_frequent(list: List[str]) -> Tuple[str, int]:
    counter = 0
    num = list[0]

    for i in list:
        current_frequency = list.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter
