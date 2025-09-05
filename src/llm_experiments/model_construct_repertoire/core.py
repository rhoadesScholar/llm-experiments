class Context:
    """
    The class bookkeeps for the context of a conversation.
    """

    def __init__(
        self,
        context_str: str | None = None,
        embodied: bool = False,
        AI_assistant: bool = False,
        **kwargs,
    ):
        self.context_str = context_str
        self.is_embodied = embodied
        self.is_AI_assistant = AI_assistant
        self.is_null = context_str is None
        self.__dict__.update(kwargs)

    def __repr__(self):
        return (
            f"Context({self.context_str}, embodied={self.is_embodied}, "
            f"AI_assistant={self.is_AI_assistant})"
        )

    def __str__(self):
        return self.context_str or ""


class Conversation:
    """
    This class keeps track of all exchanges in a conversation, and formulates
    prompts for the LLM.
    """

    def __init__(self, context: Context, is_telephone: bool = False):
        self.context = context
        self.is_telephone = is_telephone
        self.exchanges = []

    def add_exchange(self, user_input: str, bot_response: str):
        self.exchanges.append((user_input, bot_response))

    def __repr__(self):
        return f"Conversation({self.exchanges})"

    def __str__(self):
        if self.is_telephone:
            return self.exchanges[-1][-1]  # Only return the last bot response
        else:
            return "\n".join(
                f"User: {user}\nAI assistant: {bot}" for user, bot in self.exchanges
            )

    def formulate_prompt(self) -> str:
        """
        Formulates a prompt for the LLM based on the conversation history and
        context.
        """
        prompt = str(self.context) + "\n"
        if not self.is_telephone:
            for user, bot in self.exchanges:
                prompt += f"\nUser: {user}\nBot: {bot}"

        return prompt
