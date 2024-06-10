from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field

import instructor


class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


def test_get_strutured_output_ollama(self):
    slm = StructuredLangModel("llama3")
    response = slm.get_response(
        "Tell me about Harry Potter", "", response_model=Character
    )
    expected = """
{
"name": "Harry James Potter",
"age": 37,
"fact": [
"He is the chosen one.",
"He has a lightning-shaped scar on his forehead.",
"He is the son of James and Lily Potter.",
"He attended Hogwarts School of Witchcraft and Wizardry.",
"He is a skilled wizard and sorcerer.",
"He fought against Lord Voldemort and his followers.",
"He has a pet owl named Snowy."
]
}
"""

    self.assertIsInstance(response, Character)
    self.assertEqual("Harry Potter", response.name)
