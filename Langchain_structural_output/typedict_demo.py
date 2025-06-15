from typing import TypedDict

class Person(TypedDict):

    name:str
    age: int

new_Person: Person = {'name':'preet','age':24}

print(new_Person)