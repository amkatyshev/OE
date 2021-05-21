import json
import os
import sys
from typing import TextIO
import pandas as pd
from .Error import ArgumentError

labels = [
    "Другое",
    "Причина-Следствие(e1,e2)",
    "Причина-Следствие(e2,e1)",
    "Инструмент-Агент(e1,e2)",
    "Инструмент-Агент(e2,e1)",
    "Продукт-Производитель(e1,e2)",
    "Продукт-Производитель(e2,e1)",
    "Контент-Контейнер(e1,e2)",
    "Контент-Контейнер(e2,e1)",
    "Сущность-Источник(e1,e2)",
    "Сущность-Источник(e2,e1)",
    "Сущность-Цель(e1,e2)",
    "Сущность-Цель(e2,e1)",
    "Часть-Целое(e1,e2)",
    "Часть-Целое(e2,e1)",
    "Элемент-Коллекция(e1,e2)",
    "Элемент-Коллекция(e2,e1)",
    "Сообщение-Тема(e1,e2)",
    "Сообщение-Тема(e2,e1)",
]


class Struct:
    def __init__(self):
        self.concepts = []
        self.relation_labels = pd.DataFrame({'label': labels})
        self.relations = dict()

    def set_concepts(self, concepts: list):
        self.concepts = concepts

    def get_concepts(self, as_type=list):
        if as_type is not list and as_type is not set:
            raise ArgumentError("Argument 'as_type' must be list or set")
        return as_type(self.concepts)

    def add_relation(self, relation_id: int, e1: str, e2: str):
        if relation_id != 0 and e1 in self.concepts and e2 in self.concepts:
            pair = {'e1': self.concepts.index(e1), 'e2': self.concepts.index(e2)}
            if relation_id in self.relations:
                if pair not in self.relations[relation_id]:
                    self.relations[relation_id].append(pair)
            else:
                self.relations[relation_id] = [pair]

    def print(self):
        if len(self.concepts) > 0:
            print('Concepts:')
            for concept in self.concepts:
                print(concept)

        if len(self.relations) > 0:
            print('\nRelations:')
            for relation_id, concepts in self.relations.items():
                print(self.relation_labels.iloc[relation_id]['label'] + ':')
                for relation in self.relations[relation_id]:
                    print("\te1: " + self.concepts[relation['e1']] + "; e2: " + self.concepts[relation['e2']])

    def to_file(self, filename: str):
        if not isinstance(filename, str):
            raise ArgumentError("Argument 'filename' must be a str")
        result = {}
        if len(self.concepts) > 0:
            result['concepts'] = self.concepts
        if len(self.relations) > 0:
            result['relations'] = {}
            for relation_id, concepts in self.relations.items():
                label = self.relation_labels.iloc[relation_id]['label']
                result['relations'][label] = []
                for relation in self.relations[relation_id]:
                    result['relations'][label].append(
                        {"e1": self.concepts[relation['e1']], "e2": self.concepts[relation['e2']]}
                    )
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False)

