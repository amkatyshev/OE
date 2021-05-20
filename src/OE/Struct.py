import os
import sys
from typing import TextIO
import pandas as pd


class Struct:
    def __init__(self):
        self.concepts = []
        self.relation_labels = pd.read_csv(
            os.path.dirname(os.path.abspath(__file__)) + '/RelationExtractor/labels.txt', sep="\t", header=None, names=['label']
        )
        self.relations = dict()

    def set_concepts(self, concepts: list):
        self.concepts = concepts

    def get_concepts(self, as_type=list):
        if as_type is not list and as_type is not set:
            raise ValueError('as_type parameter must be list or set')
        return as_type(self.concepts)

    def add_relation(self, relation_id: int, e1: str, e2: str):
        if relation_id != 0 and e1 in self.concepts and e2 in self.concepts:
            pair = {'e1': self.concepts.index(e1), 'e2': self.concepts.index(e2)}
            if relation_id in self.relations:
                if pair not in self.relations[relation_id]:
                    self.relations[relation_id].append(pair)
            else:
                self.relations[relation_id] = [pair]

    def print(self, output=sys.stdout):
        if not isinstance(output, type(sys.stdout)):
            raise ValueError('output parameter must be TextIO type')
        sys.stdout = output

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
