from src.OE.RelationExtractor import RelationExtractor
from src.OE.ConceptExtractor import ConceptExtractor
from src.OE.OntologyExtractor import OntologyExtractor
from src.OE import pipeline


result = pipeline('test.txt')
# with open('result.txt', 'w', encoding='utf-8') as file:
result.print()
