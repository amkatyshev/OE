from src.OE import pipeline
from src.OE import ConceptExtractor
from src.OE import RelationExtractor

ce = ConceptExtractor()
ce.load_model('tests/concepts.pt')
concepts = ce.run('tests/test_computer.txt')
re = RelationExtractor()
re.load_model('tests/relations.pt')
result = re.run([concepts.get_concepts(as_type=set), 'tests/test_computer.txt'])
# with open('result.txt', 'w', encoding='utf-8') as file:
result.print()
