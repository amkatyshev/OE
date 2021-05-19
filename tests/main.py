from src.OE.RelationExtractor import RelationExtractor
from src.OE.ConceptExtractor import ConceptExtractor
from src.OE.OntologyExtractor import OntologyExtractor
from src.OE import pipeline


ce = ConceptExtractor()
ce.load_model('tests/concepts.pt')
concepts = ce.run('tests/test_computer.txt')
re = RelationExtractor()
re.load_model('tests/relations.pt')
result = re.run([concepts.get_concepts(as_type=set), 'tests/test_computer.txt'])
# with open('result.txt', 'w', encoding='utf-8') as file:
result.print()
