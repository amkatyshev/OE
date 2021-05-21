from src.OE import pipeline
from src.OE import ConceptExtractor
from src.OE import RelationExtractor

# # ce = ConceptExtractor()
# # ce.load_model('tests/concepts.pt')
# # concepts = ce.run('tests/test_complang.txt')
# re = RelationExtractor()
# re.load_model('tests/relations.pt')
# result = re.run([set(), 'tests/test_complang.txt'])

result = pipeline('tests/test_complang.txt', 'tests/concepts.pt', 'tests/relations.pt')
result.print()
