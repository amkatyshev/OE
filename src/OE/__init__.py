from .ConceptExtractor.ConceptExtractor import ConceptExtractor
from .RelationExtractor.RelationExtractor import RelationExtractor


def pipeline(data: str):
    ce = ConceptExtractor()
    re = RelationExtractor()
    ce.load_model()
    re.load_model()
    concepts = ce.run(data)
    result = re.run([concepts.get_concepts(set), data])
    return result

