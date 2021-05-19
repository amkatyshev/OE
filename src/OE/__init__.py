from .ConceptExtractor.ConceptExtractor import ConceptExtractor
from .RelationExtractor.RelationExtractor import RelationExtractor


def pipeline(data: str, concept_model: str, relation_model: str):
    ce = ConceptExtractor()
    re = RelationExtractor()
    ce.load_model(concept_model)
    re.load_model(relation_model)
    concepts = ce.run(data)
    result = re.run([concepts.get_concepts(set), data])
    return result

