import os
import numpy as np
from nltk import word_tokenize
from tqdm import tqdm
from rusenttokenize import ru_sent_tokenize
import torch
from transformers import XLMRobertaTokenizer
from ..Utils import Utils
from ..Struct import Struct
from ..OntologyExtractor import OntologyExtractor
from ..Error import FileDataError, ConceptExtractionError, ConceptModelError, ArgumentError


class ConceptExtractor(OntologyExtractor):
    def __init__(self):
        super().__init__()
        self.module_path += '/ConceptExtractor'

    def load_model(self, concept_model: str):
        super().load_model(concept_model)
        if not isinstance(concept_model, str):
            raise ArgumentError("Argument 'concept_model' must be a str")
        if len(concept_model) == 0:
            raise ArgumentError("Argument 'concept_model' filename must not be empty")
        if not os.path.isfile(concept_model):
            raise ConceptModelError("Concept model file doesn't exist")
        print('Loading model for concept extraction...', end=' ')
        try:
            self.model = torch.load(concept_model, map_location=self.device)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        except:
            raise ConceptModelError('Error with loading concept model. '
                                    'Concept model should be PyTorch model creating for concept extraction.')
        print('OK')

    def run(self, data: str) -> Struct:
        super().run(data)
        if not isinstance(data, str):
            raise ArgumentError("Error data for concept extraction. "
                                "Argument 'data' must be a string")
        if len(data) == 0:
            raise ArgumentError("Argument 'data' string must not be empty")
        if os.path.isfile(data):
            if os.path.getsize(data) == 0:
                raise FileDataError("Argument 'data' is empty file")
            with open(data, 'r', encoding='utf-8') as file:
                data = file.read()
        data = Utils.clean(data.lower())
        sentences = ru_sent_tokenize(data)

        input_ids = []
        attention_masks = []
        for sentence in tqdm(sentences, desc="Preparing sentences for concept extraction"):
            encoded = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        if len(input_ids) == 0 or len(attention_masks) == 0:
            raise ConceptExtractionError(
                'Error occured during tokenization data for concept extraction. '
                'Please, check your data is not empty and contains russian text'
            )
        dataset = torch.utils.data.TensorDataset(
            torch.cat(input_ids, dim=0),
            torch.cat(attention_masks, dim=0)
        )
        sampler = torch.utils.data.SequentialSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=4)

        try:
            self.model.eval()
        except:
            raise ConceptModelError('Wrong concept extraction model. '
                                    'Please, load correct model for concept extraction.')
        keywords = []
        for batch in tqdm(loader, desc="Concept extraction"):
            predictions = []
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                result = self.model(batch[0], token_type_ids=None, attention_mask=batch[1])
            logits = result.logits.detach().cpu().numpy()

            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            tokens_pred = []

            for ipred, prediction in enumerate(predictions):
                idx = 0
                while idx < len(prediction):
                    label = prediction[idx]
                    if label == 0:
                        cur_tokens = [self.tokenizer.convert_ids_to_tokens(batch[0][ipred].to('cpu').numpy())[idx]]
                        idx += 1
                        while prediction[idx] != 2:
                            cur_tokens.append(
                                self.tokenizer.convert_ids_to_tokens(batch[0][ipred].to('cpu').numpy())[idx])
                            idx += 1
                        tokens_pred.append(cur_tokens)
                    idx += 1
                for token_group in tokens_pred:
                    keywords.append(self.tokenizer.convert_tokens_to_string(token_group))
        keywords = list(set(keywords))
        ikw = 0
        while ikw < len(keywords):
            keyword_words = word_tokenize(keywords[ikw], language="russian")
            if len(keyword_words) == 1:
                mph = self.morph.parse(keywords[ikw])[0]
                if not mph.is_known or mph.tag.POS != 'NOUN':
                    keywords.remove(keywords[ikw])
                    ikw -= 1
                else:
                    keywords[ikw] = mph.normal_form

            else:
                for keyword_word in keyword_words:
                    mph = self.morph.parse(keyword_word)[0]
                    if not mph.is_known or len(keyword_word) < 3:
                        keywords.remove(keywords[ikw])
                        ikw -= 1
                        break
            ikw += 1
        self.data.set_concepts(list(set(keywords)))
        return self.data
