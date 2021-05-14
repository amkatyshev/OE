import os
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import numpy as np
import torch
from rusenttokenize import ru_sent_tokenize
from src.OE.OntologyExtractor import OntologyExtractor
from transformers import PreTrainedModel, XLMRobertaModel, XLMRobertaConfig, XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, \
    XLMRobertaTokenizer


class FCLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(PreTrainedModel):
    base_model_prefix = "xlm-roberta"
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, config, args):
        super(RBERT, self).__init__(config)
        self.roberta = XLMRobertaModel(config=config)
        self.num_labels = config.num_labels
        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.e1_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.e2_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.label_classifier = FCLayer(config.hidden_size * 3, self.num_labels, 0.1, use_activation=False)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        e_mask_unqueeze = e_mask.unsqueeze(1)
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)

        sum_vector = torch.bmm(e_mask_unqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()
        return avg_vector

    def forward(self, input_ids, attention_mask, labels, e1_mask, e2_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)

        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs
        return outputs


class RelationExtractor(OntologyExtractor):
    def __init__(self):
        super().__init__()
        self.module_path += '/RelationExtractor'
        self.pairs = []

    def load_model(self, model=None):
        super().load_model(model)
        if model is None:
            model = self.module_path + '/default.pt'

        print('Loading model for relation extraction...', end=' ')
        args = {
            'NUM_LABELS': len(self.data.relation_labels),
            'DROPOUT_RATE': 0.1,
            'LEARNING_RATE': 2e-5,
            'EPOCHS': 5,
            'MAX_SEQUENCE_LENGTH': 384,
            'BATCH_SIZE': 16,
            'ADAM_EPSILON': 1e-8,
            'GRADIENT_ACCUMULATION_STEPS': 1,
            'MAX_GRAD_NORM': 1.0,
            'LOGGING_STEPS': 250,
            'SAVE_STEPS': 250,
            'WEIGHT_DECAY': 0.0,
            'NUM_WARMUP_STEPS': 0,
        }
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
        config = XLMRobertaConfig.from_pretrained('xlm-roberta-base', num_labels=args['NUM_LABELS'])
        self.model = RBERT.from_pretrained(model, config=config, args=args)
        self.model.to(self.device)
        print('OK')

    def run(self, data: list):
        super().run(data)
        if len(data) != 2 or not isinstance(data[0], set) or not isinstance(data[1], str):
            raise ValueError('Error data for relation extraction. '
                             'Input parameter must be a List with 2 values, '
                             'where first element is a set of keywords and '
                             'second element is a string with text or filename')
        nltk.download('punkt')
        self.data.set_concepts(list(data[0]))
        keywords = sorted(data[0], key=len, reverse=True)
        if os.path.isfile(data[1]):
            with open(data[1], 'r', encoding='utf-8') as file:
                data = file.read()
        else:
            data = data[1]
        data = data.lower()
        data = ru_sent_tokenize(data)

        for isentence, sentence in tqdm(enumerate(data), desc="Preparing sentences for relation extraction"):
            num_entity = 1
            sentence_words = word_tokenize(sentence, language="russian")
            sentence_words_normal = [self.morph.parse(sentence_word)[0].normal_form for sentence_word in sentence_words]
            for keyword in keywords:
                keyword_words = word_tokenize(str(keyword), language="russian")
                pos_start = None
                symbol = None
                if len(keyword_words) > 1:
                    index_start = sentence.find(str(keyword))
                    if index_start > -1:
                        pos_start = index_start
                        symbol = sentence[pos_start + len(keyword)]
                        pos_end = pos_start + len(keyword)
                else:
                    try:
                        index_start = sentence_words_normal.index(keyword)
                    except ValueError:
                        index_start = -1
                    if index_start > -1:
                        pos_start = sentence.find(sentence_words[index_start])
                        symbol = sentence[pos_start + len(sentence_words[index_start])]
                        pos_end = pos_start + len(sentence_words[index_start])
                if index_start > -1:
                    tag_pos = sentence.find('<', pos_start)
                    if not symbol.isalnum() and sentence[tag_pos+1] != '/':
                        # pos_end = pos_start + len(sentence_words[index_start])
                        data[isentence] = data[isentence][:pos_end] + ' </e' + str(num_entity) + '>' + data[isentence][
                                                                                                       pos_end:]
                        data[isentence] = data[isentence][:pos_start] + '<e' + str(num_entity) + '> ' + data[isentence][
                                                                                                        pos_start:]
                        sentence = data[isentence]
                        if num_entity == 1:
                            self.pairs.append({})
                            self.pairs[len(self.pairs) - 1]['e1'] = keyword
                            num_entity = 2
                        else:
                            self.pairs[len(self.pairs) - 1]['e2'] = keyword
                            break

        ipair = 0
        while ipair < len(self.pairs):
            if len(self.pairs[ipair]) < 2:
                del self.pairs[ipair]
                ipair -= 1
            ipair += 1
        dataset = self.__convert_lines(data)
        sampler = torch.utils.data.SequentialSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1)
        result = self.__evaluate(loader)
        for i, concepts in enumerate(self.pairs):
            self.data.add_relation(result[i], concepts['e1'], concepts['e2'])
        return self.data

    def __convert_lines(self, text, max_seq_len=128, cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                        add_sep_token=False, mask_padding_with_zero=True):
        input_ids = []
        attention_masks = []
        e1_masks = []
        e2_masks = []
        labels = []
        for sentence in text:
            tokens = self.tokenizer.tokenize(sentence.lower())

            try:
                e11_p = tokens.index("<e1>")
                e12_p = tokens.index("</e1>")
                e21_p = tokens.index("<e2>")
                e22_p = tokens.index("</e2>")
            except:
                continue

            # Replace token
            tokens[e11_p] = '$'
            tokens[e12_p] = '$'
            tokens[e21_p] = '#'
            tokens[e22_p] = '#'

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            # Account for [CLS] and [SEP] with "2" and with "3" for RoBERTa
            if add_sep_token:
                special_tokens_count = 2
            else:
                special_tokens_count = 1

            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[:(max_seq_len - special_tokens_count)]

            if add_sep_token:
                tokens += [sep_token]

            tokens = [cls_token] + tokens

            input_id = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_id)

            padding_length = max_seq_len - len(input_id)
            input_id = input_id + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                try:
                    e1_mask[i] = 1
                except:
                    break
            for i in range(e21_p, e22_p + 1):
                try:
                    e2_mask[i] = 1
                except:
                    break

            assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(0)
            e1_masks.append(e1_mask)
            e2_masks.append(e2_mask)

        dataset = torch.utils.data.TensorDataset(torch.tensor(input_ids, dtype=torch.long),
                                                 torch.tensor(attention_masks, dtype=torch.long),
                                                 torch.tensor(labels, dtype=torch.long),
                                                 torch.tensor(e1_masks, dtype=torch.long),
                                                 torch.tensor(e2_masks, dtype=torch.long))
        return dataset

    def __evaluate(self, loader):
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        self.model.eval()

        for batch in tqdm(loader, desc="Relation extraction"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2],
                          'e1_mask': batch[3],
                          'e2_mask': batch[4]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        # eval_loss = eval_loss / nb_eval_steps
        if preds is not None:
            preds = np.argmax(preds, axis=1)



        return preds