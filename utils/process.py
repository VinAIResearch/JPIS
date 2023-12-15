import time
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from utils.config import *
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW
from utils import miulab


class Processor(object):
    def __init__(self, dataset, model, args):
        self.dataset = dataset
        self.model = model
        self.batch_size = args.batch_size
        self.load_dir = args.load_dir
        self.args = args

        if self.args.gpu:
            time_start = time.time()
            self.model = self.model.cuda()
            time_con = time.time() - time_start
            mylogger.info("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.criterion = nn.NLLLoss()
        if self.args.use_pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_type_path)
            bert = list(map(id, self.model.encoder.parameters()))
            base_params = filter(lambda p: id(p) not in bert, self.model.parameters())
            self.optimizer = optim.Adam(
                base_params, lr=self.args.learning_rate,
                weight_decay=self.args.l2_penalty
            )

            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in self.model.encoder.named_parameters() if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01}
            ]
            self.optimizer_bert = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate)

        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.l2_penalty
            )

        if self.load_dir:
            if self.args.gpu:
                mylogger.info("MODEL {} LOADED".format(str(self.load_dir)))
                self.model = torch.load(os.path.join(self.load_dir, 'model.pkl'))
            else:
                mylogger.info("MODEL {} LOADED".format(str(self.load_dir)))
                self.model = torch.load(os.path.join(self.load_dir, 'model.pkl'), map_location=torch.device('cpu'))

    def tokenize_batch(self, word_batch):
        piece_batch = []
        piece_span = []

        for sent_i in range(0, len(word_batch)):
            piece_batch.append([self.tokenizer.cls_token_id])
            piece_span.append([])
            count = 0

            for word_i in range(0, len(word_batch[sent_i])):
                word = word_batch[sent_i][word_i]
                piece_list = self.tokenizer.convert_tokens_to_ids([word])
                piece_batch[-1].extend(piece_list)
                piece_span[-1].append(count)
                count += len(piece_list)

            piece_batch[-1].append(self.tokenizer.sep_token_id)
        return piece_batch, piece_span

    def padding_index(self, data):
        len_list = [len(piece) for piece in data]
        max_len = max(len_list)
        for index in range(len(data)):
            data[index].extend([self.tokenizer.pad_token_id] * (max_len - len_list[index]))
        return data

    def padding_text(self, data):
        data, span = self.tokenize_batch(data)
        data = self.padding_index(data)
        return data, span

    def trans_up_ca(self, sorted_up, sorted_ca):
        up_var, ca_var = [], []
        len_up, len_ca = [3, 3, 3, 2], [8, 3, 4, 3]
        up_var, ca_var = [[] for _ in len_up], [[] for _ in len_ca]
        for item_up, item_ca in zip(sorted_up, sorted_ca):
            for i, (u, c) in enumerate(zip(item_up, item_ca)):
                u = self._cuda(torch.FloatTensor(u))
                c = self._cuda(torch.FloatTensor(c))
                up_var[i].append(u)
                ca_var[i].append(c)
        for i, (up, ca) in enumerate(zip(up_var, ca_var)):
            up_var[i] = torch.stack(up)
            ca_var[i] = torch.stack(ca)
        return up_var, ca_var

    def _cuda(self, x):
        if self.args.gpu:
            return x.cuda()
        else:
            return x

    def train(self):
        best_dev_sent = 0.0
        no_improve, step = 0, 0
        dataloader = self.dataset.batch_delivery('train')
        for epoch in range(0, self.dataset.num_epoch):

            time_start = time.time()
            self.model.train()

            for text_batch, slot_batch, intent_batch, kg_batch, up_batch, ca_batch, type_batch in tqdm(dataloader, ncols=50):
                padded_text, padded_kg, [sorted_slot, sorted_intent, sorted_up, sorted_ca, sorted_type], \
                seq_lens, kg_lens, kg_count = self.dataset.add_padding(text_batch, kg_batch,
                                                                       [(slot_batch, True),
                                                                        (intent_batch, False),
                                                                        (up_batch, False),
                                                                        (ca_batch, False),
                                                                        (type_batch, False)],
                                                                       use_pretrained=self.args.use_pretrained,
                                                                       split_index=
                                                                       self.dataset.word_alphabet.get_index(['；'])[0],
                                                                       max_length=self.args.max_length)
                if self.args.use_pretrained:
                    padded_text, span = self.padding_text(padded_text)
                text_var = self._cuda(torch.LongTensor(padded_text))
                if self.args.use_pretrained:
                    text_var = (text_var, span)
                kg_var = self._cuda(torch.LongTensor(padded_kg))
                slot_var = self._cuda(torch.LongTensor(sorted_slot))
                type_var = self._cuda(torch.LongTensor(sorted_type))
                intent_var = self._cuda(torch.LongTensor(sorted_intent)).squeeze(-1)
                up_var, ca_var = self.trans_up_ca(sorted_up, sorted_ca)

                slot_out, intent_out, loss = self.model(text_var, seq_lens, kg_var, kg_lens, kg_count, up_var, ca_var, intent_var, slot_var, type_var)
                step += 1

                self.optimizer.zero_grad()
                if self.args.use_pretrained:
                    self.optimizer_bert.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.args.use_pretrained:
                    self.optimizer_bert.step()
            time_con = time.time() - time_start


            time_start = time.time()
            dev_f1_score, dev_acc, dev_sent_acc = self.estimate("dev")

            if dev_f1_score + dev_acc + dev_sent_acc > best_dev_sent:
                no_improve = 0
                best_dev_sent = dev_acc + dev_f1_score + dev_sent_acc
                test_f1, test_acc, test_sent_acc = self.estimate("test")
                
                mylogger.info('\nTest result: slot f1 score: {:.6f}, intent acc score: {:.6f}, semantic '
                              'accuracy score: {:.6f}.'.format(test_f1, test_acc, test_sent_acc))

                model_save_dir = os.path.join(self.dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                torch.save(self.model, os.path.join(model_save_dir, "model.pkl"))
                torch.save(self.dataset, os.path.join(model_save_dir, 'dataset.pkl'))

                time_con = time.time() - time_start
                mylogger.info('[Epoch {:2d}]: In validation process, the slot f1 score is {:2.6f}, ' \
                              'the intent acc is {:2.6f}, the semantic acc is {:.2f}, cost about ' \
                              '{:2.6f} seconds.\n'.format(epoch, dev_f1_score, dev_acc, dev_sent_acc, time_con))
            else:
                no_improve += 1

            if self.args.early_stop:
                if no_improve > self.args.patience:
                    mylogger.info('early stop at epoch {}'.format(epoch))
                    break
    
    def write_evaluation_result(self, out_file, results):
        out_file = self.args.save_dir + "/" + out_file
        w = open(out_file, "w", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()

    def estimate(self, mode):
        """
        Estimate the performance of model on dev or test dataset.
        """
        with torch.no_grad():
            if mode == "dev":
                pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction("dev")
                file_results = "eval_dev_results.txt"
            elif mode == "test":
                pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction("test")
                file_results = "eval_test_results.txt"
            else:
                pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction("train")
        slot_f1_socre, slot_p, slot_r = miulab.computeF1Score(real_slot, pred_slot)
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
        self.write_evaluation_result(file_results, {
            "intent_acc": intent_acc,
            "sentence_acc": sent_acc,
            "slot_f1": slot_f1_socre,
            "slot_precision": slot_p,
            "slot_recall": slot_r
        })
        return slot_f1_socre, intent_acc, sent_acc

    def validate(self, model_path, dataset_path):
        """
        validation will write mistaken samples to files and make scores.
        """

        if self.args.gpu:
            self.model = torch.load(model_path)
            self.dataset = torch.load(dataset_path)
        else:
            self.model = torch.load(model_path, map_location=torch.device('cpu'))
            self.dataset = torch.load(dataset_path, map_location=torch.device('cpu'))

        self.dataset.quick_build_test(self.args.data_dir, 'test.json')
        mylogger.info('load {} to test'.format(self.args.data_dir))

        # Get the sentence list in test dataset.
        # sent_list = dataset.test_sentence
        with torch.no_grad():
            pred_slot, real_slot, pred_intent, real_intent, sent_list = self.prediction("test")

        slot_f1, slot_p, slot_r = miulab.computeF1Score(real_slot, pred_slot)
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        # To make sure the directory for save error prediction.
        mistake_dir = os.path.join(self.args.save_dir, "error")
        if not os.path.exists(mistake_dir):
            os.mkdir(mistake_dir)

        slot_file_path = os.path.join(mistake_dir, "slot.txt")
        intent_file_path = os.path.join(mistake_dir, "intent.txt")
        both_file_path = os.path.join(mistake_dir, "both.txt")

        # Write those sample with mistaken slot prediction.
        with open(slot_file_path, 'w') as fw:
            for w_list, r_slot_list, p_slot_list in zip(sent_list, real_slot, pred_slot):
                if r_slot_list != p_slot_list:
                    for w, r, p in zip(w_list, r_slot_list, p_slot_list):
                        fw.write(w + '\t' + r + '\t' + p + '\n')
                    fw.write('\n')

        # Write those sample with mistaken intent prediction.
        with open(intent_file_path, 'w') as fw:
            for w_list, p_intent, r_intent in zip(sent_list, pred_intent, real_intent):
                if p_intent != r_intent:
                    for w in w_list:
                        fw.write(w + '\n')
                    fw.write(r_intent + '\t' + p_intent + '\n\n')

        # Write those sample both have intent and slot errors.
        with open(both_file_path, 'w') as fw:
            for w_list, r_slot_list, p_slot_list, p_intent, r_intent in \
                    zip(sent_list, real_slot, pred_slot, pred_intent, real_intent):

                if r_slot_list != p_slot_list or r_intent != p_intent:
                    for w, r_slot, p_slot in zip(w_list, r_slot_list, p_slot_list):
                        fw.write(w + '\t' + r_slot + '\t' + p_slot + '\n')
                    fw.write(r_intent + '\t' + p_intent + '\n\n')

        return slot_f1, intent_acc, sent_acc

    def prediction(self, mode):
        self.model.eval()

        if mode == "dev":
            dataloader = self.dataset.batch_delivery('dev', batch_size=self.args.batch_size,
                                                     shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = self.dataset.batch_delivery('test', batch_size=self.args.batch_size,
                                                     shuffle=False, is_digital=False)
        else:
            dataloader = self.dataset.batch_delivery('train', batch_size=self.args.batch_size,
                                                     shuffle=False, is_digital=False)

        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []
        sent_list = []

        for text_batch, slot_batch, intent_batch, kg_batch, up_batch, ca_batch, type_batch in tqdm(dataloader, ncols=50):
            padded_text, padded_kg, [sorted_slot, sorted_intent, sorted_up, sorted_ca, sorted_type], \
            seq_lens, kg_lens, kg_count = self.dataset.add_padding(text_batch, kg_batch,
                                                                   [(slot_batch, False),
                                                                    (intent_batch, False),
                                                                    (up_batch, False),
                                                                    (ca_batch, False),
                                                                    (type_batch, False)],
                                                                   digital=False,
                                                                   use_pretrained=self.args.use_pretrained,
                                                                   split_index=
                                                                   self.dataset.word_alphabet.get_index(['；'])[0],
                                                                   max_length=self.args.max_length)
            real_slot.extend(sorted_slot)
            real_intent.extend(list(Evaluator.expand_list(sorted_intent)))
            if self.args.use_pretrained:
                sent_list.extend(padded_text)
                padded_text, span = self.padding_text(padded_text)
                var_text = self._cuda(torch.LongTensor(padded_text))
                var_text = (var_text, span)
            else:
                sent_list.extend([pt[:seq_lens[idx]] for idx, pt in enumerate(padded_text)])
                digit_text = self.dataset.word_alphabet.get_index(padded_text)
                var_text = self._cuda(torch.LongTensor(digit_text))

            digit_kg = self.dataset.word_alphabet.get_index(padded_kg)
            var_kg = self._cuda(torch.LongTensor(digit_kg))
            var_up, var_ca = self.trans_up_ca(sorted_up, sorted_ca)

            slot_logits, intent_logits, _ = self.model(var_text, seq_lens, var_kg, kg_lens, kg_count, var_up, var_ca)

            if self.args.use_crf:
                slot_idx = np.array(self.model.crf.decode(slot_logits)).tolist()
            else:
                slot_idx = np.argmax(slot_logits.detach().cpu().numpy(), axis=2).tolist()
            intent_idx = np.argmax(intent_logits.detach().cpu().numpy(), axis=1).tolist()

            # nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]
            for nested_slot, seq_len in zip(slot_idx, seq_lens):
                pred_slot.append(self.dataset.slot_alphabet.get_instance(nested_slot[:seq_len]))
            pred_intent.extend(self.dataset.intent_alphabet.get_instance(intent_idx))
        return pred_slot, real_slot, pred_intent, real_intent, sent_list


class Evaluator(object):

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """

        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
