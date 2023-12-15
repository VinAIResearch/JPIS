import random
from models.module import *
from torchcrf import CRF
from models.attention import *


class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent, num_slot_type):
        super(ModelManager, self).__init__()

        self.num_word = num_word
        self.num_slot = num_slot
        self.num_intent = num_intent
        self.num_type = num_slot_type
        self.args = args
        len_up, len_ca = [3, 3, 3, 2], [8, 3, 4, 3]

        self.embedding = nn.Embedding(
            self.num_word,
            self.args.word_embedding_dim
        )
        # Fusion Setting
        if args.use_info:
            self.hierarchical_fusion = Hierarchical_Fusion(
                self.args.encoder_hidden_dim if args.use_pretrained else self.args.encoder_hidden_dim + self.args.attention_output_dim,
                self.args.info_embedding_dim,
                self.args.dropout_rate
            )

        # Initialize an embedding object.
        if args.use_pretrained:
            self.encoder = BertEncoder(self.args.bert_dropout_rate)

            intent_info = self.args.info_embedding_dim if self.args.use_info else 0
            self.intent_decoder = nn.Linear(
                self.encoder.encoder_dim + intent_info, self.num_intent
            )
            intent_emb = 0
            if self.args.s2i:
                self.intent_label = LabelAttention(self.encoder.encoder_dim + intent_info, self.args.d_a, self.num_intent)
                self.slot_label = LabelAttention(self.encoder.encoder_dim + intent_info, self.args.d_a, self.num_type)
                self.s2i = SlotToIntent(self.encoder.encoder_dim + intent_info, self.encoder.encoder_dim + intent_info, self.args.d_c, self.args.dropout_rate)
            if self.args.i2s:
                self.soft_emb = nn.Linear(self.num_intent, self.args.intent_emb, bias=False)
                intent_emb = self.args.intent_emb
            self.slot_decoder = SlotClassifier(
                self.encoder.encoder_dim + intent_info + intent_emb,
                self.num_slot,
                dropout_rate=self.args.dropout_rate
            )

        else:
            # Initialize an LSTM Encoder object.
            self.encoder = LSTMEncoder(
                self.args.word_embedding_dim,
                self.args.encoder_hidden_dim,
                self.args.dropout_rate
            )

            # Initialize a self-attention layer.
            self.attention = SelfAttention(
                self.args.word_embedding_dim,
                self.args.attention_hidden_dim,
                self.args.attention_output_dim,
                self.args.dropout_rate
            )

            intent_info = self.args.info_embedding_dim if self.args.use_info else 0
            self.intent_decoder = nn.Linear(
                self.args.encoder_hidden_dim + self.args.attention_output_dim + intent_info, self.num_intent
            )
            intent_emb = 0
            if self.args.s2i:
                self.intent_label = LabelAttention(self.args.encoder_hidden_dim + self.args.attention_output_dim + intent_info, self.args.d_a, self.num_intent)
                self.slot_label = LabelAttention(self.args.encoder_hidden_dim + self.args.attention_output_dim + intent_info, self.args.d_a, self.num_type)
                self.s2i = SlotToIntent(self.args.encoder_hidden_dim + self.args.attention_output_dim + intent_info, self.args.encoder_hidden_dim + self.args.attention_output_dim + intent_info, self.args.d_c, self.args.dropout_rate)
            else:
                self.sentattention = UnflatSelfAttention(
                    self.args.encoder_hidden_dim + self.args.attention_output_dim,
                    self.args.dropout_rate
                )
            if self.args.i2s:
                self.soft_emb = nn.Linear(self.num_intent, self.args.intent_emb, bias=False)
                intent_emb = self.args.intent_emb
            self.slot_decoder = SlotClassifier(
                self.args.encoder_hidden_dim + self.args.attention_output_dim + intent_info + intent_emb,
                self.num_slot,
                dropout_rate=self.args.dropout_rate
            )
        
        if self.args.up:
            # self.encoder_up = nn.ModuleList([
            #     nn.Linear(up_size, self.args.info_embedding_dim)
            #     for up_size in len_up
            # ])
            self.encoder_up = nn.Linear(sum(len_up), self.args.info_embedding_dim)
        if self.args.ca:
            # self.encoder_ca = nn.ModuleList([
            #     nn.Linear(ca_size, self.args.info_embedding_dim)
            #     for ca_size in len_ca
            # ])
            self.encoder_ca = nn.Linear(sum(len_ca), self.args.info_embedding_dim)
        if self.args.kg:
            self.encoder_kg = LSTMEncoder(
                self.args.word_embedding_dim,
                self.args.info_embedding_dim,
                self.args.dropout_rate
            )
        self.softmax = nn.Softmax(dim=-1)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot, batch_first=True)

    def _cuda(self, x):
        if self.args.gpu:
            return x.cuda()
        else:
            return x

    def match_token(self, hiddens, span):
        # take the first subword hidden as the represenation for each token in the utterance
        hiddens_span = self._cuda(torch.zeros_like(hiddens))
        for i in range(len(span)):
            for idx, span_i in enumerate(span[i]):
                hiddens_span[i][idx] = hiddens[i][span_i]
        return hiddens_span
    
    def get_kg(self, hiddens, lens):
        # take the last hidden as the represenation
        embs = []
        for idx, length in enumerate(lens):
            embs.append(hiddens[idx, length - 1])
        embs = torch.stack(embs)
        return embs

    def match_kg(self, hiddens, count):
        # average the entities for each sample as KG represenation
        max_c = max(count)
        output = torch.zeros(len(count), max_c, hiddens.size(1))
        index = 0
        for i, count_i in enumerate(count):
            output[i, :count_i] = hiddens[index: index + count_i]
            # output.append(torch.mean(hiddens[index: index + count_i], dim=0))
            index += count_i
        output = self._cuda(output)
        return output

    def fusion(self, hiddens, info):
        info_emb = self.hierarchical_fusion(hiddens, torch.cat(info, dim=1))
        return info_emb

    def forward(self, text, seq_lens, kg_var, kg_lens, kg_count, up_var, ca_var, intent_labels=None, slot_labels=None, type_labels=None):
        info_emb_slot, sent_rep, info_emb_intent = None, None, None
        if self.args.use_pretrained:
            [word_tensor, span] = text
            hiddens = self.encoder(word_tensor)
            if not self.args.s2i:
                sent_rep = hiddens[:, 0]
            hiddens = hiddens[:, 1:-1]
            hiddens = self.match_token(hiddens, span)
        else:
            word_tensor = self.embedding(text)
            lstm_hiddens = self.encoder(word_tensor, seq_lens)
            attention_hiddens = self.attention(word_tensor, seq_lens)
            hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=-1)
            if not self.args.s2i:
                sent_rep = self.sentattention(hiddens, seq_lens)
        
        if self.args.up:
            # up_emb = []
            # for i, up in enumerate(up_var):
            #     up_emb.append(self.encoder_up[i](up).unsqueeze(1))
            # up_emb = torch.cat(up_emb, dim=1)
            up_emb = self.encoder_up(torch.cat(up_var, dim=-1)).unsqueeze(1)
        
        if self.args.ca:
            # ca_emb = []
            # for i, ca in enumerate(ca_var):
            #     ca_emb.append(self.encoder_ca[i](ca).unsqueeze(1))
            # ca_emb = torch.cat(ca_emb, dim=1)
            ca_emb = self.encoder_ca(torch.cat(ca_var, dim=-1)).unsqueeze(1)

        if self.args.kg:
            kg_tensor = self.embedding(kg_var)
            kg_hiddens = self.encoder_kg(kg_tensor, kg_lens, enforce_sorted=False)
            kg_emb = self.get_kg(kg_hiddens, kg_lens)
            kg_emb = self.match_kg(kg_emb, kg_count)
        
        if self.args.use_info:
            info = []
            if self.args.up:
                info.append(up_emb)
            if self.args.ca:
                info.append(ca_emb)
            if self.args.kg:
                info.append(kg_emb)
            info_emb_slot = self.fusion(hiddens, info)
            hiddens = torch.cat([hiddens, info_emb_slot], dim=-1)

        intent_vec = sent_rep
        if self.args.s2i:
            intent_lab, intent_logit = self.intent_label(hiddens)
            slot_lab, slot_logit = self.slot_label(hiddens)
            intent_vec = self.s2i(intent_lab, slot_lab)
        elif self.args.use_info:
            info_emb_intent = self.fusion(sent_rep.unsqueeze(1), info).squeeze(1)
            intent_vec = torch.cat([sent_rep, info_emb_intent], dim=-1)
        pred_intent = self.intent_decoder(intent_vec)

        if self.args.i2s:
            rand = random.random()
            if rand < 0.9 and intent_labels is not None:
                one_hot = F.one_hot(intent_labels, num_classes=self.num_intent).float()
                feed_intent = self.soft_emb(one_hot).unsqueeze(1).repeat(1, hiddens.size(1), 1)
            else:
                pred = torch.argmax(pred_intent, dim=-1)
                one_hot = F.one_hot(pred, num_classes=self.num_intent).float()
                feed_intent = self.soft_emb(one_hot).unsqueeze(1).repeat(1, hiddens.size(1), 1)
            hiddens = torch.cat([hiddens, feed_intent], dim=-1)
        pred_slot = self.slot_decoder(hiddens)

        total_loss = 0
        if intent_labels is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(pred_intent.view(-1, self.num_intent), intent_labels)
            if self.args.aux_task:
                fct = nn.BCEWithLogitsLoss()
                one_hot = F.one_hot(intent_labels, num_classes=self.num_intent)
                aux_intent = fct(intent_logit, one_hot.float())
                intent_loss += aux_intent
            total_loss += intent_loss * self.args.intent_slot_coef

        attention_mask = self.sequence_mask(torch.tensor(seq_lens))
        if slot_labels is not None:
            if self.args.use_crf:
                slot_loss = self.crf(pred_slot, slot_labels, mask=attention_mask.to(pred_slot.device).byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = pred_slot.view(-1, self.num_slot)[active_loss]
                    active_labels = slot_labels.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(pred_slot.view(-1, self.num_slot), slot_labels.view(-1))
            if self.args.aux_task:
                fct = nn.BCEWithLogitsLoss()
                aux_slot = fct(slot_logit, type_labels.float())
                slot_loss += aux_slot
            total_loss += slot_loss * (1 - self.args.intent_slot_coef)

        return pred_slot, pred_intent, total_loss
    
    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        mask = x.unsqueeze(0) < length.unsqueeze(1)
        # mask[:, 0] = 0
        return mask