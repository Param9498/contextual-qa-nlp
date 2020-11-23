import torch
import torch.nn as nn

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FirstQuestionAnsweringModel():

	def __init__(self):

		# BERT Finetuned on SQUAD

		self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
		self.squad_finetuned_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
		self.squad_finetuned_model = self.squad_finetuned_model.eval()
		self.squad_finetuned_model = self.squad_finetuned_model.to(device)

	def answer_question(self, question, answer_text):
		
		input_ids = self.bert_tokenizer.encode(question, answer_text, max_length=512, truncation=True)
		
		sep_index = input_ids.index(self.bert_tokenizer.sep_token_id)
		num_seg_a = sep_index + 1
		num_seg_b = len(input_ids) - num_seg_a

		segment_ids = [0]*num_seg_a + [1]*num_seg_b

		assert len(segment_ids) == len(input_ids)

		inps = torch.tensor([input_ids]).to(device)
		segs = torch.tensor([segment_ids]).to(device)
		start_scores, end_scores = self.squad_finetuned_model(inps, token_type_ids = segs)

		answer_start = torch.argmax(start_scores)
		answer_end = torch.argmax(end_scores)

		tokens = self.bert_tokenizer.convert_ids_to_tokens(input_ids)

		answer = tokens[answer_start]

		for i in range(answer_start + 1, answer_end + 1):
			if tokens[i][0:2] == '##':
				answer += tokens[i][2:]
			else:
				answer += ' ' + tokens[i]

		return answer