import json
from model import FirstQuestionAnsweringModel

def load_coqa(train_filename='coqa-train-v1.0.json', dev_filename='coqa-dev-v1.0.json', need = ['train', 'dev']):
	if 'train' in need:
		if not os.path.exists(train_filename):
			!wget "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json" -O $filename
		train_dataset = json.load(open('coqa-train-v1.0.json', 'r'))
	if 'dev' in need:
		if not os.path.exists(dev_filename):
			!wget "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json" -O $filename
		dev_dataset = json.load(open('coqa-dev-v1.0.json', 'r'))
	
	if 'train' in need and 'dev' in need:
		return train_dataset, dev_dataset
	elif 'train' in need and 'dev' not in need:
		return train_dataset
	elif 'dev' in need and 'train' not in need:
		return dev_dataset
	else:
		return None

dev_dataset = load_coqa(need=['dev'])

model = FirstQuestionAnsweringModel()

total = 0
predictions = []

for i in tqdm.tqdm(range(len(dev_dataset['data']))):

	id = dev_dataset['data'][i]['id']
    story = dev_dataset['data'][i]['story']

	count = 0

	other_answers_per_story = dev_dataset['data'][i]['additional_answers']

	for question_turn in dev_dataset['data'][i]['questions']:
		question = question_turn['input_text']
        turn_id = question_turn['turn_id']

		answer = answer_question(question, story)

		assert turn_id == dev_dataset['data'][i]['answers'][count]['turn_id']
		
		multiple_answers_input_text = []
		multiple_answers_span_text = []
		for key in other_answers_per_story:
			other_answers = other_answers_per_story[key]

			for other_answer in other_answers:
				if other_answer['turn_id'] == turn_id:
					multiple_answers_input_text.append(other_answer['input_text'])
					multiple_answers_input_text.append(other_answer['span_text'])

		predictions.append({
            'id': id,
            'turn_id': turn_id,
            'predicted_answer_span': answer,
            'gold_answer_input_text': [dev_dataset['data'][i]['answers'][count]['input_text']] + multiple_answers_input_text
			'gold_answer_span_text': [dev_dataset['data'][i]['answers'][count]['span_text']] + multiple_answers_span_text
        })
		count += 1
		break

print(predictions)



