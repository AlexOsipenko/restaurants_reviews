from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, Lambda

app = Flask(__name__)

bert_model = TFBertModel.from_pretrained('DeepPavlov/rubert-base-cased',  from_pt=True)

input_ids = Input(shape=(100,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(100,), dtype=tf.int32, name='attention_mask')

def bert_layer(inputs):
    input_ids, attention_mask = inputs
    return bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]

bert_output = Lambda(bert_layer, output_shape=(100, 768))([input_ids, attention_mask]) 
x = GlobalAveragePooling1D()(bert_output)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
model = tf.keras.models.load_model('model_reviews.keras', custom_objects={'bert_layer': bert_layer})
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data['text']
        tokens = tokenizer(
            [text], 
            padding='max_length', 
            truncation=True, 
            max_length=100, 
            return_tensors="tf"
        )

        input_ids = tokens['input_ids']
        attention_masks = tokens['attention_mask']
        prediction = model.predict([input_ids, attention_masks])
        prediction_label = 'good' if prediction[0][0] > 0.7 else 'bad'
        print(f"Input: {text}")
        print(f"Prediction: {prediction_label}")
        print(prediction[0][0])
        
        return jsonify({'prediction': prediction_label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)