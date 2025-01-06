import streamlit  as st 
import numpy as np   
import pickle 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load the lstm model :
model = load_model('next_word_lstm.h5')

# laod the tokenizoer 
with open("tokenizor.pkl" , 'rb') as file :
    tokenizor = pickle.load(file)


def predict_next_word ( model  ,  tokenizor , text , max_sequence_len):
    token_list = tokenizor.texts_to_sequences([text])[0]

    if len(token_list)>= max_sequence_len : 
        token_list = token_list[-(max_sequence_len-1):] # ensure that the length should be match with max_seqence 

    token_list = pad_sequences([token_list]  , maxlen=max_sequence_len -1 , padding="pre")

    predicted = model.predict(token_list , verbose=0)

    predict_word_index = np.argmax(predicted , axis=1)
    # print(predict_word_index)
    for   index , word  in tokenizor.index_word.items() :
        if index == predict_word_index :
            return word 
    return None
    

# stremlit app :

st.title("Next Word prediction with LSTM")

input_text = st.text_input("Enter the Sequence of Words :", "To be or not to be ")

if st.button("predict next word") :
    next_word = predict_next_word(model , tokenizor,input_text , max_sequence_len=model.input_shape[1]+1)
    st.write(f'next word : {next_word}')
