import streamlit as st
import tensorflow as tf
from PIL import Image
from predictor import predict_with_model



def load_model():
  model=tf.keras.models.load_model('Models.keras')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Sign Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])



st.set_option('deprecation.showfileUploaderEncoding', False)




if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions,score = predict_with_model( model,image)
   
    
    st.write("prediction =",predictions)
    # st.write(score)
    # print(f"prediction = {predictions}")
#     print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
