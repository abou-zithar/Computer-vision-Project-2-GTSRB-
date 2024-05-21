from My_utils import split_data, order_test_set,create_generators
from deeplearning_models import streesigns_model

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping


import tensorflow as tf



if __name__ == "__main__":
    if False:
        path_to_data = "D:\\Projects Computer Vision\\Computer vision Project 2 (GTSRB)\\GTSRB\\archive\\Train"
        path_to_save_train = "D:\\Projects Computer Vision\\Computer vision Project 2 (GTSRB)\\GTSRB\\archive\\Trainning_data\\train"
        path_to_save_val = "D:\\Projects Computer Vision\\Computer vision Project 2 (GTSRB)\\GTSRB\\archive\\Trainning_data\\val"
        split_data(path_to_data,path_to_save_train,path_to_save_val)

    if False:
        path_to_images= "D:\\Projects Computer Vision\\Computer vision Project 2 (GTSRB)\\GTSRB\\archive\\Test"
        path_to_csv = "D:\\Projects Computer Vision\\Computer vision Project 2 (GTSRB)\\GTSRB\\archive\\Test.csv"
        order_test_set(path_to_images=path_to_images,path_to_csv=path_to_csv)
    TRAIN =False
    TEST=True

    
    path_to_save_train = "D:\\Projects Computer Vision\\Computer vision Project 2 (GTSRB)\\GTSRB\\archive\\Trainning_data\\train"
    path_to_save_val = "D:\\Projects Computer Vision\\Computer vision Project 2 (GTSRB)\\GTSRB\\archive\\Trainning_data\\val"
    path_to_save_test = "D:\\Projects Computer Vision\\Computer vision Project 2 (GTSRB)\\GTSRB\\archive\\Test"
    #changing this values my imporve the mode accuracy 
    batch_size =64
    epochs= 15
    lr= 0.001

    train_generator,val_generator,test_generator = create_generators(batch_size,path_to_save_train,path_to_save_val,path_to_save_test)
    
    nbr_classes = train_generator.num_classes 
    if TRAIN:
        path_to_save_model = '.\Models.keras'

        ckpt_saver= ModelCheckpoint(
            path_to_save_model,
            monitor= "val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

    # if the val_accuracy didnot change after 10 epochs stop the trainning
        early_stop= EarlyStopping(
            monitor= "val_accuracy",patience=10

        )


        model = streesigns_model(nbr_classes)



        Adam =tf.keras.optimizer.Adam(learning_rate=lr,amsgrad=True)
        # we sit this to categorical_crossentropy because we use the class mode in genrator to categorical
        model.compile(optimizer = 'adam',loss= 'categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(train_generator
                ,epochs = 15,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[ckpt_saver,early_stop]
                )
        

    if TEST:
        model = tf.keras.models.load_model("Models.keras")
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_generator)

        print("Evaluating test set : ")
        model.evaluate(test_generator)

















    path_to_save_train = "D:\\Computer vision\\GTSRB\\archive\\Trainning_data\\train"
    path_to_save_val = "D:\\Computer vision\\GTSRB\\archive\\Trainning_data\\val"
    path_to_save_test = "D:\\Computer vision\\GTSRB\\archive\\Test"