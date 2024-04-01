import model_script
from keras.models import load_model
import evaluate_and_plot as eval_script


model=load_model('best_model_unet.hdf5')
print('pre-trained model is loaded successfully, now ready to predict')

X_test = model_script.X_test
eval_script.eval_plot(X_test,model)



