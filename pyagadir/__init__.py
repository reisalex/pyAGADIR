from .models import AGADIR

def predict_alphahelix_propensity(seq):
    model  = AGADIR()
    result = model.predict(seq)
    return result