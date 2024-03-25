from .models import AGADIR

def predict_alphahelix(seq):
    if not isinstance(seq, str):
        raise ValueError('Parameter `seq` passsed to predict_alphahelix should be of type str.')
    
    seq = seq.upper()
    if not set(list(seq)) <= set(list('ACDEFGHIKLMNPQRSTVWY')):
        raise ValueError('Parameter `seq` should contain only natural amino acids: ACDEFGHIKLMNPQRSTVWY.')

    model  = AGADIR()
    result = model.predict(seq)
    return result