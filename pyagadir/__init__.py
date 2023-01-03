from .models import AGADIR

def predict_alphahelix(seq):
    assert isinstance(seq,str), \
        'Parameter `seq` passsed to predict_alphahelix should be of type str.'
    seq = seq.upper()
    assert set(list(seq)) <= set(list('ACDEFGHIKLMNPQRSTVWY')), \
        'Parameter `seq` should contain only natural amino acids: ACDEFGHIKLMNPQRSTVWY.'
    model  = AGADIR()
    result = model.predict(seq)
    return result