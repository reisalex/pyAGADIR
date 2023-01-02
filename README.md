# Alpha-helix probability model (AGADIR)

```
from pyagadir import predict_alphahelix_propensity
result = predict_alphahelix_propensity('ILKSLEEFLKVTLRSTRQT')
print(result.percent_helix)
print(result.helical_propensity)
```

```
from pyagadir.models import AGADIR

# create model object with an alternative partition function
# assumption, in this case, the multiple-sequence approximation
model = AGADIR(method='ms')

result = model.predict('ILKSLEEFLKVTLRSTRQT')
print(result.percent_helix)
print(result.helical_propensity)
```