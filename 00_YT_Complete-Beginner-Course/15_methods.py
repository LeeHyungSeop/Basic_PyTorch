import torch
import torch.nn as nn

### complete model ###
torch.save(arg, PATH)

### load model. model class must be defined somewhere ###
model = Model(*args, **kwargs)
model = torch.load(PATH)
model.eval()


### STATE DICT ###
torch.save(model.state_dict(), PATH)

# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()