
import os, sys
sys.path.insert(0,os.getcwd() )

from exp.settings import settings
from exp.exp_model import Exp_Model


args=settings()
print('Args in experiment:')
print(args)
exp_model = Exp_Model(args)

exp_model.train()

exp_model.test()


