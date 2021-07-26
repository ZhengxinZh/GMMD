#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import sys

log=np.array([range(5)])
xname = sys.argv[1]
yname = sys.argv[2]
np.save('results/'+xname+'_'+yname+'/'+xname+'_'+yname,log)
