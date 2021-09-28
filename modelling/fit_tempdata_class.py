import numpy as np
import xlrd
from matplotlib import pyplot as plt
import ratfit

crud=xlrd.open_workbook('A02_GAIN_MIN20_373.xlsx')
sheet=crud.sheet_by_index(0)

nu=np.asarray(sheet.col_values(0))
nu=nu/1e6 #convert frequency from Hz to MHz
gain=np.asarray(sheet.col_values(1))

nu_min=15

ii=nu>nu_min
nu=nu[ii]
gain=gain[ii]

ord=6
x=nu
x=x-x.min()
x=x/x.max()
x=2*x-1
A=np.polynomial.legendre.legvander(x,ord)

pars=np.linalg.pinv(A)@gain
d_true=A@pars
plt.clf()
plt.plot(nu,gain)
plt.plot(nu,d_true)
#plt.plot(nu,gain-d_true)
mynoise=np.std(gain-d_true)
print('I think the per-point noise is ',mynoise)
N=np.eye(len(x))*mynoise**2
Ninv=np.eye(len(x))*mynoise**-2
mat=A.T@Ninv@A
errs=np.linalg.inv(mat)
print('parameter errors are ',np.sqrt(np.diag(errs)))

derrs=A@errs@A.T
model_sig=np.sqrt(np.diag(derrs))
plt.plot(nu,d_true+model_sig,'r')
plt.plot(nu,d_true-model_sig,'r')

