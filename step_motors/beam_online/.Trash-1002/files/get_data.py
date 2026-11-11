import epics
import numpy as np

pv_name = 'lsrp:spot:hist'
pv = epics.PV(pv_name)

value = pv.get()

np_value = np.array((value))
np_value.resize(20,32)
#print(f"Текущее значение {pv_name}: {np_value}")

for i in range(20):
	for j in range(32):
		print(f"{np_value[i][j]:5}", end=" ")
		
	print()
