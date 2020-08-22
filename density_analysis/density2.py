# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:00:00 2020

@author: paulg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:12:39 2020

@author: paulg
"""

from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import math
import numpy as np

file = r'Recycleye-data-on-SKUs-and-weights.xlsx'
df = pd.read_excel(file)

number_item=df['Multipack Value']
volume_item=df['Product Size']
mass_item=df['Primary plastic weight']
name_item=df['Global Product Name']
name2_item=df['Product Name']
categories_item=df['Sub Category Name']
bar_code=df['Product Code']

tonnage=df['% of total tonnage']

weight_min=5

#c stands for container and b for bottle and sd for soft drinks
n=len(df['Global Product Name'])

vol_min=200
vol_max=5000
filter_volume=1

filter_name=0

volume_class=1000

name_c=[]
bar_code_c=[]
V_c=[]
m_c=[]
d_c=[]

V_c_pct=[]
m_c_pct=[]
d_c_pct=[]


for i in range(n):
    v=str(volume_item[i])
    cate=str(categories_item[i])
    if (v[-1]=='L' or v[-1]=='l') and (cate=='Impulse' or 'SOFT' in cate or 'Soft' in cate):
        v1=int(v[:-2])
        m=mass_item[i]/number_item[i]
        if (v1>=vol_min and v1<=vol_max and m>weight_min) or filter_volume==0 :
            name_min=name2_item[i].lower()
            if ('water' in name_min)   or filter_name==0:
                V_c+=[v1]
                pct=int(tonnage[i]*1e5)
                V_c_pct+=[v1]*pct
                m_c+=[m]
                m_c_pct+=[m]*pct
        
                d_c+=[m/v1]
                d_c_pct+=[m/v1]*pct
                
                name_c+=[name2_item[i]]
                bar_code_c+=[bar_code[i]]
class_c,inds_class_c,counts_per_class=np.unique(V_c_pct, return_inverse=True, return_counts=True)  

ind_class=int(np.where(class_c==volume_class)[0])
mc_class=np.array(m_c_pct)[inds_class_c==ind_class]



class_c2,inds_class_c2,counts_per_class2=np.unique(V_c, return_inverse=True, return_counts=True)  

ind_class2=int(np.where(class_c2==volume_class)[0])
mc_class2=np.array(m_c)[inds_class_c2==ind_class2]
barc_class2=np.array(bar_code_c)[inds_class_c2==ind_class2]
name_class2=np.array(name_c)[inds_class_c2==ind_class2]
print('moyenne:',np.mean(mc_class))

print('std:',np.std(mc_class))


varia=mc_class
weights = np.ones_like(varia) / len(varia)*100
plt.hist(varia,bins=20,weights=weights)

plt.xlabel('Mass in g')
plt.ylabel('% of the products')
plt.show()












# plt.hist(class_c,bins=27,weights=counts_per_class/sum(counts_per_class))
# plt.xlabel('Volume in mL')
# plt.xticks(np.insert(np.arange(500, 5500, 500),0,200))

# plt.ylabel('% of total tonnage')
# plt.show()

# moyenne=[]
# std_li=[]
# for volume_class in class_c:
#     for i in range(n):
#         v=str(volume_item[i])
#         cate=str(categories_item[i])
#         if (v[-1]=='L' or v[-1]=='l') and (cate=='Impulse' or 'SOFT' in cate or 'Soft' in cate):
#             v1=int(v[:-2])
#             m=mass_item[i]/number_item[i]
#             if (v1>=vol_min and v1<=vol_max and m>weight_min) or filter_volume==0 :
#                 V_c+=[v1]
#                 pct=int(tonnage[i]*1e5)
#                 V_c_pct+=[v1]*pct
#                 m_c+=[m]
#                 m_c_pct+=[m]*pct
        
#                 d_c+=[m/v1]
#                 d_c_pct+=[m/v1]*pct
                
#                 name_c+=[name2_item[i]]
#                 bar_code_c+=[bar_code[i]]
#                 name_c+=[name_item[i]]
#     class_c,inds_class_c,counts_per_class=np.unique(V_c_pct, return_inverse=True, return_counts=True)  
    
#     ind_class=int(np.where(class_c==volume_class)[0])
#     mc_class=np.array(m_c_pct)[inds_class_c==ind_class]


#     moyenne+=[np.mean(mc_class)]
#     std_li+=[np.std(mc_class)]
