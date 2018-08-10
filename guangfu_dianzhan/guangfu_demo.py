#encoding=utf-8
import pandas as pd
result=pd.DataFrame({
    'a':[1,2,3,4],
    'b':[2,3,4,5]
})
result['var']=result.var(1)
print(result)