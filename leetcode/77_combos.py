class Solution:
    def combine(self, n: int, k: int) :
        combos = []
        

        for i in range(1, k+1) :
            entry = []
            for j in range(1, n+1) :
                combos.append(j)
        
        
        
        return combos
    
#%%
sol = Solution()
res = sol.combine(4,2)
print(res)