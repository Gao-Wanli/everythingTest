from typing import List


class Solution:
    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        
        curr = [0]*51
        ans = [0]*(len(nums)-k+1)
        for i in range(k):
            if nums[i] < 0:
                curr[nums[i]] += 1
                
        temp = x
        for j in range(-50, 0, 1):
            temp -= curr[j]
            if temp <= 0:
                ans[0] = j
                break
        
        for i in range(1, len(nums)-k+1):
            if nums[i-1] < 0:
                curr[nums[i-1]] -= 1
            if nums[i+k-1] < 0:
                curr[nums[i+k-1]] += 1
            
            temp = x
            for j in range(-50, 0, 1):
                temp -= curr[j]
                if temp <= 0:
                    ans[i] = j
                    break
        
        return ans
    

res = Solution().getSubarrayBeauty([-3,1,2,-3,0,-3], 2, 1)
print(res)
        
            