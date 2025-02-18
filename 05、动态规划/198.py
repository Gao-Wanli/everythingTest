from typing import List


class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return nums[0] 
        
        bag = [0] * len(nums)
        bag[0] = nums[0]
        bag[1] = max(nums[0], nums[1])
        
        
        for i in range(2, len(nums)):
            bag[i] = max(bag[i - 1], bag[i - 2] + nums[i])
            
        return bag[-1]
    
    
if __name__ == '__main__':
    res = Solution().rob([2, 7, 9, 3, 1])
    print(res)