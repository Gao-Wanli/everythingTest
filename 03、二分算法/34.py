from typing import List


def lower_bound(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] >= target:
            right = mid - 1
        else:
            left = mid + 1
        
    return left

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        start = lower_bound(nums, target)
        
        # nums里没有target，直接返回
        if start == -1 or start == len(nums) or nums[start] != target:
            return [-1, -1]
        end = lower_bound(nums, target+1)
        
        return [start, end-1]


if __name__=="__main__":
    res = Solution().searchRange([5,7,7,8,8,10], 6)
    print(res)