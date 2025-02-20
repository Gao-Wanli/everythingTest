from typing import List


class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        # 同一包裹苹果，可以分开装
        # 那就只考虑苹果总量
        apple_sum = sum(apple)
        
        # 从大到小
        capacity = sorted(capacity, reverse=True)
        idx = 0
        while apple_sum > 0:
            # 找最大容量
            apple_sum -= capacity[idx]
            idx += 1
        
        return idx
    
    
res = Solution().minimumBoxes([1,3,2], [4,3,1,5,2])
print(res)