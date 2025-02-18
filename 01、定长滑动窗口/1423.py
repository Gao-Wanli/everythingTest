from typing import List

class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        l = len(cardPoints)
        curr = 0
        ans = 0

        # 初始化窗口
        for i in range(0, l-k):
            curr += cardPoints[i]
        ans = curr

        # 窗口滑动
        for i in range(0, k):
            curr -= cardPoints[i]
            curr += cardPoints[i+l-k]

            if ans > curr:
                ans = curr
        
        return sum(cardPoints)-ans


if __name__=="__main__":
    res = Solution().maxScore([1,2,3,4,5,6,1], 3)
    print(res)