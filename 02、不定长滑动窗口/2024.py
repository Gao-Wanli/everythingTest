from typing import Counter


class Solution:
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        curr = Counter()
        left = 0
        ans = 0
        
        for right, c in enumerate(answerKey):
            curr[c] += 1
            while curr['T'] > k and curr['F'] > k:
                curr[answerKey[left]] -= 1
                left += 1
            ans = max(ans, right-left+1)
        
        return ans

res =Solution().maxConsecutiveAnswers("TFFT", 1)
print(res)