from collections import Counter

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        curr = Counter()
        left = 0
        ans = 0
        for right, c in enumerate(s):
            curr[c] += 1
            while curr[c] > 1:
                curr[s[left]] -= 1
                left += 1
            ans = max(ans, right-left+1)
        
        return ans
        
if __name__=="__main__":
    res = Solution().lengthOfLongestSubstring("abcabcbb")
    print(res)