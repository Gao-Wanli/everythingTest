class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        bag = [0]*(high+1)
        bag[0] = 1
        
        for i in range(high+1):
            if i >= zero:
                bag[i] += bag[i-zero]%(10**9+7)
            if i >= one:
                bag[i] += bag[i-one]%(10**9+7)
        
        return sum(bag[low:high+1])%(10**9+7)
    
    
if __name__=="__main__":
    res = Solution().countGoodStrings(3, 3, 1, 1)
    print(res)