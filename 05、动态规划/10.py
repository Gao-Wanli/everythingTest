class Solution:
    def climbStairs(self, n: int) -> int:
        bag = [0]*(n+1)

        bag[0] = bag[1] = 1      #初始化
        for i in range(2,n+1):
            bag[i] = bag[i-1] + bag[i-2]
        
        return bag[n]
        
        
if __name__ == '__main__':
    res = Solution().climbStairs(4)
    print(res)