from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        l = len(temperatures)
        ans = [0] * l
        st = []
        
        # 从右向左遍历
        for i in range(l-1, -1, -1):
            t = temperatures[i]
            # 栈顶元素小于当前元素（即小的这个元素无用了），则弹出栈顶元素
            while st and temperatures[st[-1]] <= t:
                st.pop()
            if st:
                ans[i] = st[-1] - i
            st.append(i)
            
        return ans


if __name__ == '__main__':
    res = Solution().dailyTemperatures([73,74,75,71,69,72,76,73])
    print(res)