from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def getDecimalValue(self, head: Optional[ListNode]) -> int:
        ans = 0
        
        while head != None:
            ans = ans*2 + head.val
            head = head.next
        
        return ans
    

node_list = [1, 0, 1]
node = ListNode(node_list[0])
head = node
for i in range(1, len(node_list)):
    temp = ListNode(node_list[i])
    node.next = temp
    node = node.next

res = Solution().getDecimalValue(head)
print(res)
        