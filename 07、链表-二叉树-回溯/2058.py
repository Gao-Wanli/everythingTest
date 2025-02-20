from typing import List, Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
   
        
class Solution:
    def nodesBetweenCriticalPoints(self, head: Optional[ListNode]) -> List[int]:
        pre = None
        mind, maxd = 2**31-1, -1
        idx = 0
        first_pos, pre_pos, pos = -1, -1, -1
        
        while head.next != None:
            if pre != None:
                if head.val > max(head.next.val, pre.val) or head.val < min(head.next.val, pre.val):
                    pos = idx
                    # 是极点，继续操作
                    if first_pos != -1:
                        # 不是第一个极点，可以计算
                        mind = min(mind, pos-pre_pos)
                    else:
                        first_pos = pos
                    pre_pos = pos       
                    
            # 更新list状态
            idx += 1
            pre = head
            head = head.next
        
        if pos != -1 and first_pos != pos:
            maxd = pos - first_pos
        if mind == 2**31-1:
            mind = -1
            
        return [mind, maxd]
    

node_list = [2,2,1,3]
node = ListNode(node_list[0])
head = node
for i in range(1, len(node_list)):
    temp = ListNode(node_list[i])
    node.next = temp
    node = node.next
    
res = Solution().nodesBetweenCriticalPoints(head)
print(res)
        