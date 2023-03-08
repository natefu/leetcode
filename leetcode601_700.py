from operator import itemgetter
import re
from typing import Optional, List
from math import sqrt
import bisect

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    # 628
    def maximumProduct(self, nums: List[int]) -> int:
        max_1 = max_2 = max_3 = min(nums)-1
        min_1 = min_2 = 0
        for num in nums:
            if num > max_1:
                max_1, max_2, max_3 = num, max_1, max_2
            elif num > max_2:
                max_2, max_3 = num, max_2
            elif num > max_3:
                max_3 = num
            if num < min_1:
                min_1, min_2 = num, min_1
            elif num < min_2:
                min_2 = num
        return max(max_1*max_2*max_3, max_1*min_1*min_2)

    # 646
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        pairs = sorted(pairs, key=itemgetter(1))
        ans = []
        for pair in pairs:
            if not ans:
                ans.append(pair)
            if pair[0] > ans[-1][1]:
                ans.append(pair)
        return len(ans)

    # 669
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        a
        if not root:
            return root
        if root.val < low:
            return self.trimBST(root.right, low, high)
        elif root.val > high:
            return self.trimBST(root.left, low, high)
        else:
            root.left = self.trimBST(root.left, low, high)
            root.right = self.trimBST(root.right, low, high)
            return root


