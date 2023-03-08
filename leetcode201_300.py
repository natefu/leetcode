import re
from typing import Optional, List
from math import sqrt
import bisect


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution:
    # 213
    def rob(self, nums: List[int]) -> int:
        def find_max_value(nums: List[int]) -> int:
            if len(nums) < 3:
                return max(nums)
            dp = [0 for i in range(len(nums) + 1)]
            dp[1], dp[2] = nums[0], nums[1]
            for i in range(2, len(nums)):
                dp[i + 1] = nums[i] + max(dp[i - 1], dp[i - 2])
            return max(dp[-1], dp[-2])

        if len(nums) < 3:
            return max(nums)
        return max(find_max_value(nums[:-1]), find_max_value(nums[1:]))

    # 227
    def calculate(self, s: str) -> int:
        s.replace(" ", "")
        elements = re.split('([\+,\-,\*,\/])', s)[::-1]
        new_string = []
        while elements:
            element = elements.pop()
            if element == '*':
                left = new_string.pop()
                right = int(elements.pop())
                new_string.append(left*right)
            elif element == '/':
                left = new_string.pop()
                right = int(elements.pop())
                new_string.append(left//right)
            elif element in ['-', '+']:
                new_string.append(element)
            else:
                new_string.append(int(element))
        elements = new_string[::-1]
        count = None
        while elements:
            element = elements.pop()
            if count == None:
                count = element
            elif element == '+':
                count += elements.pop()
            else:
                count -= elements.pop()
        return count

    # 235
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def find_common_ancestor(root, p, q):
            if root.val in [p.val, q.val]:
                return root
            elif root.val < p.val:
                return find_common_ancestor(root.right, p, q)
            elif root.val > q.val:
                return find_common_ancestor(root.left, p, q)
            else:
                return root
        if p.val > q.val:
            p, q = q, p
        return find_common_ancestor(root, p, q)

    # 246
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def get_common_ancestor(root, p, q):
            if not root:
                return 0
            if root.val in [p.val, q.val]:
                left = get_common_ancestor(root.left, p, q)
                right = get_common_ancestor(root.right, p, q)
                if left + right >= 1:
                    self.ancestor = root
                    return 2
                else:
                    return 1
            else:
                left = get_common_ancestor(root.left, p, q)
                right = get_common_ancestor(root.right, p, q)
                if left >= 1 and right >= 1:
                    self.ancestor = root
                    return 2
                else:
                    return left + right

        self.ancestor = root
        get_common_ancestor(root, p, q)
        return self.ancestor

    # 240
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        X, Y = len(matrix), len(matrix[0])
        x, y = 0, Y - 1
        while x < X and y >= 0:
            if matrix[x][y] == target:
                return True
            elif matrix[x][y] < target:
                x += 1
            else:
                y -= 1
        return False

    # 279
    def numSquares(self, n: int) -> int:
        target = int(sqrt(n))
        dp = [i for i in range(n + 1)]
        for i in range(target + 1):
            dp[i * i] = 1
        for i in range(n + 1):
            for j in range(target + 1):
                if i - j * j >= 0:
                    dp[i] = min(dp[i], dp[i - j * j] + 1)
        return dp[-1]

    # 300
    def lengthOfLIS(self, nums: List[int]) -> int:
        ans = []
        for num in nums:
            position = bisect.bisect_left(ans, num)
            if position == len(ans):
                ans.append(num)
            else:
                ans[position] = num
        return len(ans)

