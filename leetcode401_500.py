import re
from typing import Optional, List


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
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution:
    # 409
    def longestPalindrome(self, s: str) -> int:
        from collections import defaultdict
        s_dict = defaultdict(int)
        for letter in s:
            s_dict[letter] += 1
        count = 0
        print(s_dict)
        for _, s_c in s_dict.items():
            if s_c % 2 == 0:
                count += s_c
            else:
                if count % 2 == 0:
                    count += s_c
                else:
                    count += s_c - 1
        return count
    # 413
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return 0
        dp = []
        for index, num in enumerate(nums):
            if index < 2:
                dp.append(0)
                continue
            else:
                if num - nums[index - 1] == nums[index - 1] - nums[index - 2]:
                    dp.append(dp[-1] + 1)
                else:
                    dp.append(0)
        return sum(dp)

    # 416
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums) % 2 == 1:
            return False
        half = sum(nums)//2
        dp = [False for i in range(half+1)]
        dp[0] = True
        for num in nums:
            for i in reversed(range(half+1-num)):
                if dp[i]:
                    dp[i+num] = True
        return dp[-1]

    # 438
    def findAnagrams(self, s, p):
        from collections import deque
        start = 0
        used_letter = deque([])
        p_list = list(p)
        ans = []
        for index, letter in enumerate(s):
            if letter in p_list:
                used_letter.append(letter)
                p_list.remove(letter)
                if not p_list:
                    ans.append(index-len(p)+1)
            else:
                if letter in used_letter:
                    while used_letter:
                        pop = used_letter.popleft()
                        if pop == letter:
                            used_letter.append(pop)
                            break
                        else:
                            p_list.append(pop)
                    if not p_list:
                        ans.append(index-len(p)+1)
                else:
                    p_list = list(p)
                    used_letter = deque([])
        return ans

    # 474
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        matrix = [[0 for i in range(n+1)] for j in range(m+1)]

        for s in strs:
            xs = s.count('0')
            ys = s.count('1')
            for x in reversed(range(m-xs+1)):
                for y in reversed(range(n-ys+1)):
                    matrix[x+xs][y+ys] = max(matrix[x+xs][y+ys], matrix[x][y]+1)
        return matrix[-1][-1]

    # 485
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        max_count = 0
        count = 0
        for num in nums:
            if num == 0:
                max_count = max(max_count, count)
                count = 0
            else:
                count += 1
        return max(max_count, count)

    # 494
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """
        X - Y = target
        X + Y = SUM
        2 X = target + SUM
        X = (target+sum) // 2
        """
        target = abs(target)
        if sum(nums) < target or (sum(nums) + target) % 2:
            return 0
        length = (sum(nums) + target) // 2 + 1
        point = [0 for _ in range(length)]
        point[0] = 1
        for num in nums:
            for i in range(length - num)[::-1]:
                if point[i] != 0:
                    point[i + num] += point[i]
        return point[-1]
