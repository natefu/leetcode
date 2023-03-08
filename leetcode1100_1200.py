import re
from typing import Optional, List
from operator import itemgetter


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    # 714
    def maxProfit(self, prices: List[int], fee: int) -> int:
        if not len(prices):
            return 0
        buy = [-prices[0]]
        buy_max = [-prices[0]]
        sell = [0]
        sell_max = [0]
        if len(prices) == 0:
            return 0
        for i in range(1, len(prices)):
            buy.append(max(sell[-1], sell_max[-1])-prices[i])
            buy_max.append(max(buy_max[-1], buy[-2]))
            sell.append(buy_max[-1]+prices[i]-fee)
            sell_max.append(max(sell[-2], sell_max[-1]))
        return max(sell[-1], sell_max[-1])

    # 1004
    def longestOnes(self, nums: List[int], k: int) -> int:
        start = 0
        zero_count = k
        max_length = 0
        end = 0
        while end < len(nums):
            num = nums[end]
            if num == 0:
                if zero_count > 0:
                    zero_count -= 1
                else:
                    max_length = max(end-start, max_length)
                    while start<=end:
                        if nums[start] == 0:
                            start += 1
                            break
                        else:
                            start += 1
            end += 1
        return max(max_length, end-start)

    # 1123
    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def lca_length(root, count):
            if not root.left and not root.right:
                if count + 1 > self.length:
                    self.ancestor = root
                    self.length = count + 1
                return count + 1
            elif root.left and root.right:
                left_length = lca_length(root.left, count + 1)
                right_length = lca_length(root.right, count + 1)
                if left_length == right_length:
                    if left_length >= self.length:
                        self.length = left_length
                        self.ancestor = root
                    return left_length
                else:
                    return max(left_length, right_length)
            else:
                if root.left:
                    return lca_length(root.left, count + 1)
                else:
                    return lca_length(root.right, count + 1)

        if not root:
            return root
        self.ancestor = root
        self.length = 0
        lca_length(root, 1)
        return self.ancestor

    # 1143
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        len1, len2 = len(text1), len(text2)
        dp = [[0 for i in range(len2+1)] for j in range(len1+1)]
        for index1, letter1 in enumerate(text1):
            for index2, letter2 in enumerate(text2):
                if letter1 == letter2:
                    dp[index1+1][index2+1] = dp[index1][index2] + 1
                else:
                    dp[index1+1][index2+1] = max(dp[index1][index2], dp[index1+1][index2], dp[index1][index2+1])
        return dp[-1][-1]

    # 1710
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        boxTypes = sorted(boxTypes, key=itemgetter(1), reverse=True)
        unit_size = 0
        for count, size in boxTypes:
            if truckSize < 0:
                break
            unit_size += min(count, truckSize) * size
            truckSize -= count
        return unit_size

    # 1832
    def checkIfPangram(self, sentence: str) -> bool:
        target = 0
        for i in range(1, 27):
            target += (1<<i)
        count = 0
        for letter in sentence:
            if 'a' <= letter <= 'z':
                count |= (1<<ord(letter)-ord('a')+1)
        if target == count:
            return True
        else:
            return False

    # 2244
    def minimumRounds(self, tasks: List[int]) -> int:
        task_map = {}
        for task in tasks:
            if task in task_map:
                task_map[task] += 1
            else:
                task_map[task] = 1
        count = 0
        for _, times in task_map.items():
            if times == 1:
                return -1
            if times % 3 == 0:
                count += times // 3
            else:
                count += times // 3 + 1
        return count