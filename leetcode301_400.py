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
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution:
    # 303
    def __303_init__(self, nums: List[int]):
        self.sum_range = [0]
        for num in nums:
            self.sum_range.append(self.sum_range[-1] + num)

    def sumRange(self, left: int, right: int) -> int:
        return self.sum_range[right + 1] - self.sum_range[left]

    # 309
    def maxProfit(self, prices: List[int]) -> int:
        buy = [-prices[0]]
        buy_max = [-prices[0]]
        sell = [0]
        sell_max = [0]
        prices = prices[1:]
        for price in prices:
            buy.append(sell_max[-1]-price)
            buy_max.append(max(buy[-2], buy_max[-1]))
            sell.append(buy_max[-1]+price)
            sell_max.append(max(sell[-2], sell_max[-1]))
        return max(sell_max[-1], sell[-1])

    # 318
    def maxProduct(self, words: List[str]) -> int:
        def convert_to_int(word):
            ans = 0
            for letter in word:
                ans |= (1 << (ord(letter) - ord('a') + 1))
            return ans

        word_int_map = {}
        for word in words:
            word_int_map[word] = convert_to_int(word)
        ans = 0
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if word_int_map[words[i]] & word_int_map[words[j]] == 0:
                    ans = max(ans, len(words[i]) * len(words[j]))
        return ans

    # 322
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0:
            return 0
        dp = [-1 for i in range(amount+1)]
        for coin in coins:
            for i in range(coin, amount+1):
                if i == coin:
                    dp[i] = 1
                elif dp[i] == -1 and dp[i-coin] != -1:
                    dp[i] = dp[i-coin] + 1
                elif dp[i] != -1 and dp[i-coin] != -1:
                    dp[i] = min(dp[i], dp[i-coin]+1)
        return dp[-1]

    # 343
    def integerBreak(self, n: int) -> int:
        if n == 2:
            return 1
        elif n == 3:
            return 2
        dp = [i for i in range(n+1)]
        for i in range(4, n+1):

            for j in range(1, i//2+1):
                dp[i] = max(dp[i], dp[i-j]*dp[j])
        return dp[-1]

    # 376
    def wiggleMaxLength(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 1
        ans = []
        flag = None
        for num in nums:
            if not ans:
                ans.append(num)
            elif flag == None:
                if num == ans[0]:
                    continue
                elif num > ans[0]:
                    flag = True
                else:
                    flag = False
                ans.append(num)
            if num == ans[-1]:
                continue
            elif flag and num > ans[-1] or not flag and num < ans[-1]:
                ans[-1] = num
            else:
                ans.append(num)
                flag = not flag
        return len(ans)

    # 377
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0 for i in range(target+1)]
        dp[0] = 1
        nums = sorted(nums)
        for i in range(target+1):
            for num in nums:
                if i < num:
                    break
                dp[i] += dp[i-num]
        return dp[-1]
