import re
from typing import Optional, List


class Solution:

    # 518
    def change(self, amount: int, coins: List[int]) -> int:
        if amount == 0:
            return 1
        dp = [0 for i in range(amount+1)]
        dp[0] = 1
        for coin in coins:
            for i in range(amount-coin+1):
                dp[i+coin] += dp[i]
        if dp[-1] == -1:
            return 0
        else:
            return dp[-1]

    # 566
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        X, Y = len(mat), len(mat[0])
        if X * Y != r * c:
            return mat

        ans = [[0 for i in range(c)] for _ in range(r)]
        n_x, n_y = 0, 0
        for x in range(X):
            for y in range(Y):
                ans[n_x][n_y] = mat[x][y]
                n_y += 1
                if n_y == c:
                    n_y = 0
                    n_x += 1
        return ans

    # 583
    def minDistance(self, word1: str, word2: str) -> int:
        l1, l2 = len(word1), len(word2)
        dp = [[0 for i in range(l2 + 1)] for i in range(l1 + 1)]

        for index1, letter1 in enumerate(word1):
            for index2, letter2 in enumerate(word2):
                if letter1 == letter2:
                    dp[index1 + 1][index2 + 1] = dp[index1][index2] + 1
                else:
                    dp[index1 + 1][index2 + 1] = max(dp[index1][index2], dp[index1 + 1][index2], dp[index1][index2 + 1])
        return len(word1) + len(word2) - 2 * dp[-1][-1]
