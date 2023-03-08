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

    # 101
    # https://leetcode.com/problems/symmetric-tree/submissions/
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def isSymmetricTree(tree1, tree2):
            if not tree1 and not tree2:
                return True
            elif not tree1 or not tree2:
                return False
            elif tree1.val != tree2.val:
                return False
            else:
                return isSymmetricTree(tree1.left, tree2.right) and isSymmetricTree(tree1.right, tree2.left)

        return isSymmetricTree(root.left, root.right)

    # 102
    # https://leetcode.com/problems/binary-tree-level-order-traversal/submissions/
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        ans = []
        stack = [root]

        while stack:
            new_stack = []
            line = []
            for element in stack:
                line.append(element.val)
                if element.left:
                    new_stack.append(element.left)
                if element.right:
                    new_stack.append(element.right)
            ans.append(line)
            stack = new_stack
        return ans

    # 103
    # https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/submissions/
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        stack = [root]
        ans = []
        flag = True

        while stack:
            new_stack = []
            line = []
            for element in stack:
                line.append(element.val)
                if element.left:
                    new_stack.append(element.left)
                if element.right:
                    new_stack.append(element.right)

            if flag:
                ans.append(line)
                flag = False
            else:
                ans.append(line[::-1])
                flag = True
            stack = new_stack
        return ans

    # 104
    # https://leetcode.com/problems/maximum-depth-of-binary-tree/submissions/
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        stack = [root]
        count = 0
        while stack:
            count += 1
            new_stack = []
            for element in stack:
                if element.left:
                    new_stack.append(element.left)
                if element.right:
                    new_stack.append(element.right)
            stack = new_stack
        return count

    # 107
    # https://leetcode.com/problems/binary-tree-level-order-traversal-ii/
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        stack = [root]
        ans = []

        while stack:
            new_stack = []
            line = []
            for element in stack:
                if element.left:
                    new_stack.append(element.left)
                if element.right:
                    new_stack.append(element.right)
                line.append(element.val)
            ans = [line] + ans
            stack = new_stack
        return ans

    # 108
    # https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/submissions/
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        start, end = 0, len(nums)-1
        mid = start + (end-start) // 2
        node = TreeNode(nums[mid])
        node.left = self.sortedArrayToBST(nums[:mid])
        node.right = self.sortedArrayToBST(nums[mid+1:])
        return node

    # 109
    # https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/submissions/
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        if not head:
            return None
        dummy = ListNode(-1)
        dummy.next = head
        fast = slow = dummy
        pre = slow
        while fast and fast.next:
            fast = fast.next.next
            pre = slow
            slow = slow.next
        element = slow
        pre.next = None
        right = element.next
        element.next = None
        left = dummy.next
        node = TreeNode(element.val)
        node.left = self.sortedListToBST(left)
        node.right = self.sortedListToBST(right)
        return node

    # 110
    # https://leetcode.com/problems/balanced-binary-tree/submissions/
    def isBalanced(self, root: TreeNode) -> bool:
        def isBalancedTree(root):
            if not root:
                return 0, True
            else:
                left_count, left_flag = isBalancedTree(root.left)
                right_count, right_flag = isBalancedTree(root.right)
                if left_flag and right_flag and abs(left_count - right_count) <= 1:
                    return max(left_count, right_count) + 1, True
                else:
                    return -1, False

        _, flag = isBalancedTree(root)
        return flag

    # 111
    # https://leetcode.com/problems/minimum-depth-of-binary-tree/submissions/
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        stack = [root]
        ans = 0
        while stack:
            new_stack = []
            ans += 1
            for element in stack:
                if not element.left and not element.right:
                    return ans
                if element.left:
                    new_stack.append(element.left)
                if element.right:
                    new_stack.append(element.right)
            stack = new_stack
        return ans

    # 112
    # https://leetcode.com/problems/path-sum/submissions/
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def traverse(path, root):
            if not root.left and not root.right:
                if path == targetSum:
                    return True
                else:
                    return False
            result = False
            if root.left:
                result = result or traverse(path+root.left.val, root.left)
            if not result and root.right:
                result = result or traverse(path+root.right.val, root.right)
            return result

        if not root:
            return False
        return traverse(root.val, root)

    # 113
    # https://leetcode.com/problems/path-sum-ii/
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        def traverse(root, path):
            if not root.left and not root.right:
                if sum(path) == targetSum:
                    self.result.append(path)
            if root.left:
                traverse(root.left, path+[root.left.val])
            if root.right:
                traverse(root.right, path+[root.right.val])
        if not root:
            return []
        self.result = []
        traverse(root, [root.val])
        return self.result

    #114
    # https://leetcode.com/problems/flatten-binary-tree-to-linked-list/submissions/
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return
        stack = [root]
        pre = TreeNode()
        while stack:
            node = stack.pop()
            pre.right = node
            pre = node
            while node.left or node.right:
                if node.right:
                    stack.append(node.right)
                if node.left:
                    node.right = node.left
                    node.left = None
                    node = node.right
                    pre = node
                else:
                    break

    # 116
    # https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return root
        stack = [root]
        while stack:
            new_stack = []
            for node in stack:
                if node.left:
                    new_stack.append(node.left)
                if node.right:
                    new_stack.append(node.right)
            if new_stack:
                pre = new_stack[0]
                for index in range(1, len(new_stack)):
                    pre.next = new_stack[index]
                    pre = new_stack[index]
            stack = new_stack
        return root

    # 117
    # https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        stack = [root]
        while stack:
            new_stack = []
            for node in stack:
                if node.left:
                    new_stack.append(node.left)
                if node.right:
                    new_stack.append(node.right)
            if new_stack:
                pre = new_stack[0]
                for index in range(1, len(new_stack)):
                    pre.next = new_stack[index]
                    pre = new_stack[index]
            stack = new_stack
        return root

    # 118
    # https://leetcode.com/problems/pascals-triangle/
    def generate(self, n: int) -> List[List[int]]:
        if n == 1:
            return [[1]]
        if n == 2:
            return [[1], [1, 1]]
        ans = [[1], [1, 1]]
        for i in range(2, n):
            line = []
            for j in range(i + 1):
                if j == 0 or i == j:
                    line.append(1)
                else:
                    line.append(ans[i - 1][j - 1] + ans[i - 1][j])
            ans.append(line)
        return ans

    # 119
    # https://leetcode.com/problems/pascals-triangle-ii/submissions/
    def getRow(self, n: int) -> List[int]:
        if n == 0:
            return [1]
        if n == 1:
            return [1, 1]
        ans = [1, 1]
        for i in range(2, n+1):
            line = []
            for j in range(i + 1):
                if j == 0 or i == j:
                    line.append(1)
                else:
                    line.append(ans[j - 1] + ans[j])
            ans = line
        return ans

    # 120
    # https://leetcode.com/problems/triangle/
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        for i in range(1, len(triangle)):
            for j in range(len(triangle[i])):
                if j == 0:
                    triangle[i][j] += triangle[i-1][j]
                elif j == len(triangle[i])-1:
                    triangle[i][j] += triangle[i-1][j-1]
                else:
                    triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j])
        return min(triangle[-1])

    # 121
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        low = prices[0]
        for index in range(1, len(prices)):
            if prices[index] < low:
                low = prices[index]
            else:
                profit = prices[index] - low
                max_profit = max(max_profit, profit)
        return max_profit

    # 122
    # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        for index in range(1, len(prices)):
            profit = prices[index] - prices[index-1]
            if profit > 0:
                max_profit += profit
        return max_profit

    # 124
    # https://leetcode.com/problems/binary-tree-maximum-path-sum/
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def maxPath(root):
            if not root.left and not root.right:
                return root.val, root.val
            if root.left:
                left_max_with, left_max = maxPath(root.left)
            else:
                left_max_with, left_max = float('-inf'), float('-inf')
            if root.right:
                right_max_with, right_max = maxPath(root.right)
            else:
                right_max_with, right_max = float('-inf'), float('-inf')
            max_with = max(left_max_with+root.val, right_max_with+root.val, root.val)
            current_max = max(max_with, left_max, right_max, left_max_with+root.val+right_max_with)
            return max_with, current_max
        if not root:
            return 0
        _, result = maxPath(root)
        return result

    # 125
    # https://leetcode.com/problems/valid-palindrome/submissions/
    def isPalindrome(self, s: str) -> bool:
        if s == '':
            return True
        rule = re.compile(r'[^a-zA-Z0-9]')
        s = rule.sub('', s)
        s = s.lower()
        return s == s[::-1]

    # 128
    # https://leetcode.com/problems/longest-consecutive-sequence/
    def longestConsecutive(self, nums: List[int]) -> int:
        nums_set = set(nums)
        max_length = 0
        for num in nums:
            target = num
            length = 0
            while target in nums_set:
                length += 1
                nums_set.remove(target)
                target += 1
            target = num - 1
            while target in nums_set:
                length += 1
                nums_set.remove(target)
                target -= 1
            max_length = max(max_length, length)
        return max_length

    # 129
    # https://leetcode.com/problems/sum-root-to-leaf-numbers/
    def sumNumbers(self, root: TreeNode) -> int:
        def traverse(root, path):
            if not root.left and not root.right:
                self.result += path
            if root.left:
                traverse(root.left, path*10+root.left.val)
            if root.right:
                traverse(root.right, path*10+root.right.val)
        if not root:
            return 0
        self.result = 0
        traverse(root, root.val)
        return self.result

    # 130
    # https://leetcode.com/problems/surrounded-regions/
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        X, Y = len(board), len(board[0])

        def dfs(stack):
            while stack:
                x, y = stack.pop()
                if x - 1 >= 0 and board[x - 1][y] == 'O':
                    board[x - 1][y] = 'A'
                    stack.append((x - 1, y))
                if x + 1 < X and board[x + 1][y] == 'O':
                    board[x + 1][y] = 'A'
                    stack.append((x + 1, y))
                if y - 1 >= 0 and board[x][y - 1] == 'O':
                    board[x][y - 1] = 'A'
                    stack.append((x, y - 1))
                if y + 1 < Y and board[x][y + 1] == 'O':
                    board[x][y + 1] = 'A'
                    stack.append((x, y + 1))

        for i in range(X):
            if board[i][0] == 'O':
                board[i][0] = 'A'
                dfs([(i, 0)])
            if board[i][Y - 1] == 'O':
                board[i][Y - 1] = 'A'
                dfs([(i, Y - 1)])

        for i in range(Y):
            if board[0][i] == 'O':
                board[0][i] = 'A'
                dfs([(0, i)])
            if board[X - 1][i] == 'O':
                board[X - 1][i] = 'A'
                dfs([(X - 1, i)])

        for i in range(X):
            for j in range(Y):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'A':
                    board[i][j] = 'O'

    # 131
    # https://leetcode.com/problems/palindrome-partitioning/
    def partition(self, s: str) -> List[List[str]]:
        def traverse(s, path):
            if not s:
                self.result.append(path)
            for index in range(len(s)):
                element = s[:index+1]
                if element == element[::-1]:
                    traverse(s[index+1:], path+[element])
        if not s:
            return []
        self.result = []
        traverse(s, [])
        return self.result

    # 133
    # https://leetcode.com/problems/clone-graph/
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return None
        node_dict = {}
        nodes = [node]
        while nodes:
            new_nodes = []
            for element in nodes:
                if element in node_dict:
                    continue
                else:
                    node_dict[element] = Node(element.val)
                new_nodes += element.neighbors
            nodes = new_nodes
        nodes = [node]
        used_nodes = set()
        while nodes:
            new_nodes = []
            for element in nodes:
                if element in used_nodes:
                    continue
                else:
                    node_dict[element].neighbors = [
                        node_dict[neighbor] for neighbor in element.neighbors
                    ]
                    used_nodes.add(element)
                    new_nodes += element.neighbors
            nodes = new_nodes
        return node_dict[node]

    # 134
    # https://leetcode.com/problems/gas-station/
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        remain = 0
        start = 0
        for index in range(len(gas)):
            if remain + gas[index] >= cost[index]:
                remain = remain + gas[index] - cost[index]
            else:
                remain = 0
                start = index + 1
        return start

    # 135
    # https://leetcode.com/problems/candy/submissions/
    def candy(self, ratings: List[int]) -> int:
        candies = [1 for _ in ratings]

        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1
        for i in reversed(range(0, len(ratings)-1)):
            if ratings[i] > ratings[i+1]:
                candies[i] = max(candies[i], candies[i+1]+1)
        return sum(candies)

    # 136
    # https://leetcode.com/problems/single-number/submissions/
    def singleNumber(self, nums: List[int]) -> int:
        target = 0
        for num in nums:
            target = target ^ num
        return target

    # 137
    # https://leetcode.com/problems/single-number-ii/submissions/
    def singleNumber(self, nums: List[int]) -> int:
        # 1
        '''
        num_set = set(nums)
        sum_nums = 3 * sum(num_set)
        for num in nums:
            sum_nums -= num
        return sum_nums // 2
        '''

        # 2
        signed = 0
        for index, num in enumerate(nums):
            if num < 0:
                signed += 1
                nums[index] *= -1

        target = 0
        for i in range(32):
            count = 0
            for num in nums:
                if num & (1<<i) != 0:
                    count += 1
            count = count % 3
            target += (count<<i)
        if signed % 3:
            return target*-1
        return target

    # 138
    # https://leetcode.com/problems/copy-list-with-random-pointer/
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None
        node_dict = {}
        dummy_head = head
        new_head = Node(head.val)
        node_dict[head] = new_head
        while head.next:
            current_node = Node(head.next.val)
            node_dict[head.next] = current_node
            new_head.next = current_node
            new_head = current_node
            head = head.next
        head = dummy_head
        while head:
            if head.random != None:
                node_dict[head].random = node_dict[head.random]
            head = head.next
        return node_dict[dummy_head]

    # 139
    # https://leetcode.com/problems/word-break/submissions/
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_lengths = sorted(list(set([len(word) for word in wordDict])))
        words = set(wordDict)
        dp = [False for i in range(len(s)+1)]
        dp[0] = True
        for index in range(len(s)):
            if dp[index]:
                for word_length in word_lengths:
                    if index+word_length > len(s):
                        break
                    if s[index:index+word_length] in words:
                        dp[index+word_length] = True
        return dp[-1]

    # 140
    # https://leetcode.com/problems/word-break-ii/
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        def break_words(s, dp, index, word_lengths, path):
            if index == len(s):
                self.result.append(path[:-1])
            for word_length in word_lengths:
                if index + word_length > len(s):
                    break
                else:
                    if dp[index + word_length] and s[index:index + word_length] in words:
                        break_words(s, dp, index + word_length, word_lengths, path + s[index:index + word_length] + ' ')

        word_lengths = sorted(list(set([len(word) for word in wordDict])))
        words = set(wordDict)
        dp = [False for i in range(len(s) + 1)]
        dp[0] = True
        for index in range(len(s)):
            if dp[index]:
                for word_length in word_lengths:
                    if index + word_length > len(s):
                        break
                    if s[index:index + word_length] in words:
                        dp[index + word_length] = True
        if dp[-1]:
            self.result = []
            break_words(s, dp, 0, word_lengths, '')
            return self.result
        else:
            return []

    # 141
    # https://leetcode.com/problems/linked-list-cycle/
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False

        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    # 142
    # https://leetcode.com/problems/linked-list-cycle-ii/submissions/
    def detectCycle(self, head: ListNode) -> ListNode:
        head_index_set = set()
        while head:
            if head in head_index_set:
                return head
            else:
                head_index_set.add(head)
                head = head.next
        return None

    # 143
    # https://leetcode.com/problems/reorder-list/submissions/
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        def reverse(node):
            suf = None
            while node:
                next_node = node.next
                node.next = suf
                suf = node
                node = next_node
            return suf
        dummy = ListNode(0, head)
        fast = slow = dummy
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        first = head
        second = slow.next
        slow.next = None
        second = reverse(second)

        dummy = ListNode(-1)
        while first and second:
            dummy.next = first
            first = first.next
            dummy = dummy.next
            dummy.next = second
            second = second.next
            dummy = dummy.next
        if first:
            dummy.next = first
        if second:
            dummy.next = second

    # 144
    # https://leetcode.com/problems/binary-tree-preorder-traversal/
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        ans = []
        stack = [root]

        while stack:
            node = stack.pop()
            while node:
                ans.append(node.val)
                if node.right:
                    stack.append(node.right)
                node = node.left
        return ans

    # 145
    # https://leetcode.com/problems/binary-tree-postorder-traversal/
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        ans = []
        stack = [(root, 0)]
        while stack:
            node, count = stack.pop()
            if count == 1:
                ans.append(node.val)
            else:
                stack.append((node, count+1))
                if node.right:
                    stack.append((node.right, 0))
                if node.left:
                    stack.append((node.left, 0))
        return ans

    # 147
    # https://leetcode-cn.com/problems/insertion-sort-list/
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head:
            return head

        target = head
        head = head.next
        target.next = None
        dummy = ListNode(-1, target)
        while head:
            current_node = head
            head = head.next
            current_node.next = None
            pre = dummy
            target = pre.next
            while target:
                if current_node.val < target.val:
                    pre.next = current_node
                    current_node.next = target
                    break
                else:
                    pre = target
                    target = target.next
            if current_node.val >= pre.val:
                pre.next = current_node
        return dummy.next

    # 148
    # https://leetcode.com/problems/sort-list/submissions/
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
            if not head or not head.next:
                return head
            dummy = ListNode(-1, head)
            fast = slow = dummy
            while fast and fast.next:
                fast = fast.next.next
                slow = slow.next
            first = head
            second = slow.next
            slow.next = None
            first = self.sortList(first)
            second = self.sortList(second)
            dummy = pre = ListNode(-1)
            while first and second:
                if first.val < second.val:
                    pre.next = first
                    first = first.next
                else:
                    pre.next = second
                    second = second.next
                pre = pre.next
            pre.next = first or second
            return dummy.next

    # 149
    # https://leetcode.com/problems/max-points-on-a-line/submissions/
    def maxPoints(self, points: List[List[int]]) -> int:
        def getKey(x_1, y_1, x_2, y_2):
            if x_1 == x_2:
                return f'None-{x_1}'
            k = (y_1-y_2)/(x_1-x_2)
            b = (x_1*y_2-x_2*y_1)/(x_1-x_2)
            if k == 0:
                k = 0
            if b == 0:
                b = 0
            return f'{k}-{b}'
        if len(points) < 3:
            return len(points)

        point_dict = {}
        for index1 in range(len(points)):
            x_1, y_1 = points[index1]
            for index2 in range(index1+1, len(points)):
                x_2, y_2 = points[index2]
                key = getKey(x_1, y_1, x_2, y_2)
                if key not in point_dict:
                    point_dict[key] = set([(x_1, y_1), (x_2, y_2)])
                else:
                    point_dict[key].add((x_2, y_2))
        return max([len(item) for item in point_dict.values()])

    # 150
    # https://leetcode.com/problems/evaluate-reverse-polish-notation/submissions/
    def evalRPN(self, tokens: List[str]) -> int:
        def operation(o, n1, n2):
            return {'+': lambda x, y: x+y, '-':lambda x, y: x-y, '*': lambda x, y: x*y}[o](n1, n2)
        stack = []
        for token in tokens:
            if token not in ['+', '-', '*', '/']:
                stack.append(int(token))
            else:
                element2 = stack.pop()
                element1 = stack.pop()
                if token == '/':
                    sig = -1 if element1*element2<0 else 1
                    token = abs(element1)//abs(element2)*sig
                else:
                    token = operation(token, element1, element2)
                stack.append(token)
        return stack.pop()

    # 152
    def maxProduct(self, nums: List[int]) -> int:
        if not nums:
            return None
        max_prod = max(nums)
        pos = [0] if nums[0] < 0 else [nums[0]]
        neg = [0] if nums[0] > 0 else [nums[0]]
        for index in range(1, len(nums)):
            num = nums[index]
            if num >= 0:
                pos.append(max(pos[-1]*num, num))
                neg.append(neg[-1]*num)
            else:
                pos.append(neg[-1]*num)
                neg.append(min(num, pos[-2]*num))
            max_prod = max(max_prod, pos[-1])
        return max_prod


    # 198
    def rob(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return max(nums)
        dp = [0 for i in range(len(nums) + 1)]
        dp[1] = nums[0]
        dp[2] = nums[1]
        for i in range(2, len(nums)):
            dp[i + 1] = nums[i] + max(dp[i - 1], dp[i - 2])
        return max(dp[-1], dp[-2])