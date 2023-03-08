import heapq

from operator import itemgetter
from typing import List, Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_index = {}
        for index, num in enumerate(nums):
            if target - num in num_index:
                return [index, num_index[target - num]]
            else:
                num_index[num] = index
        return []

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        def count_nodes(node: Optional[ListNode]):
            count = 0
            while node:
                node = node.next
                count += 1
            return count

        count1 = count_nodes(l1)
        count2 = count_nodes(l2)
        if count1 < count2:
            l1, l2 = l2, l1
        dummy = l1
        add = 0
        pre = None
        while l1 and l2:
            l1.val += l2.val + add
            add = l1.val // 10
            l1.val %= 10
            pre = l1
            l1 = l1.next
            l2 = l2.next
        while l1:
            l1.val += add
            add = l1.val // 10
            l1.val %= 10
            pre = l1
            l1 = l1.next
        if add:
            pre.next = ListNode(add)
        return dummy

    def lengthOfLongestSubstring(self, s: str) -> int:
        max_length = 0
        letter_index = {}
        current_start = 0
        for index, letter in enumerate(s):
            if letter in letter_index:
                if letter_index[letter] < current_start:
                    letter_index[letter] = index
                    continue
                else:
                    max_length = max(max_length, index-current_start)
                    current_start = letter_index[letter] + 1
                    letter_index[letter] = index
            else:
                letter_index[letter] = index
        return max(max_length, len(s)-current_start)

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def findK(nums1, nums2, k):
            if len(nums1) > len(nums2):
                nums1, nums2 = nums2, nums1
            if len(nums1) == 0:
                return nums2[k-1]
            if k == 1:
                return min(nums1[0], nums2[0])
            index1 = min(k//2, len(nums1))
            index2 = k - index1
            if nums1[index1-1] == nums2[index2-1]:
                return nums1[index1-1]
            elif nums1[index1-1] < nums2[index2-1]:
                return findK(nums1[index1:], nums2, k-index1)
            else:
                return findK(nums1, nums2[index2:], k-index2)
        total_length = len(nums1) + len(nums2)
        if total_length % 2:
            return findK(nums1, nums2, total_length//2+1)
        else:
            return (findK(nums1, nums2, total_length//2) + findK(nums1, nums2, total_length//2+1))/2

    # https://leetcode.com/problems/longest-palindromic-substring/submissions/
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ''
        max_str = s[0]
        for index in range(len(s)):
            length = 1
            while index>=length and index+length<len(s) and s[index-length] == s[index+length]:
                t_max = s[index-length:index+length+1]
                if len(max_str) < len(t_max):
                    max_str = t_max
                length += 1
            length = 0
            while index>=length and index+1+length<len(s) and s[index-length] == s[index+length+1]:
                t_max = s[index-length:index+length+2]
                if len(max_str) < len(t_max):
                    max_str = t_max
                length += 1
        return max_str

    # https://leetcode.com/problems/zigzag-conversion/submissions/
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        arrs = ['' for _ in range(numRows)]
        step = 1
        index = 0
        for letter in s:
            arrs[index] += letter
            if step == 1 and index == numRows-1:
                step = -1
            elif step == -1 and index == 0:
                step = 1
            index += step
        return ''.join(arrs)

    #https://leetcode.com/problems/reverse-integer/submissions/
    def reverse(self, x: int) -> int:
        is_negative = 1 if x>0 else -1
        x = abs(x)
        ans = 0
        while x:
            ans = ans * 10 + x % 10
            x//=10
        ans *= is_negative
        if ans <= 2**31*-1 or ans >=2**31-1:
            return 0
        return ans

    # https://leetcode.com/problems/string-to-integer-atoi/submissions/
    def myAtoi(self, s: str) -> int:
        if not s:
            return 0
        s = s.strip()
        if not s:
            return 0
        signed = 1
        if s[0] == '+' or s[0] == '-':
            if s[0] == '-':
                signed = -1
            s = s[1:]
        if not s:
            return 0
        ans = 0
        for index, letter in enumerate(s):
            if ord(letter) >= ord('0') and ord(letter) <= ord('9'):
                ans = ans*10 + ord(letter) - ord('0')
            else:
                return ans*signed
            if ans*signed >=2**31-1:
                return 2**31-1
            if ans*signed <=2**31*-1:
                return 2**31*-1
        return ans*signed

    #https://leetcode.com/problems/palindrome-number/
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        t = x
        new_t = 0
        while t:
            new_t = new_t*10 + t %10
            t //= 10
        return new_t == x

    #https://leetcode.com/problems/regular-expression-matching/submissions/
    def isMatch(self, s: str, p: str) -> bool:
        dp = [[False for _ in range(len(s) + 1)] for _ in range(len(p) + 1)]
        dp[0][0] = True
        for index, p_l in enumerate(p):
            if p_l == '*':
                dp[index + 1][0] = dp[index - 1][0]
        for s_index, s_letter in enumerate(s):
            for p_index, p_letter in enumerate(p):
                if p_letter in [s_letter, '.']:
                    dp[p_index + 1][s_index + 1] = dp[p_index][s_index]
                elif p_letter == '*':
                    # dp[p_index][s_index+1] 表示*==1
                    # dp[p_index-1][s_index+1] 表示*==0
                    # dp[p_index+1][s_index-1] and p[p_index-1] in [s[s_index] or .]
                    dp[p_index + 1][s_index + 1] = dp[p_index][s_index + 1] or dp[p_index - 1][s_index + 1] or (
                                dp[p_index + 1][s_index] and p[p_index - 1] in [s_letter, '.'])
        return dp[-1][-1]

    # https://leetcode.com/problems/container-with-most-water/submissions/
    def maxArea(self, height: List[int]) -> int:
        max_area = 0
        left, right = 0, len(height)-1
        while left < right:
            max_area = max(max_area, (right-left) * min(height[left], height[right]))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area

    # https://leetcode.com/problems/integer-to-roman/submissions/
    def intToRoman(self, num: int) -> str:
        num_roman_map = {
            1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX',
            10: 'X', 20: 'XX', 30: 'XXX', 40: 'XL', 50: 'L', 60: 'LX', 70: 'LXX', 80: 'LXXX', 90: 'XC',
            100: 'C', 200: 'CC', 300: 'CCC', 400: 'CD', 500: 'D', 600: 'DC', 700: 'DCC', 800: 'DCCC', 900: 'CM',
            1000: 'M', 2000: 'MM', 3000: 'MMM'
        }
        ans = ''
        index = 1
        while num:
            target = num%10*index
            if target != 0:
                ans = num_roman_map[target] + ans
            num //= 10
            index *= 10
        return ans

    # https://leetcode.com/problems/roman-to-integer/submissions/
    def romanToInt(self, s: str) -> int:
        letter_num_dict = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
        }
        ans = 0
        for index, letter in enumerate(s):
            if index >= 1 and letter_num_dict[s[index - 1]] < letter_num_dict[letter]:
                ans -= 2 * letter_num_dict[s[index - 1]]
            ans += letter_num_dict[letter]
        return ans

    # https://leetcode.com/problems/longest-common-prefix/submissions/
    def longestCommonPrefix(self, strs: List[str]) -> str:
        prex = ''
        length = 0
        min_length = min(strs)
        for length in range(len(min_length)+1):
            for index, string in enumerate(strs):
                if index == 0:
                    prex = string[:length]
                else:
                    if prex == string[:length]:
                        continue
                    else:
                        return prex[:-1]
        return prex

    # https://leetcode.com/problems/3sum/submissions/
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)
        ans = []
        for target in range(len(nums)):
            before, after = 0, len(nums)-1
            while before < target and target < after:
                if nums[before] + nums[target] + nums[after] == 0:
                    if [nums[before], nums[target], nums[after]] not in ans:
                        ans.append([nums[before], nums[target], nums[after]])
                    before += 1
                    after -= 1
                    while nums[before] == nums[before-1] and before < target:
                        before += 1
                    while nums[after] == nums[after+1] and after > target:
                        after -= 1
                elif nums[before] + nums[target] + nums[after] > 0:
                    after -= 1
                else:
                    before += 1
        return list(ans)

    # https://leetcode.com/problems/3sum-closest/
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums = sorted(nums)
        ans = nums[0] + nums[1] + nums[2]
        for index in range(len(nums)):
            before, after = 0, len(nums) - 1
            while before < index and after > index:
                three_sum = nums[before] + nums[index] + nums[after]
                if abs(three_sum - target) <= abs(ans - target):
                    ans = three_sum
                if three_sum == target:
                    return ans
                elif three_sum > target:
                    after -= 1
                    while nums[after] == nums[after + 1] and after > index:
                        after -= 1
                else:
                    before += 1
                    while nums[before] == nums[before - 1] and before < index:
                        before += 1
        return ans

    # https://leetcode.com/problems/letter-combinations-of-a-phone-number/
    def letterCombinations(self, digits: str) -> List[str]:
        num_letter_dict = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }

        def traverse(digits, path):
            if not digits:
                self.result.append(path)
                return
            for element in num_letter_dict[digits[0]]:
                traverse(digits[1:], path+element)
        if not digits:
            return []
        self.result = []
        traverse(digits, '')
        return self.result

    # https://leetcode.com/problems/4sum/submissions/
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        for i in range(len(nums)-3):
            for j in range(i+1,len(nums)-2):
                left = j+1
                right = len(nums)-1
                while left<right :
                    currsum = nums[left] + nums[right] + nums[i] + nums[j]
                    if currsum == target:
                        if [nums[i], nums[j], nums[right], nums[left]] not in result:
                            result.append([nums[i], nums[j], nums[right], nums[left]])
                    if currsum<target:
                        left += 1
                    else:
                        right -= 1
        return result

    # https://leetcode.com/problems/remove-nth-node-from-end-of-list/submissions/
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        fast = slow = dummy
        while fast.next:
            fast = fast.next
            slow = slow.next
            n -= 1
            if n == 0:
                slow = dummy
        slow.next = slow.next.next
        return dummy.next

    # https://leetcode.com/problems/valid-parentheses/
    def isValid(self, s: str) -> bool:
        stack = []
        brackets = {'}':'{', ')': '(', ']': '['}
        for letter in s:
            if letter in ['(', '[', '{']:
                stack.append(letter)
            else:
                if stack:
                    left = stack.pop()
                    if brackets[letter] != left:
                        return False
                else:
                    return False
        return len(stack) == 0

    # https://leetcode.com/problems/merge-two-sorted-lists/
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = head = ListNode(-1)
        while l1 and l2:
            if l1.val < l2.val:
                dummy.next = l1
                l1 = l1.next
            else:
                dummy.next = l2
                l2 = l2.next
            dummy = dummy.next
        if l1 or l2:
            dummy.next = l1 or l2
        return head.next

    # https://leetcode.com/problems/generate-parentheses/submissions/
    # 22
    def generateParenthesis(self, n: int) -> List[str]:
        def generate(left, right, path):
            if left == 0:
                for i in range(right):
                    path += ')'
                self.result.append(path)
            else:
                generate(left-1, right, path+'(')
                if left < right:
                    generate(left, right-1, path+')')
        self.result = []
        generate(n, n, '')
        return self.result

    # https://leetcode.com/problems/merge-k-sorted-lists/submissions/
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        ls = []
        for l in lists:
            while l:
                ls.append(l)
                l = l.next
        ls.sort(key=lambda a: a.val, reverse=True)
        head = ListNode()
        dummy = head

        while ls:
            head.next = ls.pop()
            head = head.next
        head.next = None
        return dummy.next

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        stack = []
        for index, l in enumerate(lists):
            if l:
                heapq.heappush(stack, (l.val, index, l))
        dummy = ListNode()
        head = dummy
        while stack:
            _, index, l = heapq.heappop(stack)
            dummy.next = l
            dummy = dummy.next
            l = l.next
            if l:
                heapq.heappush(stack, (l.val, index, l))
        return head.next

    # https://leetcode.com/problems/swap-nodes-in-pairs/submissions/
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = pre = ListNode(-1)
        while head and head.next:
            first = head
            second = head.next
            head = second.next

            first.next = None
            second.next = None
            second.next = first
            dummy.next = second
            dummy = first
        if head:
            dummy.next = head
        return pre.next

    # https://leetcode.com/problems/reverse-nodes-in-k-group/submissions/
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def reverse(head):
            dummy = head
            tail = None
            while head:
                new_head = head.next
                head.next = tail
                tail = head
                head = new_head
            return tail, dummy

        if k == 1:
            return head
        dummy = pre = ListNode(-1)
        dummy.next = head
        count = 0
        while head:
            count += 1
            if count == k:
                new_head = head.next
                head.next = None
                reversed_head, reversed_tail = reverse(pre.next)
                pre.next = reversed_head
                reversed_tail.next = new_head
                count = 0
                pre = reversed_tail
                head = reversed_tail
            head = head.next
        return dummy.next

    # https://leetcode.com/problems/remove-duplicates-from-sorted-array/submissions/
    def removeDuplicates(self, nums: List[int]) -> int:
        target = 1
        for index in range(1, len(nums)):
            if nums[index] == nums[index-1]:
                continue
            else:
                nums[target] = nums[index]
                target += 1
        return target

    # https://leetcode.com/problems/remove-element/submissions/
    def removeElement(self, nums: List[int], val: int) -> int:
        target = 0
        for index, num in enumerate(nums):
            if num == val:
                continue
            else:
                nums[target] = num
                target += 1
        return target

    # https://leetcode.com/problems/divide-two-integers/
    def divide(self, dividend: int, divisor: int) -> int:
        quotient = 0
        signed = 1
        if dividend < 0:
            signed *= -1
            dividend = abs(dividend)
        if divisor < 0:
            signed *= -1
            divisor = abs(divisor)
        for i in reversed(range(32)):
            if (divisor << i) <= dividend:
                dividend -= (divisor << i)
                quotient += (1<<i)
        if signed > 0:
            return min(quotient, (1<<31)-1)
        else:
            return max(-1*quotient, -1<<31)

    # https://leetcode.com/problems/substring-with-concatenation-of-all-words/submissions/
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        word_int_dict = {}
        sum_total = 0
        mul_total = 1
        index = 1
        for word in words:
            if word not in word_int_dict:
                word_int_dict[word] = index
            sum_total += word_int_dict[word]
            mul_total *= word_int_dict[word]
            index += 1

        word_length = len(words[0])
        word_num = []
        for i in range(len(s) - word_length + 1):
            word = s[i:i + word_length]
            word_num.append(word_int_dict.get(word, 0))
        position = []
        for index in range(len(s) - len(words) * word_length + 1):
            tmp_sum = 0
            tmp_mul = 1
            start = index
            for i in words:
                tmp_sum += word_num[start]
                tmp_mul *= word_num[start]
                start += word_length
            if sum_total == tmp_sum and mul_total == tmp_mul:
                position.append(index)
        return position






    # https://leetcode.com/problems/ -valid-parentheses/submissions/
    def longestValidParentheses(self, s: str) -> int:
        is_valid = [False for i in s]
        stack = []
        for index, element in enumerate(s):
            if element == '(':
                stack.append(index)
            else:
                if stack:
                    pre_index = stack.pop()
                    is_valid[pre_index] = True
                    is_valid[index] = True
        max_count = 0
        tmp_count = 0
        for i in is_valid:
            if i:
                tmp_count += 1
            else:
                max_count = max(max_count, tmp_count)
                tmp_count = 0
        return max(max_count, tmp_count)

    # https://leetcode.com/problems/search-in-rotated-sorted-array/
    # 33
    def search(self, nums: List[int], target: int) -> int:
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return mid
            elif nums[start] < nums[mid]:
                if nums[start] <= target < nums[mid]:
                    end = mid-1
                else:
                    start = mid+1
            elif nums[start] > nums[mid]:
                if nums[mid] < target <= nums[end]:
                    start = mid+1
                else:
                    end = mid-1
            else:
                start += 1
        return -1

    # https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def findLeft(nums, target):
            start, end = 0, len(nums)-1
            ans = -1
            while start <= end:
                mid = start + (end-start) // 2
                if nums[mid] == target:
                    ans = mid
                    end = mid - 1
                elif nums[mid] < target:
                    start = mid + 1
                else:
                    end = mid - 1
            return ans

        def findRight(nums, target):
            start, end = 0, len(nums)-1
            ans = -1
            while start <= end:
                mid = start + (end-start) // 2
                if nums[mid] == target:
                    ans = mid
                    start = mid + 1
                elif nums[mid] < target:
                    start = mid + 1
                else:
                    end = mid - 1
            return ans

        return [findLeft(nums, target), findRight(nums, target)]

    # https://leetcode.com/problems/search-insert-position/
    def searchInsert(self, nums: List[int], target: int) -> int:
        start, end = 0, len(nums) - 1
        while start<=end:
            mid = start + (end-start) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                start = mid+1
            else:
                end = mid-1

        return start

    # https://leetcode.com/problems/valid-sudoku/
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        def validate(nums):
            num_list = []
            for num in nums:
                if num == '.':
                    continue
                else:
                    if num in num_list:
                        return False
                    else:
                        num_list.append(num)
            return True

        def validateRow():
            for i in range(9):
                if not validate(board[i]):
                    return False
            return True

        def validateCol():
            for i in range(9):
                col = [row[i] for row in board]
                if not validate(col):
                    return False
            return True

        def validateBlock():
            for i in range(3):
                for j in range(3):
                    block = []
                    for row in range(i * 3, i * 3 + 3):
                        for col in range(j * 3, j * 3 + 3):
                            block.append(board[row][col])
                            if not validate(block):
                                return False
            return True

        return validateRow() and validateCol() and validateBlock()





    # https://leetcode.com/problems/count-and-say/
    def countAndSay(self, n: int) -> str:
        start = '1'
        for i in range(n - 1):
            count = 0
            target = ''
            new_start = ''
            for index, element in enumerate(start):
                if index == 0:
                    count += 1
                    target = element
                else:
                    if element == target:
                        count += 1
                    else:
                        new_start += f'{count}{target}'
                        count = 1
                        target = element
            new_start += f'{count}{target}'
            start = new_start
        return start

    # https://leetcode.com/problems/combination-sum/
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def traverse(path, nums):
            if sum(path) == target:
                self.result.append(path)
            else:
                for index, num in enumerate(nums):
                    if sum(path) + num <= target:
                        traverse(path+[num], nums[index:])
        if not candidates or target < min(candidates):
            return []
        candidates = sorted(candidates)
        self.result = []
        traverse([], candidates)
        return self.result

    # https://leetcode.com/problems/combination-sum-ii/
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def traverse(nums, path):
            if sum(path) == target:
                self.result.append(path)
            else:
                for index, num in enumerate(nums):
                    if index>0 and num == nums[index-1]:
                        continue
                    if sum(path) + num <= target:
                        traverse(nums[index+1:], path+[num])
        if not candidates or target<min(candidates):
            return []
        candidates = sorted(candidates)
        self.result = []
        traverse(candidates, [])
        return self.result

    # https://leetcode.com/problems/first-missing-positive/submissions/
    def firstMissingPositive(self, nums: List[int]) -> int:
        index = 0
        while index < len(nums):
            num = nums[index]
            if num>0 and num<len(nums) and nums[num-1] != nums[index]:
                nums[num-1], nums[index] = nums[index], nums[num-1]
            else:
                index += 1
        for i in range(len(nums)):
            if nums[i] != i+1:
                return i+1
        return len(nums)+1

    # https://leetcode.com/problems/trapping-rain-water/submissions/
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        left, right = 0, len(height)-1
        left_wall, right_wall = height[left], height[right]
        area = 0
        # one side of the wall will always stand still until the other side is higher than current wall.
        while left <= right:
            if height[left] < height[right]:
                area += max(left_wall-height[left], 0)
                left_wall = max(left_wall, height[left])
                left += 1
            else:
                area += max(right_wall-height[right], 0)
                right_wall = max(right_wall, height[right])
                right -= 1
        return area

    # https://leetcode.com/problems/multiply-strings/submissions/
    def multiply(self, nums1: str, nums2: str) -> str:
        if len(nums1) == 1 and int(nums1) == 0 or len(nums2) == 1 and int(nums2) == 0:
            return '0'
        result = [0 for _ in range(len(nums1) + len(nums2))]
        nums1 = nums1[::-1]
        nums2 = nums2[::-1]
        for index1, num1 in enumerate(nums1):
            for index2, num2 in enumerate(nums2):
                result[index1+index2] += int(num1)*int(num2)
        add = 0
        for i in range(len(result)):
            result[i] += add
            add = result[i] // 10
            result[i] = str(result[i] % 10)
        result = ''.join(result[::-1])
        if result[0] == '0':
            return result[1:]
        return result

    # https://leetcode.com/problems/wildcard-matching/
    def isMatch(self, s: str, p: str) -> bool:
        dp = [[False for _ in range(len(p)+1)] for _ in range(len(s)+1)]
        s_index, p_index = 0, 0
        dp[0][0] = True
        for index, p_l in enumerate(p):
            if p_l == '*':
                dp[0][index + 1] = dp[0][index]
        for s_i, s_e in enumerate(s):
            for p_i, p_e in enumerate(p):
                if s_e == p_e or p_e == '?':
                    dp[s_i+1][p_i+1] = dp[s_i][p_i]
                elif p_e == '*':
                    dp[s_i+1][p_i+1] = dp[s_i][p_i] or dp[s_i+1][p_i] or dp[s_i][p_i+1]
        return dp[-1][-1]

    # https://leetcode.com/problems/jump-game-ii/
    def jump(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return 0
        count = 1
        current_max = nums[0]
        next_max = nums[0]
        if next_max >= len(nums):
            return count
        for index in range(1, len(nums)):
            if index <= current_max:
                next_max = max(next_max, nums[index]+index)
            else:
                count += 1
                current_max = next_max
                next_max = max(current_max, index + nums[index])
                if current_max >= len(nums):
                    return count
        return count

    # https://leetcode.com/problems/permutations/submissions/
    def permute(self, nums: List[int]) -> List[List[int]]:
        def traverse(nums, path):
            if not nums:
                self.result.append(path)
            else:
                for index, num in enumerate(nums):
                    traverse(nums[:index]+nums[index+1:], path+[num])
        self.result = []
        if not nums:
            return self.result
        traverse(nums, [])
        return self.result

    # https://leetcode.com/problems/permutations-ii/
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def traverse(nums, path):
            if not nums:
                self.result.append(path)
            else:
                for index, num in enumerate(nums):
                    if index > 0 and num == nums[index - 1]:
                        continue
                    else:
                        traverse(nums[:index] + nums[index + 1:], path + [num])

        if not nums:
            return []
        nums = sorted(nums)
        self.result = []
        traverse(nums, [])
        return self.result

    # https://leetcode.com/problems/rotate-image/
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """

        size = len(matrix)
        for i in range(size):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        for i in range(size):
            for j in range(size // 2):
                matrix[i][j], matrix[i][size-1-j] = matrix[i][size-1-j], matrix[i][j]

    # https://leetcode-cn.com/problems/group-anagrams/submissions/
    # 49
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        word_dict = {}
        for word in strs:
            key = ''.join(sorted(word))
            try:
                word_dict[key].append(word)
            except:
                word_dict[key] = [word]
        return list(word_dict.values())

    # https://leetcode.com/problems/powx-n/
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        elif x == 0:
            return 0
        elif n % 2 == 0:
            return self.myPow(x*x, n//2)
        elif n > 0:
            return x * self.myPow(x*x, (n-1)//2)
        else:
            return 1.0/x * self.myPow(1.0/(x*x), abs(n+1)//2)

    # https://leetcode.com/problems/n-queens/submissions/
    def solveNQueens(self, n: int) -> List[List[str]]:
        def is_valid_position(x, y, positions):
            for p_x, p_y in positions:
                if p_x == x or p_y == y or x + y == p_x + p_y or x - y == p_x - p_y:
                    return False
            return True

        def dfs(positions, level):
            if level == -1:
                draw_position(positions)
                return

            for i in range(n):
                if is_valid_position(i, level, positions):
                    dfs(positions + [(i, level)], level - 1)

        def draw_position(positions):
            ans = ['' for i in range(n)]
            for p_x, p_y in positions:
                line = ''
                for i in range(n):
                    if i == p_y:
                        line += 'Q'
                    else:
                        line += '.'
                ans[p_x] = line
            self.result.append(ans)

        self.result = []
        dfs([], n - 1)
        return self.result

    # https://leetcode.com/problems/n-queens-ii/submissions/
    def totalNQueens(self, n: int) -> int:
        def is_valid_position(x, y, positions):
            for p_x, p_y in positions:
                if p_x == x or p_y == y or x + y == p_x + p_y or x - y == p_x - p_y:
                    return False
            return True

        def dfs(positions, level):
            if level == -1:
                self.result += 1
                return

            for i in range(n):
                if is_valid_position(i, level, positions):
                    dfs(positions + [(i, level)], level - 1)

        self.result = 0
        dfs([], n - 1)
        return self.result

    # https://leetcode.com/problems/maximum-subarray/submissions/
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        current = nums[0]
        for index in range(1, len(nums)):
            if nums[index] > 0:
                if current < 0:
                    current = nums[index]
                else:
                    current += nums[index]
            else:
                if current + nums[index] >= 0:
                    current += nums[index]
                else:
                    current = nums[index]
            max_sum = max(max_sum, current)
        return max_sum

    # https://leetcode.com/problems/spiral-matrix/submissions/
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        n, m = len(matrix[0]), len(matrix)
        left, right, top, bottom = 0, len(matrix[0]), 0, len(matrix)
        ans = []
        while left <= right and top <= bottom:
            for i in range(left, right):
                ans.append(matrix[top][i])
            top += 1
            for i in range(top, bottom):
                ans.append(matrix[i][right-1])
            right -= 1
            for i in reversed(range(left, right)):
                ans.append(matrix[bottom-1][i])
            bottom -= 1
            for i in reversed(range(top, bottom)):
                ans.append(matrix[i][left])
            left += 1
        return ans[:n*m]

    # https://leetcode.com/problems/jump-game/submissions/
    def canJump(self, nums: List[int]) -> bool:
        step = [False for num in nums]
        step[0] = True
        for index, num in enumerate(nums):
            if step[index] == False:
                return False
            else:
                for i in range(index, min(len(nums), index + num + 1)):
                    step[i] = True

                if step[-1]:
                    return True
        return step[-1]

    # https://leetcode.com/problems/merge-intervals/
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return intervals
        intervals = sorted(intervals, key=itemgetter(0))
        start, end = 0, 0
        result = []
        for index, (x_s, x_e) in enumerate(intervals):
            if index == 0:
                start, end = x_s, x_e
            else:
                if x_s <= end or x_e <= end:
                    end = max(end, x_e)
                else:
                    result.append([start, end])
                    start, end = x_s, x_e
        result.append([start, end])
        return result

    # https://leetcode.com/problems/insert-interval/submissions/
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval)
        intervals = sorted(intervals, key=itemgetter(0))
        start, end = 0, 0
        result = []
        for index, (x_s, x_e) in enumerate(intervals):
            if index == 0:
                start, end = x_s, x_e
            else:
                if x_s <= end or x_e <= end:
                    end = max(end, x_e)
                else:
                    result.append([start, end])
                    start, end = x_s, x_e
        result.append([start, end])
        return result

    # https://leetcode.com/problems/length-of-last-word/submissions/
    def lengthOfLastWord(self, s: str) -> int:
        lists = s.split()
        if lists == []:
            return 0
        else:
            return len(lists[-1])

    # https://leetcode.com/problems/spiral-matrix-ii/submissions/
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0 for i in range(n)] for i in range(n)]
        left, right, top, bottom = 0, n, 0, n
        ans = []
        index = 1
        while left <= right and top <= bottom:
            for i in range(left, right):
                matrix[top][i] = index
                index += 1
                if index > n*n:
                    break
            top += 1
            for i in range(top, bottom):
                matrix[i][right-1] = index
                index += 1
                if index > n*n:
                    break
            right -= 1
            for i in reversed(range(left, right)):
                matrix[bottom-1][i]  = index
                index += 1
                if index > n*n:
                    break
            bottom -= 1
            for i in reversed(range(top, bottom)):
                matrix[i][left] = index
                index += 1
                if index > n*n:
                    break
            left += 1
        return matrix





    # https://leetcode.com/problems/rotate-list/submissions/
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if k == 0 or head == None:
            return head
        current = head
        count = 1
        while current.next:
            count += 1
            current = current.next
        current.next = head
        k = count - k%count
        while k:
            current = current.next
            k -= 1
        head = current.next
        current.next = None
        return head

    # https://leetcode-cn.com/problems/unique-paths/submissions/
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1 for i in range(n)] for j in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]

    # https://leetcode.com/problems/unique-paths-ii/submissions/
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0 for i in range(n)] for j in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            else:
                dp[i][0] = 1
        for i in range(n):
            if obstacleGrid[0][i] == 1:
                break
            else:
                dp[0][i] = 1
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    continue
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]

    # https://leetcode.com/problems/minimum-path-sum/submissions/
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        for i in range(1, m):
            grid[i][0] += grid[i-1][0]

        for i in range(1, n):
            grid[0][i] += grid[0][i-1]

        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        return grid[-1][-1]

    # https://leetcode.com/problems/plus-one/submissions/
    def plusOne(self, digits: List[int]) -> List[int]:
        digits = digits[::-1]
        add = 1
        for i in range(len(digits)):
            digits[i] += 1
            add = digits[i] // 10
            digits[i] %= 10
            if add == 0:
                break
        if add == 1:
            digits.append(add)
        return digits[::-1]

    # https://leetcode.com/problems/add-binary/submissions/
    def addBinary(self, a: str, b: str) -> str:
        a = list(a)
        b = list(b)
        a = a[::-1]
        b = b[::-1]
        add = '0'
        if len(a) < len(b):
            a, b = b, a
        for i in range(len(b)):
            if a[i] == b[i] == '1':
                if add == '1':
                    a[i] = '1'
                else:
                    a[i] = '0'
                add = '1'
            elif a[i] == '1' or b[i] == '1':
                if add == '1':
                    a[i] = '0'
                    add = '1'
                else:
                    a[i] = '1'
                    add = '0'
            else:
                a[i] = add
                add = '0'
        for i in range(len(b), len(a)):
            if a[i] == add == '1':
                add = '1'
                a[i] = '0'
            elif a[i] == '1' or add == '1':
                add = '0'
                a[i] = '1'
                break
            else:
                break
        if add == '1':
            a += '1'
        a = a[::-1]
        return ''.join(a)

    # https://leetcode.com/problems/text-justification/submissions/
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        page = []
        line = []
        for word in words:
            if sum([len(w) for w in line]) + len(line) + len(word) <= maxWidth:
                line.append(word)
            else:
                page.append(line)
                line = [word]
        page.append(line)
        paragraph = []
        for index, line in enumerate(page):

            if index == len(page) - 1:
                justify_line = ' '.join(line)
                for i in range(maxWidth - len(justify_line)):
                    justify_line += ' '
                paragraph.append(justify_line)
                break
            word_length = sum([len(w) for w in line])
            word_count = len(line)
            space_count = maxWidth - word_length
            if word_count == 1:
                justify_line = line[0]
                for i in range(space_count):
                    justify_line += ' '
                paragraph.append(justify_line)
            else:
                average_white_space = space_count // (word_count - 1)
                remaining_white_space = space_count % (word_count - 1)
                justify_line = ''
                for index, word in enumerate(line):
                    justify_line += word
                    if index == len(line) - 1:
                        paragraph.append(justify_line)
                    else:
                        for i in range(average_white_space):
                            justify_line += ' '
                        if remaining_white_space:
                            justify_line += ' '
                            remaining_white_space -= 1

        return paragraph

    # https://leetcode.com/problems/sqrtx/submissions/
    def mySqrt(self, x: int) -> int:
        if x <= 1:
            return x
        start, end = 1, x//2 + 1
        ans = 0
        while start <= end:
            mid = start + (end-start) // 2
            if mid * mid == x:
                return mid
            elif mid*mid < x:
                ans = mid
                start = mid + 1
            else:
                end = mid - 1
        return ans

    # https://leetcode.com/problems/climbing-stairs/submissions/
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2
        dp = [0 for i in range(n+1)]
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]

    # https://leetcode.com/problems/simplify-path/submissions/
    def simplifyPath(self, path: str) -> str:
        parts = path.split('/')
        stack = []
        for part in parts:
            if not part:
                continue
            elif part == '.':
                continue
            elif part == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(part)
        return '/'+'/'.join(stack)

    # https://leetcode.com/problems/edit-distance/submissions/
    def minDistance(self, word1: str, word2: str) -> int:
        len1, len2 = len(word1), len(word2)
        dp = [[0 for i in range(len2+1)] for i in range(len1+1)]
        for i in range(len1+1):
            dp[i][0] = i
        for i in range(len2+1):
            dp[0][i] = i
        for i1, e1 in enumerate(word1):
            for i2, e2 in enumerate(word2):
                if e1 == e2:
                    dp[i1+1][i2+1] = dp[i1][i2]
                else:
                    dp[i1+1][i2+1] = min(dp[i1][i2], dp[i1+1][i2], dp[i1][i2+1]) + 1
        return dp[-1][-1]

    # https://leetcode.com/problems/set-matrix-zeroes/submissions/
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        position = []
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    position.append((i, j))
        for (x, y) in position:
            for i in range(m):
                matrix[i][y] = 0
            for i in range(n):
                matrix[x][i] = 0

    # https://leetcode.com/problems/search-a-2d-matrix/submissions/
    # 74
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        top, right = 0, n-1
        while top < m and right >= 0:
            if matrix[top][right] == target:
                return True
            elif matrix[top][right] < target:
                top += 1
            else:
                right -= 1
        return False

    # https://leetcode.com/problems/sort-colors/submissions/
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        zero, one, two = 0, 0, len(nums)-1
        while one <= two:
            if nums[one] == 0:
                nums[one], nums[zero] = nums[zero], nums[one]
                one += 1
                zero += 1
            elif nums[one] == 1:
                one += 1
            else:
                nums[one], nums[two] = nums[two], nums[one]
                two -= 1




    # https://leetcode.com/problems/combinations/submissions/
    def combine(self, n: int, k: int) -> List[List[int]]:
        def traverse(nums, k, path):
            if k == 0:
                self.result.append(path)
            else:
                for index, num in enumerate(nums):
                    traverse(nums[index+1:], k-1, path+[num])
        nums = [i for i in range(1, n+1)]
        self.result = []
        traverse(nums, k, [])
        return self.result

    # https://leetcode.com/problems/subsets/submissions/
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def traverse(nums, path):
            self.result.append(path)
            for index, num in enumerate(nums):
                traverse(nums[index+1:], path+[num])
        self.result = []
        traverse(nums, [])
        return self.result

    # https://leetcode.com/problems/word-search/submissions/
    def exist(self, board: List[List[str]], word: str) -> bool:
        X, Y = len(board), len(board[0])
        def dfs(x, y, word, used):
            if not word:
                return True
            if x+1 < X:
                if (x+1, y) not in used and board[x+1][y] == word[0]:
                    result = dfs(x+1, y, word[1:], used+[(x+1, y)])
                    if result:
                        return result
            if x-1 >= 0:
                if (x-1, y) not in used and board[x-1][y] == word[0]:
                    result = dfs(x-1, y, word[1:], used+[(x-1, y)])
                    if result:
                        return result
            if y+1 < Y:
                if (x, y+1) not in used and board[x][y+1] == word[0]:
                    result = dfs(x, y+1, word[1:], used+[(x, y+1)])
                    if result:
                        return result
            if y-1 >= 0:
                if (x, y-1) not in used and board[x][y-1] == word[0]:
                    result = dfs(x, y-1, word[1:], used+[(x, y-1)])
                    if result:
                        return result
        for x in range(X):
            for y in range(Y):
                if board[x][y] == word[0]:
                    result = dfs(x, y, word[1:], [(x, y)])
                    if result:
                        return True
        return False

    # https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/submissions/
    def removeDuplicates(self, nums: List[int]) -> int:
        count = 0
        start = 0
        for index, num in enumerate(nums):
            if index == 0:
                count += 1
                start += 1
            else:
                if num == nums[index-1]:
                    count += 1
                    if count >= 3:
                        continue
                    else:
                        nums[start] = nums[index]
                        start += 1
                else:
                    nums[start] = nums[index]
                    count = 1
                    start += 1
        return start

    # https://leetcode.com/problems/search-in-rotated-sorted-array-ii/submissions/
    # 81
    def search(self, nums: List[int], target: int) -> bool:
        start, end = 0, len(nums)-1
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return True
            elif nums[start] < nums[mid]:
                if nums[start] <= target < nums[mid]:
                    end = mid-1
                else:
                    start = mid+1
            elif nums[start] > nums[mid]:
                if nums[mid] < target <= nums[end]:
                    start = mid+1
                else:
                    end = mid-1
            else:
                start += 1
        return False

    # https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/submissions/
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        pre = dummy
        while head:
            if head.next and head.val == head.next.val:
                while head.next and head.next.val == head.val:
                    head.next = head.next.next
                pre.next = head.next
                head = head.next
            else:
                pre = head
                head = head.next
        return dummy.next

    # https://leetcode.com/problems/remove-duplicates-from-sorted-list/submissions/
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = head
        while head:
            if head.next and head.next.val == head.val:
                head.next = head.next.next
            else:
                head = head.next
        return dummy

    # https://leetcode.com/problems/largest-rectangle-in-histogram/submissions/
    def largestRectangleArea(self, heights: List[int]) -> int:
        if not heights:
            return 0
        stack = []
        heights.append(-1)
        area = 0
        for index, height in enumerate(heights):
            while stack and heights[stack[-1]] > height:
                h = heights[stack.pop()]
                if stack:
                    w = index - 1 - stack[-1]
                else:
                    w = index
                area = max(area, h*w)
            stack.append(index)
        return area

    # https://leetcode.com/problems/maximal-rectangle/submissions/
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        def maxArea(heights, area):
            if not heights:
                return 0
            stack = []
            heights.append(-1)
            for index, height in enumerate(heights):
                while stack and heights[stack[-1]] > height:
                    h = heights[stack.pop()]
                    w = index - stack[-1] - 1 if stack else index
                    area = max(area, h*w)
                stack.append(index)
            heights.pop()
            return area

        if not matrix:
            return 0
        X, Y = len(matrix), len(matrix[0])
        histograms = [[0 for i in range(Y)] for j in range(X)]
        for i in range(0, Y):
            if matrix[0][i] == '1':
                histograms[0][i] = 1
        for i in range(1, X):
            if matrix[i][0] == '1':
                histograms[i][0] = 1 + histograms[i-1][0]
        for i in range(1, X):
            for j in range(1, Y):
                if matrix[i][j] == '1':
                    histograms[i][j] = histograms[i-1][j] + 1
        area = 0
        for heights in histograms:
            area = maxArea(heights, area)
        return area

    # https://leetcode.com/problems/partition-list/submissions/
    def partition(self, head: ListNode, x: int) -> ListNode:
        pre, suf = ListNode(-1), ListNode(-1)
        dummy = pre
        summy = suf
        while head:
            if head.val < x:
                pre.next = head
                head = head.next
                pre = pre.next
                pre.next = None
            else:
                suf.next = head
                head = head.next
                suf = suf.next
                suf.next = None
        pre.next = summy.next
        return dummy.next



    # https://leetcode.com/problems/merge-sorted-array/submissions/
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        index = m+n-1
        while m > 0 and n>0:
            if nums1[m-1] > nums2[n-1]:
                nums1[index] = nums1[m-1]
                m -= 1
            else:
                nums1[index] = nums2[n-1]
                n -= 1
            index -= 1
        for i in range(n):
            nums1[i] = nums2[i]



    # https://leetcode.com/problems/subsets-ii/submissions/
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def traverse(nums, path):
            self.result.append(path)
            for index, num in enumerate(nums):
                if index > 0 and num == nums[index-1]:
                    continue
                else:
                    traverse(nums[index+1:], path+[num])
        nums = sorted(nums)
        self.result = []
        traverse(nums, [])
        return self.result


    # https://leetcode.com/problems/decode-ways/submissions/
    def numDecodings(self, s: str) -> int:
        dp = [0 for _ in range(len(s)+1)]
        dp[0] = 1
        for i, e in enumerate(s):
            if i == 0:
                if e == '0':
                    return 0
                else:
                    dp[i+1] += dp[i]
            else:
                if e == '0':
                    if s[i-1] in ['1', '2']:
                        dp[i+1] = dp[i-1]
                    else:
                        return 0
                else:
                    dp[i+1] += dp[i]
                    if int(s[i-1:i+1]) <= 26 and int(s[i-1:i+1])>9:
                        dp[i+1] += dp[i-1]
        return dp[-1]

    # https://leetcode.com/problems/reverse-linked-list-ii/submissions/
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        def reverse(head, count):
            pre = None
            node = head
            while count and node:
                next_node = node.next
                node.next = pre
                pre = node
                node = next_node
                count -= 1
            head.next = node
            return pre
        count = n-m+1
        dummy = ListNode(1)
        dummy.next = head
        current = dummy
        while current:
            m -= 1
            if m == 0:
                current.next = reverse(current.next, count)
                break
            current = current.next
        return dummy.next

    # https://leetcode.com/problems/restore-ip-addresses/
    def restoreIpAddresses(self, s: str) -> List[str]:
        def traverse(s, k, path):
            if len(s) == 0 and k == 0 and path[1:] not in self.ans:
                self.ans.append(path[1:])
            elif len(s) == 0 or k == 0:
                return
            else:
                if len(s) < k or len(s) > 3 * k:
                    return
                elif len(s) == k:
                    traverse(s[1:], k - 1, path + '.' + s[0])
                elif len(s) == 3 * k:
                    if int(s[:3]) > 99 and int(s[:3]) < 256:
                        traverse(s[3:], k - 1, path + '.' + s[:3])
                elif s[0] == '0':
                    traverse(s[1:], k - 1, path + '.' + '0')
                else:
                    for i in range(2):
                        traverse(s[i + 1:], k - 1, path + '.' + s[:i + 1])
                    if (int(s[:3])) < 256:
                        traverse(s[3:], k - 1, path + '.' + s[:3])

        self.ans = []
        traverse(s, 4, "")
        return self.ans


    # https://leetcode.com/problems/binary-tree-inorder-traversal/submissions/
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        stack = [root]
        ans = []
        while stack:
            node = stack.pop()
            if node.right:
                stack.append(node.right)
                node.right = None
            if node.left:
                left, node.left = node.left, None
                stack.append(node)
                stack.append(left)
            else:
                ans.append(node.val)
        return ans

    # https://leetcode.com/problems/unique-binary-search-trees-ii/submissions/
    def generateTrees(self, n: int) -> List[TreeNode]:
        def traverse(nums):
            if not nums:
                return [None]
            else:
                trees = []
                for index, num in enumerate(nums):
                    lefts = traverse(nums[:index])
                    rights = traverse(nums[index+1:])
                    for left in lefts:
                        for right in rights:
                            node = TreeNode(num)
                            node.left = left
                            node.right = right
                            trees.append(node)
                return trees
        nums = [i for i in range(1, n+1)]
        return traverse(nums)

    # https://leetcode.com/problems/unique-binary-search-trees/submissions/
    def numTrees(self, n: int) -> int:
        dp = [1, 1, 2]
        if n <= 2:
            return dp[n]
        for i in range(3, n+1):
            value = 0
            for j in range(i):
                value += dp[j] * dp[i-1-j]
            dp.append(value)
        return dp[-1]

    # https://leetcode-cn.com/problems/interleaving-string/submissions/
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        dp = [[False for i in range(len(s2)+1)] for i in range(len(s1)+1)]
        dp[0][0] = True
        for i in range(len(s1)):
            if s1[i] == s3[i]:
                dp[i+1][0] = True
            else:
                break
        for i in range(len(s2)):
            if s2[i] == s3[i]:
                dp[0][i+1] = True
            else:
                break
        for i in range(len(s1)):
            for j in range(len(s2)):
                print(i, j, i+j+1)
                print(s1[i], s2[j], s3[i+j+1])
                dp[i+1][j+1] = dp[i+1][j] and s3[i+j+1]  == s2[j] or dp[i][j+1] and s3[i+j+1] == s1[i]
        return dp[-1][-1]

    # https://leetcode.com/problems/validate-binary-search-tree/submissions/
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def isValid(node, minimal, maximal):
            if minimal is not None and node.val <= minimal:
                return False
            if maximal is not None and node.val >= maximal:
                return False

            if node.left:
                left_result = isValid(node.left, minimal, node.val)
            else:
                left_result = True
            if not left_result:
                return False

            if node.right:
                right_result = isValid(node.right, node.val, maximal)
            else:
                right_result = True
            if not right_result:
                return False
            return True

        return isValid(root, None, None)

    # https://leetcode.com/problems/recover-binary-search-tree/submissions/
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        def findLeft(root):
            stack = [(root, 0)]
            pre = None
            while stack:
                node, count = stack.pop()
                if count == 1:
                    if not pre:
                        pre = node
                    elif node.val > pre.val:
                        pre = node
                    else:
                        return pre
                else:
                    if node.right:
                        stack.append((node.right, 0))
                    stack.append((node, 1))
                    if node.left:
                        stack.append((node.left, 0))

        def findRight(root):
            stack = [(root, 0)]
            pre = None
            while stack:
                node, count = stack.pop()
                if count == 1:
                    if not pre:
                        pre = node
                    elif node.val < pre.val:
                        pre = node
                    else:
                        return pre
                else:
                    if node.left:
                        stack.append((node.left, 0))
                    stack.append((node, 1))
                    if node.right:
                        stack.append((node.right, 0))

        node1 = findLeft(root)
        node2 = findRight(root)
        node1.val, node2.val = node2.val, node1.val

    # https://leetcode.com/problems/same-tree/submissions/
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        elif not p or not q:
            return False
        elif p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False
