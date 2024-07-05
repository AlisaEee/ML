import numpy as np
def most_frequent(nums):
    nums.sort()
    last = nums[0]
    ans = 0
    cnt = 1
    ans_cnt = 1
    for i in range(1,len(nums)):
        if nums[i] == last:
            cnt+=1
        elif cnt > ans_cnt:
            ans_cnt = cnt
            ans = last
            cnt = 1
        else:
            cnt = 1
        last = nums[i]
    if cnt > ans_cnt:
        ans_cnt = cnt
        ans = last
    return ans
print(most_frequent(np.array([1,2,3,3,3,4,2,2,3])))
