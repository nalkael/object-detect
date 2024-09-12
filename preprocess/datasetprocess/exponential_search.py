import os

def exponential_search(A: list[int], x: int) -> int:

    # expand expontentially until A[i] >= x or reach the end of the array
    i = 1; n = len(A)
    while i < n and A[i] < x:
        i = i * 2

    # Now A[x] should in between A[i//2] and A[min(i, n-1)] 
    low = i // 2
    high = min(i, n - 1)

    # Step 2: Binary Search within the identified range
    while low <= high:
        mid = (low + high) // 2
        if A[mid] == x:
            return mid
        elif A[mid] > x:
            high = mid -1
        else:
            low = mid + 1
    
    return -1

def main():

    A = [1,2,9,13,19,27,32,47,59,61,72,85,99,103,127,133,141,156]
    x = 1

    print(exponential_search(A, x))

if __name__ == '__main__':
    main()

    
    