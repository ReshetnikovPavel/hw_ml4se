# Importing the sys module to use sys.stdin for reading input
import sys
# Defining a function to calculate the factorial of a number
def factorial(n):
    """
    Calculate the factorial of a number.
    
    Parameters:
    n (int): The number to calculate the factorial of.
    
    Returns:
    int: The factorial of the number.
    """
    if n == 0: # Base case: the factorial of 0 is 1
        return 1
    else:
        return n * factorial(n-1) # Recursive call
# Defining a function to check if a number is prime
def is_prime(n):
    """
    Check if a number is prime.
    
    Parameters:
    n (int): The number to check.
    
    Returns:
    bool: True if the number is prime, False otherwise.
    """
    if n <= 1: # Numbers less than or equal to 1 are not prime
        return False
    for i in range(2, n):
        if n % i == 0: # If n is divisible by any number other than 1 and itself, it's not prime
            return False
    return True # If no divisors are found, the number is prime
# Main program
if __name__ == "__main__":
    # Reading a number from standard input
    print("Enter a number:")
    num = int(input())
    
    # Calculating and printing the factorial of the number
    print(f"The factorial of {num} is {factorial(num)}")
    
    # Checking if the number is prime and printing the result
    if is_prime(num):
        print(f"{num} is a prime number.")
    else:
        print(f"{num} is not a prime number.")
