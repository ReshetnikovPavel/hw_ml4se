from typing import List, Optional
from llama_cpp import Llama, BaseLlamaTokenizer
from transformers import GemmaTokenizer
from itertools import combinations

from datasets import load_dataset, concatenate_datasets
import pandas as pd


class GemmaTokenizerForLLamaCpp(BaseLlamaTokenizer):
    def __init__(self):
        self.tokenizer = GemmaTokenizer.from_pretrained("google/codegemma-7b-it")

    def tokenize(
        self, text: bytes, add_bos: bool = True, special: bool = True
    ) -> List[int]:
        return self.tokenizer.encode(text.decode('utf-8'))

    def detokenize(
        self, tokens: List[int], prev_tokens: Optional[List[int]] = None
    ) -> bytes:
        return self.tokenizer.decode(tokens).encode('utf-8')


path = "/home/pavelresh/college/ml4se/hw_ml4se/hw7/codegemma-1.1-7b-it-Q4_K_M.gguf"

tokenizer = GemmaTokenizerForLLamaCpp()
llm = Llama(model_path=path,
            chat_format="gemma",
            tokenizer=tokenizer,
            verbose=True,
            n_ctx=8192)


def are_clones(func1, func2):
    system_prompt = """You are a helpful assistent who is proficient in programming in java."""
#     prompt = """
# Given two code snippets, write an analysis and determine if they are code clones. A code clone is a piece of code that appears in two or more places in a software system. To do this, follow these steps:
#
# 1. Compare the structure of both snippets.
# 2. Look for similar variable names, function names, and overall logic.
# 3. Consider if the snippets perform the same operation despite differences in variable names or minor syntactical changes.
# 4. At the end of your analysis write `[Yes]` if provided snippets are clones and `[No]` otherwise.
#
# Example 1:
# - Snippet 1: `public int add(int a, int b) { return a + b; }`
# - Snippet 2: `public int sum(int x, int y) { return x + y; }`
#
# - *Analysis* The structure of these snippets is identical, only names differ. And these snippets perform exactly the same operation of getting a sum of two numbers. So, the answer is: Yes, these are code clones. [Yes]
#
# Example 2:
# - Snippet 1: `public void calculateFactorial(int n) { int result = 1; for (int i = 1; i <= n; i++) { result *= i; } System.out.println("Factorial of " + n + " is " + result); }`
# - Snippet 2: `public void findPrimeNumbers(int limit) { for (int i = 2; i <= limit; i++) { boolean isPrime = true; for (int j = 2; j < i; j++) { if (i % j == 0) { isPrime = false; break; } } if (isPrime) { System.out.println(i + " is a prime number."); } } }`
#
# - *Analysis* Both functions implement different algorithms, so the structure is different. Variable names are mostly different. The first function states that it calculates an nth factorial and the other one finds prime numbers to some limit. The operation they perform is very different. So, the answer is: No, these are not code clones. [No]
#
# Now, analyze the following two code snippets:
#
# """
#     f"""- Snippet A: `{func1}`
#     - Snippet B: `{func2}`
#
#     Are Snippet A and Snippet B code clones?
#     """

    # prompt = f"Are these functions type 1, type 2, type 3 or type 4 clones? `{func1}` `{func2}` Answer yes or no"
#     prompt = ("""There are 4 types of code clones:
#         - Type 1: two code snippets that are exactly the same except for formatting or comments
#         - Type 2: two code snippets that are exactly the same except for different variable or function names
#         - Type 3: two code snippets that are very similar but some statements were added or removed. Or two snippets that contain the same code fragments.
#         - Type 4: two code fragments that implement semantically very similar functionality in them, but the structure may be very different.
#
#         Please provide a detailed reasoning process for detecting code clones in the following two code snippets. Based on your analysis,
# respond with `yes` if the code snippets are clones or `no` if they are not""" +
    prompt = ("Are there big chunks of code in these two snippets that implement the same thing?" + 
    f"""
    - Snippet 1: `{func1}`
    - Snippet 2: `{func2}`
    """)
    messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": prompt
                },
               ]
    output = llm.create_chat_completion(messages=messages)
    # print(output)
    message = output['choices'][0]['message']
    content = message['content']
    yes = ["Yes", "yes"]
    no = ["No", "no"]
    if any(word in content for word in yes):
        return True, content
    if any(word in content for word in no):
        return False, content
    return None, content


# dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")
# test = dataset['test'].filter(lambda example: not example['label'])
#
# for i in range(10):
#     func1 = test['func1'][i]
#     func2 = test['func2'][i]
#     print(func1, func2)
#     answer = are_clones(func1, func2)
#
#     print(answer, test['label'][i])

type2_clones = [
    {
        "func1":  """
public double calculateArea(double width, double height) {
    return width * height;
}
            """,
        "func2":  """
public double computeArea(double length, double breadth) {
    return length * breadth;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public int findMax(int a, int b) {
    return (a > b) ? a : b;
}
            """,
        "func2":  """
public int getMax(int x, int y) {
    return (x > y) ? x : y;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public boolean isPrime(int num) {
    for (int i = 2; i < num; i++) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}
            """,
        "func2":  """
public boolean checkPrime(int number) {
    for (int j = 2; j < number; j++) {
        if (number % j == 0) {
            return false;
        }
    }
    return true;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public String reverseString(String str) {
    String reversed = "";
    for (int i = str.length() - 1; i >= 0; i--) {
        reversed += str.charAt(i);
    }
    return reversed;
}
            """,
        "func2":  """
public String flipString(String input) {
    String result = "";
    for (int j = input.length() - 1; j >= 0; j--) {
        result += input.charAt(j);
    }
    return result;
}
            """,
        "label": True,
    },

    {
        "func1":  """
public int sumArray(int[] arr) {
    int sum = 0;
    for (int i : arr) {
        sum += i;
    }
    return sum;
}
            """,
        "func2":  """
public int totalArray(int[] array) {
    int total = 0;
    for (int j : array) {
        total += j;
    }
    return total;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public double findAverage(int[] numbers) {
    int sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    return (double) sum / numbers.length;
}
            """,
        "func2":  """
public double calculateAverage(int[] data) {
    int total = 0;
    for (int value : data) {
        total += value;
    }
    return (double) total / data.length;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public boolean isPalindrome(String word) {
    String reversed = "";
    for (int i = word.length() - 1; i >= 0; i--) {
        reversed += word.charAt(i);
    }
    return word.equals(reversed);
}
            """,
        "func2":  """
public boolean checkPalindrome(String text) {
    String flipped = "";
    for (int j = text.length() - 1; j >= 0; j--) {
        flipped += text.charAt(j);
    }
    return text.equals(flipped);
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public int findFactorial(int n) {
    int factorial = 1;
    for (int i = 1; i <= n; i++) {
        factorial *= i;
    }
    return factorial;
}
            """,
        "func2":  """
public int calculateFactorial(int number) {
    int result = 1;
    for (int j = 1; j <= number; j++) {
        result *= j;
    }
    return result;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public boolean isEven(int num) {
    return num % 2 == 0;
}
            """,
        "func2":  """
public boolean checkEven(int number) {
    return number % 2 == 0;
}
            """,
        "label":  True,
    },

    {
        "func1":  """
public double convertCelsiusToFahrenheit(double celsius) {
    return (celsius * 9/5) + 32;
}
            """,
        "func2":  """
public double celsiusToFahrenheit(double tempCelsius) {
    return (tempCelsius * 9/5) + 32;
}
            """,
        "label":  True,
    },
]

type3_clones = [
    {
        "func1": """
public double calculateArea(double width, double height) {
    return width * height;
}
            """,
        "func2": """
public double computeArea(double length, double breadth) {
    if (length <= 0 || breadth <= 0) {
        throw new IllegalArgumentException("Dimensions must be positive.");
    }
    return length * breadth;
}
            """,
        "label": True,
    },
    {
        "func1": """
public double calculateArea(double width, double height) {
    return width * height;
}
            """,
        "func2": """
public double computeArea(double length, double breadth) {
    if (length <= 0 || breadth <= 0) {
        throw new IllegalArgumentException("Dimensions must be positive.");
    }
    return length * breadth;
}
            """,
        "label": True,
    },

    {
        "func1": """
public int findMax(int a, int b) {
    return (a > b) ? a : b;
}
            """,
        "func2": """
public int getMax(int x, int y) {
    int max = (x > y) ? x : y;
    System.out.println("Max value is: " + max);
    return max;
}
            """,
        "label": True,
    },

    {
        "func1": """
public boolean isPrime(int num) {
    for (int i = 2; i < num; i++) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}
            """,
        "func2": """
public boolean checkPrime(int number) {
    if (number <= 1) {
        return false;
    }
    for (int i = 2; i * i <= number; i++) {
        if (number % i == 0) {
            return false;
        }
    }
    return true;
}
            """,
        "label": True,
    },

    {
        "func1": """
public String reverseString(String str) {
    String reversed = "";
    for (int i = str.length() - 1; i >= 0; i--) {
        reversed += str.charAt(i);
    }
    return reversed;
}
            """,
        "func2": """
public String flipString(String input) {
    if (input == null) {
        return null;
    }
    String result = "";
    for (int j = input.length() - 1; j >= 0; j--) {
        result += input.charAt(j);
    }
    return result;
}
            """,
        "label": True,
    },

    {
        "func1": """
public int sumArray(int[] arr) {
    int sum = 0;
    for (int i : arr) {
        sum += i;
    }
    return sum;
}
            """,
        "func2": """
public int totalArray(int[] array) {
    int total = 0;
    for (int j : array) {
        if (j < 0) {
            throw new IllegalArgumentException("Negative numbers are not allowed.");
        }
        total += j;
    }
    return total;
}
            """,
        "label": True,
    },

    {
        "func1": """
public double findAverage(int[] numbers) {
    int sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    return (double) sum / numbers.length;
}
            """,
        "func2": """
public double calculateAverage(int[] data) {
    if (data.length == 0) {
        throw new IllegalArgumentException("Array must not be empty.");
    }
    int total = 0;
    for (int value : data) {
        total += value;
    }
    return (double) total / data.length;
}
            """,
        "label": True,
    },

    {
        "func1": """
public boolean isPalindrome(String word) {
    String reversed = "";
    for (int i = word.length() - 1; i >= 0; i--) {
        reversed += word.charAt(i);
    }
    return word.equals(reversed);
}
            """,
        "func2": """
public boolean checkPalindrome(String text) {
    String flipped = "";
    for (int j = text.length() - 1; j >= 0; j--) {
        flipped += text.charAt(j);
    }
    return text.equalsIgnoreCase(flipped);
}
            """,
        "label": True,
    },

    {
        "func1": """
public int findFactorial(int n) {
    int factorial = 1;
    for (int i = 1; i <= n; i++) {
        factorial *= i;
    }
    return factorial;
}
            """,
        "func2": """
public int calculateFactorial(int number) {
    if (number < 0) {
        throw new IllegalArgumentException("Factorial is not defined for negative numbers.");
    }
    int result = 1;
    for (int j = 1; j <= number; j++) {
        result *= j;
    }
    return result;
}
            """,
        "label": True,
    },

    {
        "func1": """
public boolean isEven(int num) {
    return num % 2 == 0;
}
            """,
        "func2": """
public boolean checkEven(int number) {
    if (number % 2 == 0) {
        return true;
    } else {
        return false;
    }
}
            """,
        "label": True,
    },

    {
        "func1": """
public double convertCelsiusToFahrenheit(double celsius) {
    return (celsius * 9/5) + 32;
}
            """,
        "func2": """
public double celsiusToFahrenheit(double tempCelsius) {
    if (tempCelsius < -273.15) {
        throw new IllegalArgumentException("Temperature below absolute zero is not possible.");
    }
    return (tempCelsius * 9/5) + 32;
}
            """,
        "label": True,
    },
]

type4_clones = [

        {
            "func1": """
public int factorialIterative(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}
            """,
            "func2": """
public int factorialRecursive(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorialRecursive(n - 1);
    }
}
            """,
            "label": True,
        },

        {
            "func1": """
public int findMaxIterative(int[] array) {
    int max = array[0];
    for (int i : array) {
        if (i > max) {
            max = i;
        }
    }
    return max;
}
            """,
            "func2": """
import java.util.Arrays;

public int findMaxStream(int[] array) {
    return Arrays.stream(array).max().orElse(Integer.MIN_VALUE);
}
            """,
            "label": True,
        },
        {
            "func1": """
public boolean isPalindromeIterative(String str) {
    int start = 0;
    int end = str.length() - 1;
    while (start < end) {
        if (str.charAt(start) != str.charAt(end)) {
            return false;
        }
        start++;
        end--;
    }
    return true;
}
            """,
            "func2": """
public boolean isPalindromeStringBuilder(String str) {
    return str.equals(new StringBuilder(str).reverse().toString());
}
            """,
            "label": True,
        },

        {
            "func1": """
public int sumArrayIterative(int[] array) {
    int sum = 0;
    for (int i : array) {
        sum += i;
    }
    return sum;
}
            """,
            "func2": """
import java.util.Arrays;

public int sumArrayStream(int[] array) {
    return Arrays.stream(array).sum();
}
            """,
            "label": True,
        },

        {
            "func1": """
public String reverseStringIterative(String str) {
    String reversed = "";
    for (int i = str.length() - 1; i >= 0; i--) {
        reversed += str.charAt(i);
    }
    return reversed;
}
            """,
            "func2": """
public String reverseStringStringBuilder(String str) {
    return new StringBuilder(str).reverse().toString();
}
            """,
            "label": True,
        },

        {
            "func1": """
public boolean isPrimeIterative(int num) {
    if (num <= 1) {
        return false;
    }
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}
            """,
            "func2": """
import java.util.stream.IntStream;

public boolean isPrimeStream(int num) {
    return num > 1 && IntStream.rangeClosed(2, (int) Math.sqrt(num)).noneMatch(i -> num % i == 0);
}
            """,
            "label": True,
        },
        {
            "func1": """
public double findAverageIterative(int[] array) {
    int sum = 0;
    for (int i : array) {
        sum += i;
    }
    return (double) sum / array.length;
}
            """,
            "func2": """

import java.util.Arrays;

public double findAverageStream(int[] array) {
    return Arrays.stream(array).average().orElse(0.0);
}
            """,
            "label": True,
        },

        {
            "func1": """
public boolean containsValueIterative(int[] array, int value) {
    for (int i : array) {
        if (i == value) {
            return true;
        }
    }
    return false;
}
            """,
            "func2": """
import java.util.Arrays;

public boolean containsValueStream(int[] array, int value) {
    return Arrays.stream(array).anyMatch(i -> i == value);
}
            """,
            "label": True,
        },
        {
            "func1": """
public void bubbleSort(int[] array) {
    int n = array.length;
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (array[j] > array[j+1]) {
                // swap array[j+1] and array[j]
                int temp = array[j];
                array[j] = array[j+1];
                array[j+1] = temp;
            }
        }
    }
}
            """,
            "func2": """
import java.util.Arrays;
import java.util.Comparator;

public void sortArrayStream(int[] array) {
    int[] sortedArray = Arrays.stream(array)
                                 .boxed()
                                 .sorted(Comparator.naturalOrder())
                                 .mapToInt(i -> i)
                                 .toArray();
    System.arraycopy(sortedArray, 0, array, 0, array.length);
}
            """,
            "label": True,
        },
        {
            "func1": """
import java.util.ArrayList;
import java.util.List;

public List<String> toUpperCaseIterative(List<String> list) {
    List<String> upperCaseList = new ArrayList<>();
    for (String str : list) {
        upperCaseList.add(str.toUpperCase());
    }
    return upperCaseList;
}
            """,
            "func2": """
import java.util.List;
import java.util.stream.Collectors;

public List<String> toUpperCaseStream(List<String> list) {
    return list.stream()
               .map(String::toUpperCase)
               .collect(Collectors.toList());
}
            """,
            "label": True,
        },
]


# for item in type2_clones:
#     func1 = item['func1']
#     func2 = item['func2']
#     print(are_clones(func1, func2))
# print(are_clones(type4_clones[0]['func1'], type4_clones[1]['func1']))


def write_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['content', 'is_clone'])
    df.to_csv(filename, index=False)


data2_not_clones = []
func1 = [x['func1'] for x in type2_clones]
func2 = [x['func2'] for x in type2_clones]
for item in (list(combinations(func1, 2)) + list(combinations(func2, 2)))[:10]:
    func1 = item[0]
    func2 = item[1]
    data = are_clones(func1, func2)
    print(data)
    data2_not_clones.append(data)

write_to_csv(data2_not_clones, 'type2_not_clones.csv')

# data2_clones = []
# for i, item in enumerate(type2_clones):
#     func1 = item['func1']
#     func2 = item['func2']
#     data = are_clones(func1, func2)
#     print(data)
#     data2_clones.append(data)
#
# write_to_csv(data2_clones, 'type2_clones.csv')
#
# data3_clones = []
# for item in type3_clones:
#     func1 = item['func1']
#     func2 = item['func2']
#     data = are_clones(func1, func2)
#     print(data)
#     data3_clones.append(data)
#
# write_to_csv(data3_clones, 'type3_clones.csv')

#
# data4_clones = []
# for item in type4_clones:
#     func1 = item['func1']
#     func2 = item['func2']
#     data = are_clones(func1, func2)
#     print(data)
#     data4_clones.append(data)
#
# write_to_csv(data4_clones, 'type4_clones.csv')


# dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")
#
#
# def make_50_50(dataset, count):
#     true_dataset = dataset.filter(lambda example: example['label'] == True)
#     false_dataset = dataset.filter(lambda example: example['label'] == False)
#     true_examples = true_dataset.select(range(count // 2))
#     false_examples = false_dataset.select(range(count // 2))
#     balanced_examples = concatenate_datasets([true_examples, false_examples])
#     balanced_examples = balanced_examples.shuffle(seed=42)
#     return balanced_examples
#
#
# test = make_50_50(dataset['test'], 20)
#
# data_big_code_bench = []
# for item in test:
#     func1 = item['func1']
#     func2 = item['func2']
#     data = are_clones(func1, func2)
#     print(data)
#     data_big_code_bench.append(data)
#
# write_to_csv(data_big_code_bench, 'big_clone_bench.csv')
