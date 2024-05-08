from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from datasets import load_dataset, concatenate_datasets

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


def print_metrics_my_own_examples(clones_path, not_clones_path):
    clones = pd.read_csv(clones_path)
    print(clones)
    not_clones = pd.read_csv(not_clones_path)
    print(not_clones)

    y_test = [True for _ in range(len(clones))] + [False for _ in range(len(not_clones))]
    y_pred = [(True if x else False) for x in clones['is_clone']] + [(True if x == 'True' else False) for x in not_clones['is_clone']]
    report = classification_report(y_test, y_pred)
    print('CLASSIFICATION REPORT: ')
    print(report)
    matrix = confusion_matrix(y_test, y_pred)
    print('CONFUSION MATRIX:')
    print(matrix)
    return report, matrix


def print_metrics(path, test):
    clones = pd.read_csv(path)
    print(clones)

    y_test = test['label']
    y_pred = [(True if x else False) for x in clones['is_clone']]
    print(y_pred)
    report = classification_report(y_test, y_pred)
    print('CLASSIFICATION REPORT: ')
    print(report)
    matrix = confusion_matrix(y_test, y_pred)
    print('CONFUSION MATRIX:')
    print(matrix)
    return report, matrix



dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")


def make_50_50(dataset, count):
    true_dataset = dataset.filter(lambda example: example['label'] == True)
    false_dataset = dataset.filter(lambda example: example['label'] == False)
    true_examples = true_dataset.select(range(count // 2))
    false_examples = false_dataset.select(range(count // 2))
    balanced_examples = concatenate_datasets([true_examples, false_examples])
    balanced_examples = balanced_examples.shuffle(seed=42)
    return balanced_examples


test = make_50_50(dataset['test'], 20)

def analize(dir):
    print_metrics_my_own_examples(f'/home/pavelresh/college/ml4se/hw_ml4se/hw7/{dir}/type2_clones.csv', '/home/pavelresh/college/ml4se/hw_ml4se/hw7/simple_prompt/type2_not_clones.csv')
    print_metrics_my_own_examples(f'/home/pavelresh/college/ml4se/hw_ml4se/hw7/{dir}/type3_clones.csv', '/home/pavelresh/college/ml4se/hw_ml4se/hw7/simple_prompt/type2_not_clones.csv')
    print_metrics_my_own_examples(f'/home/pavelresh/college/ml4se/hw_ml4se/hw7/{dir}/type4_clones.csv', '/home/pavelresh/college/ml4se/hw_ml4se/hw7/simple_prompt/type2_not_clones.csv')
    print_metrics(f'/home/pavelresh/college/ml4se/hw_ml4se/hw7/{dir}/big_clone_bench.csv', test)

analize('simple_prompt')
analize('big_chunks')
